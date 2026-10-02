"""First five-second native Tao request using saved teacher and host-resident KV.

Writes normalized latents, not a finished video. VAE decoding follows only after
this worker exits and releases full CPU/GPU state. No multi-request test.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
from backend_files import sha


def _memory_snapshot():
    """Diagnostic process/system readings, not peak attribution or a safety gate."""
    try:
        import psutil
        memory = psutil.Process().memory_info()
        return {'process_rss_bytes': memory.rss,
                'process_private_bytes': getattr(memory, 'private', None),
                'system_available_bytes': psutil.virtual_memory().available}
    except Exception as error:
        return {'unavailable': f'{type(error).__name__}: {error}'}


def _generate(request, root, report, progress, source):
    """Owned model/tensor frame; output is on disk, report contains receipts only."""
    sys.path.insert(0,str(source/'src'))
    import psutil
    import torch
    from safetensors.torch import save_file
    from taomate_h3.model.dit import MiniMaxH3DiT
    from taomate_h3.model.weight_loading import load_minimax_h3_dit_weights
    from taomate_h3.model.layers import set_inference_packed_dense_flash_attn3
    from taomate_h3.inference.lora_checkpoint import apply_h3_lora_checkpoint,materialize_h3_lora_bf16_buffers_
    from taomate_input_preflight import load_inputs
    from taomate_local_transport import local_transport
    from taomate_local_runtime import make_local_runtime
    from taomate_prepared_pipeline import prepared_pipeline
    torch.set_num_threads(4)
    text, prompt, teacher, known, report['prepared_teacher'] = load_inputs(request, torch)
    assert str(torch.cuda.get_device_properties(0).uuid).removeprefix('GPU-').lower() == request['gpu_uuid'].removeprefix('GPU-').lower()
    verified = json.loads(Path(request['download_receipt']).read_text())
    assert verified['status'] == 'download_and_transformer_checksums_pass'
    assert psutil.virtual_memory().available > verified['total_bytes'] + 24*1024**3
    for item in verified['files']:
        assert (Path(request['base'])/item['file']).stat().st_size == item['size']
    set_inference_packed_dense_flash_attn3(False)
    with local_transport() as context, torch.inference_mode():
        progress('loading_official_CPU_base')
        start = time.perf_counter()
        model = MiniMaxH3DiT.allocate('meta',parallel_context=context).eval()
        dtypes = {n:p.dtype for n,p in model.named_parameters()}
        model.to_empty(device='cpu')
        report['base_load'] = load_minimax_h3_dit_weights(model,Path(request['base'])/'FL2VA/transformer')
        assert all(p.dtype == dtypes[n] and p.device.type == 'cpu' for n,p in model.named_parameters())
        report['loading_seconds'] = time.perf_counter()-start
        progress('installing_real_Tao_adapter')
        report['adapter'] = apply_h3_lora_checkpoint(model,Path(request['adapter']))
        report['adapter_buffers'] = materialize_h3_lora_bf16_buffers_(model)
        pipeline = prepared_pipeline(model,context)
        runtime = make_local_runtime(teacher,minimum_free_bytes=2*1024**3)
        calls = []
        def note(index):
            calls.append(index)
            if index == 49:
                progress('streaming_forward',complete_forwards=len(calls)//50)
        handles = [b.register_forward_hook(lambda _m,_a,_o,index=i: note(index)) for i,b in enumerate(model.blocks)]
        try:
            progress('preparing_and_generating_one_request')
            start = time.perf_counter()
            generated = pipeline.generate(video_noise_seed=request['video_seed'],audio_noise_seed=request['audio_seed'],width=864,height=480,
                precomputed_text=(text['hidden'],text['tags']),denoise_loop=runtime.run)
            torch.cuda.synchronize()
            report['generation_seconds'] = time.perf_counter()-start
            video,audio = generated.video_latents.cpu(),generated.audio_latents.cpu()
            assert video.shape == (1,24,37,30,54) and torch.isfinite(video).all()
            assert audio.shape == known.shape and torch.equal(audio,known)
            assert len(runtime.executions) == 1
            execution = runtime.executions[0]
            assert calls == list(range(50))*(execution.denoise_forwards+execution.clean_forwards)
            assert execution.denoise_forwards == execution.phase_count*3
            assert execution.clean_forwards == execution.phase_count
            assert execution.audio_teacher['published_clean_audio_exact_match']
            assert execution.attention_backend['backend'] == 'torch_CUDA_SDPA_not_FA3'
            assert execution.attention_backend['cache_transfer_bytes'] > 0
            report['host_cache_retention'] = list(runtime._cache.host.retention_receipts)
            assert report['host_cache_retention'] and all(item['layers'] == 50 for item in report['host_cache_retention'])
            progress('saving_normalized_latents')
            output = root/'tao-normalized-latents.safetensors'
            save_file({'video':video.contiguous(),'audio':audio.contiguous()},str(output),metadata={
                'scope':'one native request; no VAE decode or human acceptance','normalization':'upstream normalized latents',
                'width':'864','height':'480','audio_noise_seed':str(request['audio_seed']),'video_noise_seed':str(request['video_seed'])})
            progress('building_generation_receipt')
            report.update(status='first_Tao_request_latents_pass',execution=execution.to_dict(),
                timing=runtime.dit_timing_receipt(),weight_block_calls=len(runtime.last_weight_receipt['block_calls']),
                output_sha256=sha(output),video_shape=list(video.shape),audio_shape=list(audio.shape),
                teacher_reused_without_rerun=True,clean_audio_bitexact=True)
        finally:
            for handle in handles:
                handle.remove()
            runtime.release_retained_state()
            assert runtime._cache is None
            report['retained_state_released'] = True
            progress('generation_cache_released')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--request', type=Path, required=True)
    request_path = parser.parse_args().request
    root = request_path.parent
    request = json.loads(request_path.read_text())
    report = {'status':'incomplete','scope':__doc__}
    def progress(stage, **extra):
        value = dict(stage=stage,time=time.time(),memory=_memory_snapshot(),**extra)
        (root/'worker-live.json').write_text(json.dumps(value,indent=2))
        print(json.dumps(value),flush=True)
    try:
        assert request['schema'] == 't8-taomate-prepared-first-request-v1'
        for seed_name in ('video_seed', 'audio_seed'):
            if type(request.get(seed_name)) is not int or not 0 <= request[seed_name] < 2**64:
                raise ValueError('Prepared Tao seeds must be unsigned64 integers')
        for path,value in request['identities'].items():
            assert sha(path) == value, path
        source = Path(request['source'])
        assert subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip() == request['source_revision']
        assert not subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip()
        cpu = json.loads(Path(request['cpu_receipt']).read_text())
        assert cpu['status'] == 'full_official_DiT_packed_CPU_forward_pass'
        assert cpu['weight_offload']['output_bitexact'] and not cpu['cuda_initialized']
        assert cpu['parameters'] == 33122992896
        _generate(request, root, report, progress, source)
        # The isolated generation frame must end before input re-hashing.
        # Collect cyclic model/pipeline owners too; no sampling or tensor edits.
        import gc
        gc.collect()
        progress('generation_scope_released')
        report['generation_scope_released_before_postflight'] = True
        progress('postflight_input_hashes')
        for path,value in request['identities'].items():
            assert sha(path) == value,path
        progress('postflight_complete')
    except BaseException as error:
        report.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:
        (root/'report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps({'status':report['status']}),flush=True)

if __name__ == '__main__':
    main()
