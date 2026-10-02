"""Ordered requests through one owned native runtime, independent of loading/UI."""

import time
import sys

from taomate_stream_assembly import NativeStreamAssembly


def execute_stream(
    model, pipeline, runtime, prepared, *, interrupt, progress, save_request, torch
):
    if not prepared:
        raise ValueError("A prepared stream cannot be empty")
    assembly = NativeStreamAssembly()
    records, calls, handles = [], [], []
    current = [0]

    def completed(index):
        calls.append(index)
        if index == len(model.blocks) - 1:
            progress(
                "streaming_forward",
                request_index=current[0],
                total_forwards=len(calls) // len(model.blocks),
            )

    try:
        if len(model.blocks) != 50:
            raise ValueError("Pinned Tao H3 runtime requires50 native blocks")
        for index, block in enumerate(model.blocks):
            handles.append(
                block.register_forward_hook(lambda _m, _a, _o, i=index: completed(i))
            )
        for index, item in enumerate(prepared):
            current[0] = index
            interrupt()
            began, before = time.perf_counter(), len(calls)
            output = pipeline.generate(
                video_noise_seed=item["video_seed"],
                audio_noise_seed=item["audio_seed"],
                width=864,
                height=480,
                precomputed_text=(item["text"]["hidden"], item["text"]["tags"]),
                denoise_loop=runtime.run,
            )
            torch.cuda.synchronize()
            interrupt()
            video = output.video_latents.detach().cpu().contiguous()
            audio = output.audio_latents.detach().cpu().contiguous()
            if video.shape != (1, 24, 37, 30, 54):
                raise ValueError(
                    "Native request output differs from864x480 transport geometry"
                )
            if len(runtime.executions) != index + 1:
                raise ValueError("Runtime did not record exactly one ordered request")
            execution = runtime.executions[index]
            forwards = execution.denoise_forwards + execution.clean_forwards
            if (
                execution.denoise_forwards != execution.phase_count * 3
                or execution.clean_forwards != execution.phase_count
                or calls[before:] != list(range(50)) * forwards
            ):
                raise ValueError(
                    "Native stream forward schedule or block dispatch changed"
                )
            if (
                not execution.audio_teacher["published_clean_audio_exact_match"]
                or execution.attention_backend["backend"] != "torch_CUDA_SDPA_not_FA3"
            ):
                raise ValueError("Unexpected teacher binding or attention dispatch")
            assembly.append(video, audio, item["known_audio"], execution)
            output_digest = save_request(index, video, audio)
            records.append(
                dict(
                    index=index,
                    seconds=time.perf_counter() - began,
                    execution=execution.to_dict(),
                    output_sha256=output_digest,
                    clean_audio_bitexact=True,
                    transport_prefix_matches_previous=True,
                    weight_block_calls=len(runtime.last_weight_receipt["block_calls"]),
                )
            )
            progress("request_latents_saved", request_index=index)
        if (
            len(runtime.request_owner.records) != len(prepared)
            or runtime._native_frame_offset != assembly.native_frames
        ):
            raise ValueError("Final request owner and assembled timeline differ")
        receipt = dict(
            requests=records,
            owner_records=list(runtime.request_owner.records),
            timing=assembly.timing(),
            dit_timing=runtime.dit_timing_receipt(),
            host_cache_retention=list(runtime._cache.host.retention_receipts),
        )
        return assembly.joined(), receipt
    finally:
        # Cleanup is scoped to this run; an exception never leaves a reusable KV
        # owner or a success record for a partial request sequence.
        original = sys.exc_info()[1]
        errors = []
        for handle in handles:
            try:
                handle.remove()
            except BaseException as error:
                errors.append(f"hook removal: {type(error).__name__}: {error}")
        try:
            runtime.release_retained_state()
        except BaseException as error:
            errors.append(f"runtime release: {type(error).__name__}: {error}")
        if errors:
            if original is None:
                raise RuntimeError("Owned stream cleanup failed: " + "; ".join(errors))
            add_note = getattr(original, "add_note", None)
            if callable(add_note):
                add_note("Owned stream cleanup also failed: " + "; ".join(errors))
            else:
                print(
                    "Owned stream cleanup also failed: " + "; ".join(errors),
                    file=sys.stderr,
                )
