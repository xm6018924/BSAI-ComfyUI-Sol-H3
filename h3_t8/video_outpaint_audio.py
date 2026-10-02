"""Bounded native H3 audio posterior encoding, not final audio synthesis.

The CNN uses aligned waveform halos. The posterior attention sees the complete
causal prefix via temporary disk KV blocks, never independent per-video windows.
RAM/VRAM tensor sizes are bounded; prefix attention still has quadratic compute
and the scratch files grow linearly. Callers own model loading and GPU leases.
"""
from __future__ import annotations

import hashlib
import math
from contextlib import closing
from pathlib import Path
import tempfile

import torch
import torch.nn.functional as F
from safetensors.torch import load, save_file


def _check_tensor(tensor, label):
    if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point() or not torch.isfinite(tensor).all():
        raise ValueError(f"{label} must be a finite floating-point tensor")


def causal_prefix_attention(query, query_start, read_kv, *, block_tokens=64, interrupt_check=None):
    """Online-softmax SDPA with absolute causal positions and bounded KV reads."""
    _check_tensor(query, "query")
    if query.ndim != 4 or min(query.shape) < 1:
        raise ValueError("query must be nonempty [batch, heads, tokens, channels]")
    if isinstance(query_start, bool) or not isinstance(query_start, int) or query_start < 0:
        raise ValueError("query_start must be nonnegative")
    if isinstance(block_tokens, bool) or not isinstance(block_tokens, int) or not 1 <= block_tokens <= 128:
        raise ValueError("block_tokens must be in [1, 128]")
    if query.shape[-2] > block_tokens:
        raise ValueError("query exceeds the block budget")
    q = query.float()
    maximum = torch.full((*q.shape[:-1], 1), -torch.inf, device=q.device)
    denominator = torch.zeros_like(maximum)
    accumulator = torch.zeros_like(q)
    positions = torch.arange(query_start, query_start + q.shape[-2], device=q.device)
    stop = query_start + q.shape[-2]
    for start in range(0, stop, block_tokens):
        if interrupt_check:
            interrupt_check()
        end = min(start + block_tokens, stop)
        key, value = read_kv(start, end)
        for tensor in (key, value):
            _check_tensor(tensor, "KV block")
            if tensor.shape != (*query.shape[:2], end-start, query.shape[-1]):
                raise ValueError("KV block shape differs from query")
        scores = (q @ key.to(device=q.device, dtype=torch.float32).transpose(-1, -2)) / math.sqrt(q.shape[-1])
        allowed = torch.arange(start, end, device=q.device)[None, :] <= positions[:, None]
        scores.masked_fill_(~allowed, -torch.inf)
        new_maximum = torch.maximum(maximum, scores.amax(dim=-1, keepdim=True))
        rescale = torch.exp(maximum - new_maximum)
        probability = torch.exp(scores - new_maximum)
        accumulator = accumulator * rescale + probability @ value.to(device=q.device, dtype=torch.float32)
        denominator = denominator * rescale + probability.sum(dim=-1, keepdim=True)
        maximum = new_maximum
    result = (accumulator / denominator).to(query.dtype)
    _check_tensor(result, "attention result")
    return result


def _encoder_halo(stage):
    from comfy.ldm.minimax import audio_vae as native

    if (getattr(stage, "sample_rate", None), getattr(stage, "hop_length", None)) != (32000, 800):
        raise ValueError("audio encoder requires native H3 32000Hz/800-hop architecture")
    if not isinstance(stage.encoder, native.Encoder) or not isinstance(stage.pre_block, native.AttnProjection):
        raise ValueError("audio encoder requires native H3 encoder and posterior head")
    allowed = (native.Encoder, native.EncoderBlock, native.ResidualUnit, native.Snake1d,
               torch.nn.Sequential, torch.nn.Conv1d)
    if any(not isinstance(module, allowed) for module in stage.encoder.modules()):
        raise ValueError("audio encoder contains an unsupported temporal operation")
    expected = [(7, 1, 1, 3)]
    for stride in (2, 4, 4, 5, 5):
        for dilation in (1, 3, 9):
            expected.extend(((7, 1, dilation, 3*dilation), (1, 1, 1, 0)))
        expected.append((2*stride, stride, 1, math.ceil(stride/2)))
    expected.append((3, 1, 1, 1))
    convs = [module for module in stage.encoder.modules() if isinstance(module, torch.nn.Conv1d)]
    actual = [(m.kernel_size[0], m.stride[0], m.dilation[0], m.padding[0]) for m in convs]
    if actual != expected or any(m.padding_mode != "zeros" for m in convs):
        raise ValueError("audio encoder receptive-field contract changed")
    left = right = 0
    jump = 1
    for kernel, stride, dilation, pad in actual:
        left -= pad * jump
        right += ((kernel-1)*dilation-pad)*jump
        jump *= stride
    if jump != 800:
        raise ValueError("audio encoder hop changed")
    return math.ceil(max(-left, right)/jump)


def _qkv(attention, value):
    import comfy.ops

    weight, _, stream = comfy.ops.cast_bias_weight(attention.qkv, value, offloadable=True)
    try:
        bias = torch.cat((attention.q_bias, attention.zero_k_bias, attention.v_bias))
        projected = F.linear(value, weight, comfy.ops.cast_to_input(bias, value))
    finally:
        comfy.ops.uncast_bias_weight(attention.qkv, weight, None, stream)
    return projected.reshape(value.shape[0], value.shape[1], 3, attention.num_heads,
                             attention.head_dim).permute(2, 0, 3, 1, 4).unbind(0)


def iter_encode_outpaint_audio(stage, read_samples, sample_count, *, device="cpu", dtype=torch.float32,
                               block_tokens=64, scratch_parent=None, interrupt_check=None):
    """Yield (start_token, CPU float32 [1,32,2,T]) for one source shot.

    read_samples(start, stop) returns exact CPU [1,2,stop-start] canonical 32kHz
    stereo samples. It must be file-backed/bounded in production. All original
    samples, including a fractional final token, are encoded before frame-aligned
    model windows select observed tokens. Cancellation discards temporary KV;
    durable output commits and source/model revalidation belong to the caller.
    """
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    if isinstance(block_tokens, bool) or not isinstance(block_tokens, int) or not 1 <= block_tokens <= 128:
        raise ValueError("block_tokens must be in [1, 128]")
    if dtype != torch.float32:
        raise ValueError("native H3 audio encoding currently requires float32")
    halo = _encoder_halo(stage)
    total = math.ceil(sample_count/800)
    head = stage.pre_block
    # The caller must close this generator on cancellation; runtime adapters use
    # contextlib.closing. Temporary files are never persistent checkpoint inputs.
    with tempfile.TemporaryDirectory(prefix="t8-outpaint-audio-kv-", dir=scratch_parent) as temporary:
        blocks = {}

        def read_kv(start, stop):
            path, digest = blocks[start]
            payload = path.read_bytes()
            if hashlib.sha256(payload).hexdigest() != digest:
                raise ValueError("temporary audio KV block changed")
            data = load(payload)
            if data["key"].shape[-2] != stop-start:
                raise ValueError("temporary audio KV block has the wrong length")
            return data["key"], data["value"]

        for start in range(0, total, block_tokens):
            if interrupt_check:
                interrupt_check()
            stop = min(total, start+block_tokens)
            left, right = max(0, start-halo), min(total, stop+halo)
            sample_start, sample_stop = left*800, min(sample_count, right*800)
            wave = read_samples(sample_start, sample_stop)
            _check_tensor(wave, "source waveform")
            if wave.shape != (1, 2, sample_stop-sample_start) or wave.device.type != "cpu":
                raise ValueError("source waveform reader must return exact CPU stereo bounds")
            with torch.inference_mode():
                wave = F.pad(wave.to(device=device, dtype=dtype), (0, right*800-sample_stop))
                encoded = stage.encoder(wave.reshape(2, 1, -1))
                if encoded.shape[-1] != right-left:
                    raise ValueError("native audio CNN returned an unexpected token count")
                features = encoded[..., start-left:stop-left].transpose(1, 2).contiguous()
                query, key, value = _qkv(head.attn, head.norm1(features))
                path = Path(temporary) / f"kv-{start:012d}.safetensors"
                save_file({"key": key.to("cpu").contiguous(), "value": value.to("cpu").contiguous()}, str(path))
                blocks[start] = (path, hashlib.sha256(path.read_bytes()).hexdigest())
                attended = causal_prefix_attention(query, start, read_kv, block_tokens=block_tokens,
                                                     interrupt_check=interrupt_check)
                attended = head.attn.proj(F.adaptive_avg_pool1d(attended.mean(dim=1), head.attn.out_dim))
                posterior = head.proj(head.norm3(features)) + attended
                posterior = posterior + head.mlp(head.norm2(posterior))
                latent = stage.mean_proj(posterior.transpose(1, 2))
                mean = stage.latents_mean.to(latent).view(1, -1, 1)
                std = stage.latents_std.to(latent).view(1, -1, 1)
                if torch.any(std <= 0):
                    raise ValueError("audio latent normalizer must have positive standard deviations")
                latent = (latent-mean)/std
                _check_tensor(latent, "encoded audio")
                if latent.shape != (2, 32, stop-start):
                    raise ValueError("audio posterior shape differs from native H3")
                output = latent.reshape(1, 2, 32, stop-start).permute(0, 2, 1, 3).to("cpu").contiguous()
                del wave, encoded, features, query, key, value, attended, posterior, latent
            yield start, output


def iter_encode_outpaint_audio_vae(vae, read_samples, sample_count, *, block_tokens=64,
                                   scratch_parent=None, interrupt_check=None):
    """Use Comfy's device/offload lifecycle without its independent tiled fallback.

    The source coordinator owns the serial GPU lease and loaded-state fingerprint.
    This adapter never unloads all models or mutates VAE settings. An OOM remains
    explicit; retrying independent audio tiles would lose the causal prefix.
    """
    import comfy.model_management as management

    vae.throw_exception_if_invalid()
    if vae.vae_dtype != torch.float32:
        raise ValueError("native H3 audio VAE must execute in float32")
    if isinstance(block_tokens, bool) or not isinstance(block_tokens, int) or not 1 <= block_tokens <= 128:
        raise ValueError("block_tokens must be in [1, 128]")
    if isinstance(sample_count, bool) or not isinstance(sample_count, int) or sample_count < 1:
        raise ValueError("sample_count must be a positive integer")
    halo = _encoder_halo(vae.first_stage_model)
    if interrupt_check:
        interrupt_check()
    waveform_budget = min(math.ceil(sample_count/800), block_tokens+halo*2)*800
    with management.cuda_device_context(vae.device):
        memory = vae.memory_used_encode((1, 2, waveform_budget), vae.vae_dtype)
        management.load_models_gpu([vae.patcher], memory_required=memory, force_full_load=vae.disable_offload)
        with closing(iter_encode_outpaint_audio(
            vae.first_stage_model, read_samples, sample_count, device=vae.device, dtype=vae.vae_dtype,
            block_tokens=block_tokens, scratch_parent=scratch_parent, interrupt_check=interrupt_check,
        )) as chunks:
            yield from chunks


def loaded_audio_vae_identity(vae, *, interrupt_check=None):
    """Fingerprint actual loaded state plus the audio-specific execution contract."""
    from .video_outpaint_source_runtime import loaded_video_vae_identity
    from .video_outpaint_plan import canonical

    _encoder_halo(vae.first_stage_model)
    # Reuse the bounded tensor/buffer/patch verification, not a model-name label.
    # The common state report also hashes video adapter files, causing harmless
    # conservative invalidation if those files change; no old cache is relabeled.
    state = loaded_video_vae_identity(vae, interrupt_check=interrupt_check)
    contract = {"loaded_state_sha256": state.get('execution_weight_sha256', state["sha256"]), "sample_rate": 32000, "hop_length": 800,
                "audio_encoder_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "audio_file_sha256": hashlib.sha256(Path(__file__).with_name("video_outpaint_audio_file.py").read_bytes()).hexdigest(),
                "audio_decode_sha256": hashlib.sha256(Path(__file__).with_name("video_outpaint_audio_decode.py").read_bytes()).hexdigest(),
                "posterior": "native_cnn_halo13_disk_causal_prefix_fp32_v1"}
    result = {"schema": "t8.h3.outpaint.loaded_audio_vae_identity/v1",
            "sha256": hashlib.sha256(canonical(contract).encode()).hexdigest(),
            "contract": contract, "tensor_count": state["tensor_count"], "tensor_bytes": state["tensor_bytes"],
            "max_copy_bytes": state["max_copy_bytes"], "model_filename_trusted": False}
    if state.get('portable_cache_reuse') is False:
        result.update(sha256=state['sha256'], portable_cache_reuse=False,
                      execution_selection=state['execution_selection'], opaque_internal_state_verified=False)
    return result
