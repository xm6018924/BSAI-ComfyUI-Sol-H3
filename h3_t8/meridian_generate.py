"""Native merged-DMD3NFE generation and complete H264 delivery, independent of dual loops."""

from fractions import Fraction
import json
from pathlib import Path
import time
import uuid

from .meridian_checkpoint_io import atomic_json, canonical_identity, file_sha
from .meridian_media import original_audio_window
from .meridian_plan import audio_policy, source_map
from .meridian_runtime import check, file_identity, store
from .meridian_resources import ResourceTrace, owned_offload, note_cleanup_failure


def fixed_assets(runtime, frames):
    import torch

    if type(frames) is not int or frames < 22 or (frames - 5) % 17:
        raise ValueError(
            "Native Meridian length requires17k+5 frames (k>=1); no global frame-count upper cap"
        )
    root = Path(runtime["source"]) / "assets"
    paths = [root / f"fixed_embed_{frames}.pt", root / f"silence_audio_{frames}.pt"]
    if any(not p.is_file() for p in paths):
        raise ValueError(
            f"Missing frozen native assets for{frames}frames; install correct paired assets, not pad/slice another length"
        )
    identities = [file_identity(p) for p in paths]
    embed, audio = [torch.load(p, map_location="cpu", weights_only=True) for p in paths]
    if (
        not isinstance(embed, dict)
        or "prompt_embeds" not in embed
        or "text_token_tags" not in embed
    ):
        raise ValueError("Invalid fixed raw prompt embedding bundle")
    if embed["prompt_embeds"].ndim != 3 or embed["prompt_embeds"].shape[-1] != 5120:
        raise ValueError(
            "Native Meridian needs raw5120 embedding, not projected Comfy conditioning"
        )
    sound = audio["audio_x0"]
    if (
        not torch.isfinite(sound).all()
        or sound.shape[0] % 2
        or not torch.isfinite(embed["prompt_embeds"]).all()
    ):
        raise ValueError("Invalid paired native audio/embed assets")
    return embed, sound, identities


def own_vae(runtime):
    """External VAE delegates unchanged. Our file loader alone uses qualified static FP32."""
    if runtime["video_vae"] is not None:
        return runtime["video_vae"]
    import torch
    import comfy.sd
    import comfy.utils
    import comfy.model_patcher
    import comfy.model_management as mm

    check()
    vae = comfy.sd.VAE(
        sd=comfy.utils.load_torch_file(runtime["vae"]["path"]), dtype=torch.float32
    )
    if getattr(vae, "latent_channels", None) != 24:
        raise ValueError(
            "Select native H3 video VAE (24 latent channels), not audio/LTX/image VAE"
        )
    vae.first_stage_model.to(dtype=torch.float32)
    vae.patcher = comfy.model_patcher.ModelPatcher(
        vae.first_stage_model,
        load_device=vae.device,
        offload_device=mm.vae_offload_device(),
    )
    vae.disable_offload = True
    try:
        mm.load_models_gpu([vae.patcher], force_full_load=True)
        check()
    except BaseException as error:
        # This VAE has not yet returned to the generation finally block.
        try:
            owned_offload(
                lambda: vae.patcher.unpatch_model(vae.patcher.offload_device),
                "VAE_loading_cancel",
            )
        except BaseException as cleanup_error:
            note_cleanup_failure(error, cleanup_error, "VAE")
        raise
    return vae


def anchor(vae, pixels):
    """Upstream posterior seed42 sample + FP16 round before native normalization."""
    import torch
    import comfy.model_management as mm

    check()
    x = pixels.permute(3, 0, 1, 2)[None].float().div(255)
    mm.load_models_gpu(
        [vae.patcher],
        memory_required=vae.memory_used_encode(x.shape, vae.vae_dtype),
        force_full_load=vae.disable_offload,
    )
    x = vae.process_input(x).to(vae.vae_dtype)
    with torch.inference_mode(), mm.cuda_device_context(vae.device):
        moments = vae.first_stage_model.encode_temporal(x, vae.device).float()
    check()
    mean, logvar = moments.chunk(2, dim=1)
    noise = torch.randn(
        mean.shape,
        generator=torch.Generator().manual_seed(42),
        device="cpu",
        dtype=mean.dtype,
    ).to(mean.device)
    raw = (mean + torch.exp(0.5 * logvar.clamp(-30, 20)) * noise).half().float()
    model = vae.first_stage_model
    value = (
        (raw - model.latents_mean.view(1, -1, 1, 1, 1).to(raw))
        / model.latents_std.view(1, -1, 1, 1, 1).to(raw)
    ).cpu()
    if not torch.isfinite(value).all():
        raise ValueError("Nonfinite native video reference")
    return value


def generate(prepared, seed=1234, audio_mode="silent", output_directory=""):
    """Create the independent job receipt before expensive resource consumption."""
    import folder_paths

    root = Path(
        output_directory or Path(folder_paths.get_output_directory()) / "meridian"
    ).resolve()
    root.mkdir(parents=True, exist_ok=True)
    job = uuid.uuid4().hex
    delivery = dict(
        root=root,
        job=job,
        path=root / f"Meridian_{job}.mp4",
        partial=root / f".Meridian_{job}.partial.mp4",
        report_path=root / f"Meridian_{job}.json",
    )
    report = dict(
        schema="t8.meridian.generation.v1",
        status="running",
        job=job,
        current_stage="resource_verification",
        published=False,
        human_qualified=False,
    )
    started = time.monotonic()
    atomic_json(delivery["report_path"], report)
    trace = ResourceTrace("generation")
    try:
        with trace:
            value = _generate(prepared, seed, audio_mode, delivery)
        report = json.loads(delivery["report_path"].read_text(encoding="utf8"))
        report.update(
            wall_seconds=time.monotonic() - started,
            current_stage="complete",
            resources=trace.report(),
        )
        atomic_json(delivery["report_path"], report, replace_existing=True)
        return value[0], value[1], json.dumps(report, ensure_ascii=False)
    except BaseException as error:
        report = json.loads(delivery["report_path"].read_text(encoding="utf8"))
        report.update(
            status="interrupted"
            if type(error).__name__
            in ("InterruptedError", "InterruptProcessingException")
            else "failed",
            error=f"{type(error).__name__}: {error}",
            wall_seconds=time.monotonic() - started,
            resources=trace.report(),
        )
        atomic_json(delivery["report_path"], report, replace_existing=True)
        raise


def generation_stage(delivery, stage, records=None):
    report = json.loads(delivery["report_path"].read_text(encoding="utf8"))
    report["current_stage"] = stage
    if records is not None:
        report["stages"] = records
    atomic_json(delivery["report_path"], report, replace_existing=True)


def _generate(prepared, seed, audio_mode, delivery):
    import torch
    import comfy.sd
    import comfy.sample
    from comfy.nested_tensor import NestedTensor
    from comfy.patcher_extension import WrappersMP
    from comfy_api.latest import InputImpl, Types
    from .sampling import setup_dual_clock_sampling

    plan, material = prepared["plan"], prepared["material"]
    runtime = dict(material["runtime"])
    # Generation, not camera authoring, consumes these potentially huge payloads.
    runtime["checkpoint"] = file_identity(runtime["checkpoint"]["path"])
    if runtime["vae"] is not None:
        runtime["vae"] = file_identity(runtime["vae"]["path"])
    policy = audio_policy(plan, audio_mode)
    if audio_mode == "source_1to1" and material["video"] is None:
        raise ValueError("Original audio requires VIDEO material and1:1 time")
    embed, audio, assets = fixed_assets(runtime, plan["frames"])
    if type(seed) is not int or seed < 0 or seed >= 2**64:
        raise ValueError("Seed must be an unsigned64-bit integer")
    cache = store(runtime)
    records = [material["receipt"], prepared["receipt"]]
    generation_stage(delivery, "VAE_load", records)
    vae = own_vae(runtime)
    try:
        generation_stage(delivery, "encode", records)
        encode_contract = dict(
            schema="t8.meridian.encode.v1",
            warp=prepared["identity"],
            vae=runtime["vae"],
            posterior_seed=42,
            compute="static_FP32_existing_file_weights"
            if runtime["vae"]
            else "external_unchanged",
            half_round_before_normalization=True,
        )
        references, receipt = cache.run(
            "encode",
            encode_contract,
            lambda: [anchor(vae, prepared[k]) for k in ("condition", "render")],
            portable=runtime["vae"] is not None,
        )
        records.append(receipt)
        n, (w, h) = plan["frames"], prepared["canvas"]
        t = (n - 5) // 17 * 5 + 2
        cw, ch = prepared["condition_canvas"]
        if any(
            tuple(ref.shape) != (1, 24, t, ch // 16, cw // 16) for ref in references
        ):
            raise ValueError("Source/warp native reference shape mismatch")
        base = runtime["model"]
        patch_warnings = list(runtime["warnings"])

        def sampling():
            check()
            patcher = (
                base
                if base is not None
                else comfy.sd.load_diffusion_model(
                    runtime["checkpoint"]["path"],
                    model_options={"dtype": torch.bfloat16},
                )
            )
            if not hasattr(patcher.model.diffusion_model, "named_modules"):
                raise ValueError("MODEL does not contain a native diffusion model")
            if base is None:
                from comfy_kitchen.tensor import QuantizedTensor

                layers = [
                    m
                    for _, m in patcher.model.diffusion_model.named_modules()
                    if isinstance(getattr(m, "weight", None), QuantizedTensor)
                ]
                if len(layers) != 250 or any(
                    m.weight._qdata.dtype != torch.int8 or not m.weight._params.convrot
                    for m in layers
                ):
                    raise ValueError(
                        "Actual loaded Meridian model does not have250 native ConvRot INT8 layers"
                    )
            latent = {
                "samples": NestedTensor(
                    (
                        torch.zeros(1, 24, t, h // 16, w // 16),
                        torch.zeros(1, 32, 2, audio.shape[0] // 2),
                    )
                )
            }
            positive = [
                [
                    embed["prompt_embeds"],
                    dict(
                        minimax_token_tags=embed["text_token_tags"],
                        minimax_refs=[
                            dict(
                                kind="video",
                                latent=r,
                                latent_t=t,
                                latent_h=ch // 16,
                                latent_w=cw // 16,
                                ref_audio_t=0,
                            )
                            for r in references
                        ],
                        minimax_visual_cond_noise_aug=0.999,
                    ),
                ]
            ]
            # Clone by the original scheduler, preserve all foreign patches/wrappers.
            model, sampler, sigmas = setup_dual_clock_sampling(
                patcher, latent, 3, 3.0, 3.0, "euler", "native_flow"
            )
            expected = torch.tensor([1.0, 0.8571428060531616, 0.5999999642372131, 0.0])
            if not torch.allclose(sigmas.cpu(), expected, atol=1e-7, rtol=0):
                raise ValueError("Native DMD sigma contract changed")
            calls, callbacks = [], []

            def count(executor, *args, **kwargs):
                check()
                result = executor(*args, **kwargs)
                calls.append(len(calls))
                check()
                return result

            model.add_wrapper_with_key(
                WrappersMP.DIFFUSION_MODEL, "t8-meridian-" + uuid.uuid4().hex, count
            )

            def callback(step, *args):
                check()
                callbacks.append(step)

            # Keep any native condition augmentation RNG private; no global reseed leak.
            devices = [torch.cuda.current_device()] if torch.cuda.is_available() else []
            with torch.random.fork_rng(devices=devices), torch.inference_mode():
                torch.manual_seed(seed)
                noise = comfy.sample.prepare_noise(latent["samples"], seed)
                sampled = comfy.sample.sample_custom(
                    model,
                    noise,
                    1.0,
                    sampler,
                    sigmas,
                    positive,
                    positive,
                    latent["samples"],
                    callback=callback,
                    disable_pbar=False,
                    seed=seed,
                )
            check()
            if len(calls) != 3 or callbacks != [0, 1, 2]:
                raise ValueError(
                    "Native3-forward generation contract not met; preserve failure, do not silently rerun"
                )
            video, sound = sampled.unbind()
            if not torch.isfinite(video).all() or not torch.isfinite(sound).all():
                raise ValueError("Nonfinite generated native AV")
            return dict(
                video=video.cpu(),
                audio=sound.cpu(),
                forward_boundaries=len(calls),
                callbacks=callbacks,
                sigmas=sigmas.cpu().tolist(),
                foreign_stack_qualified=False,
                native_base_checked=base is None,
            )

        # Opaque MODEL or VAE can change without filename identity: never persist false portability.
        sample_contract = dict(
            schema="t8.meridian.sample.v1",
            encode=canonical_identity(encode_contract),
            checkpoint=runtime["checkpoint"],
            assets=assets,
            seed=seed,
            recipe=runtime["metadata"]["t8_recipe"],
            implementation=runtime["implementation"],
            torch_version=str(torch.__version__),
        )
        generation_stage(delivery, "sample", records)
        sampled, receipt = cache.run(
            "sample",
            sample_contract,
            sampling,
            portable=base is None and runtime["vae"] is not None,
        )
        records.append(receipt)
        generation_stage(delivery, "decode", records)
        check()
        images = vae.decode(sampled["video"])[0].cpu()
        if tuple(images.shape) != (n, h, w, 3) or not torch.isfinite(images).all():
            raise ValueError("Final native VAE decoded dimensions/frame count differ")
        # Generation never delivers the internal model silence prior as an invented soundtrack.
        sound = (
            original_audio_window(material["video"], source_map(plan)[0], n, check)
            if audio_mode == "source_1to1"
            else None
        )
        job, path, partial, report_path = (
            delivery[k] for k in ("job", "path", "partial", "report_path")
        )
        report = dict(
            schema="t8.meridian.generation.v1",
            status="running",
            job=job,
            plan=plan,
            stages=records,
            policy=policy,
            warnings=patch_warnings,
            published=False,
            human_qualified=False,
            current_stage="delivery_full_decode",
            recipe=runtime["metadata"]["t8_recipe"],
            external_MODEL_preserved=base is not None,
            external_VAE_preserved=runtime["video_vae"] is not None,
        )
        atomic_json(report_path, report, replace_existing=True)
        try:
            check()
            InputImpl.VideoFromComponents(
                Types.VideoComponents(
                    images=images, frame_rate=Fraction(24), audio=sound
                )
            ).save_to(
                str(partial),
                format=Types.VideoContainer.MP4,
                codec=Types.VideoCodec.H264,
            )
            check()
            import av

            with av.open(str(partial)) as container:
                stream = container.streams.video[0]
                if stream.codec_context.name != "h264" or (
                    stream.width,
                    stream.height,
                ) != (w, h):
                    raise ValueError("H264 MP4 output dimensions/codec mismatch")
                pts = []
                for frame in container.decode(video=0):
                    check()
                    pts.append(Fraction(frame.pts) * Fraction(frame.time_base))
                if len(pts) != n or any(
                    b - a != Fraction(1, 24) for a, b in zip(pts, pts[1:])
                ):
                    raise ValueError("Incomplete final MP4 or non24fps delivery")
            with av.open(str(partial)) as container:
                if audio_mode == "silent" and container.streams.audio:
                    raise ValueError("Silent delivery unexpectedly contains audio")
                decoded_audio = 0
                if audio_mode == "source_1to1":
                    if not container.streams.audio:
                        raise ValueError("Original audio delivery missing")
                    for frame in container.decode(audio=0):
                        check()
                        decoded_audio += frame.samples
                    if not decoded_audio:
                        raise ValueError("Empty delivered audio")
            sha = file_sha(partial, cancelled=lambda: (check(), False)[1])
            check()
            # Our UUID destination only, no overwrite of user media.
            import os

            os.link(partial, path)
            partial.unlink()
            report.update(
                status="complete_full_AV_decoded_not_human",
                path=str(path),
                sha256=sha,
                frames=n,
                dimensions=[w, h],
                decoded_audio_samples=decoded_audio,
                forward_boundaries=sampled["forward_boundaries"],
                actual_new_forward_boundaries=0
                if receipt["reused"]
                else sampled["forward_boundaries"],
                callbacks=sampled["callbacks"],
            )
            atomic_json(report_path, report, replace_existing=True)
            return (
                InputImpl.VideoFromFile(str(path)),
                str(path),
                json.dumps(report, ensure_ascii=False),
            )
        except BaseException as error:
            report.update(
                status="interrupted"
                if type(error).__name__
                in ("InterruptedError", "InterruptProcessingException")
                else "failed",
                error=f"{type(error).__name__}: {error}",
            )
            atomic_json(report_path, report, replace_existing=True)
            raise
        finally:
            if partial.exists():
                partial.unlink()
    finally:
        if runtime["video_vae"] is None:
            # Only independent VAE residency; no user model or globally shared VAE mutation.
            import sys

            pending = sys.exc_info()[1]
            try:
                owned_offload(
                    lambda: vae.patcher.unpatch_model(vae.patcher.offload_device),
                    "VAE_finish",
                )
            except BaseException as cleanup_error:
                if pending is None:
                    raise
                note_cleanup_failure(pending, cleanup_error, "VAE")
