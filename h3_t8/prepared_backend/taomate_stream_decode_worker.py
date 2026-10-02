"""Decode a joined native Tao stream with one global video/audio timing map.

No repeated segment PCM, sampling, volume normalization or seam postprocessing.
Uses the pinned upstream global endpoint/audio timing filters, not per-segment
retiming. Native floating outputs are diagnostic artifacts, not user delivery.
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from backend_files import sha


def run(request, root, report, progress):
    sys.path[:0] = [request["core"], str(Path(request["source"]) / "src")]
    import comfy.cli_args

    comfy.cli_args.args.cpu = True
    import torch
    import comfy.sd
    import comfy.utils
    import numpy as np
    import soundfile as sf
    from safetensors.torch import load_file
    from safetensors import safe_open
    from taomate_stream_timing import validate_stream_timing
    from taomate_h3.inference.media_timing import (
        endpoint_preserving_video_filter,
        exact_audio_delivery_filter,
    )

    torch.set_num_threads(2)
    assert (
        str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-").lower()
        == request["gpu_uuid"].removeprefix("GPU-").lower()
    )
    if request.get("schema") != "t8-taomate-stream-decode-v1":
        raise ValueError("Expected native stream decode schema")
    with safe_open(request["latent"], framework="pt", device="cpu") as src:
        metadata = src.metadata() or {}
    if (
        metadata.get("schema") != "t8-taomate-stream-latents-v1"
        or metadata.get("normalization") != "upstream normalized latents"
    ):
        raise ValueError("Unqualified stream latent schema/normalization")
    timing = validate_stream_timing(
        json.loads(metadata["timing"]), request["request_count"]
    )
    if (metadata.get("width"), metadata.get("height")) != ("864", "480"):
        raise ValueError("Stream latent geometry changed")
    nf, pf = timing["native_frames"], timing["published_frames"]
    ns, ps = timing["native_samples"], timing["published_samples"]
    seconds = pf // 24
    tensors = load_file(request["latent"])
    video, audio = tensors["video"], tensors["audio"]
    assert video.shape == (1, 24, timing["video_latents"], 30, 54) and audio.shape == (
        2,
        32,
        timing["audio_latents"],
    )
    assert torch.isfinite(video).all() and torch.isfinite(audio).all()
    stages = []
    for kind, path, z, dtype, shape in (
        ("video", request["video_vae"], video, torch.float16, (1, 3, nf, 480, 864)),
        (
            "audio",
            request["audio_vae"],
            audio.permute(1, 0, 2).unsqueeze(0).contiguous(),
            torch.float32,
            (1, 2, ns),
        ),
    ):
        progress("loading_" + kind + "_VAE")
        state, metadata = comfy.utils.load_torch_file(path, return_metadata=True)
        vae = comfy.sd.VAE(
            sd=state, metadata=metadata, device=torch.device("cpu"), dtype=dtype
        )
        del state
        vae.throw_exception_if_invalid()
        model = vae.first_stage_model.eval().to("cuda:0")
        if kind == "video":
            assert (
                vae.latent_channels == 24
                and vae.handles_tiling
                and model.decode_output_shape(z.shape) == shape
            )
        else:
            assert vae.latent_channels == 32 and vae.audio_sample_rate == 32000
        try:
            progress("decoding_joined_" + kind)
            torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            with torch.inference_mode():
                decoded = (
                    model.decode(z.to(device="cuda:0", dtype=dtype)).detach().cpu()
                )
            torch.cuda.synchronize()
            assert decoded.shape == shape and torch.isfinite(decoded).all(), (
                kind,
                decoded.shape,
            )
            stages.append(
                dict(
                    kind=kind,
                    seconds=time.perf_counter() - start,
                    peak_allocated=torch.cuda.max_memory_allocated(),
                    shape=list(decoded.shape),
                    reverse_normalization_count=1,
                )
            )
            if kind == "video":
                assert decoded.min() >= 0 and decoded.max() <= 1
                rgb = decoded[0].permute(1, 2, 3, 0).contiguous()
            else:
                pcm = decoded[0].T.contiguous().numpy()
                sf.write(root / "native-audio.wav", pcm, 32000, subtype="FLOAT")
        finally:
            model.cpu()
            del model, vae
    video_filter = endpoint_preserving_video_filter(
        native_frames=nf, published_frames=pf
    )
    audio_filter = exact_audio_delivery_filter(
        native_samples=ns, published_samples=ps, sample_rate=32000
    )
    # Preserve completed decode diagnostics even if delivery fails afterwards.
    report.update(stages=stages, video_filter=video_filter, audio_filter=audio_filter,
                  native_audio_sha256=sha(root / "native-audio.wav"))
    destination = root / "tao-stream.mp4"
    command = [
        "ffmpeg",
        "-nostdin",
        "-v",
        "error",
        "-xerror",
        "-n",
        "-f",
        "rawvideo",
        "-pixel_format",
        "rgb24",
        "-video_size",
        "864x480",
        "-framerate",
        "24",
        "-i",
        "pipe:0",
        "-i",
        str(root / "native-audio.wav"),
        "-map",
        "0:v:0",
        "-map",
        "1:a:0",
        "-vf",
        video_filter,
        "-af",
        audio_filter,
        "-frames:v",
        str(pf),
        "-t",
        str(seconds),
        "-c:v",
        "libx264",
        "-threads:v",
        "1",
        "-preset",
        "fast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    progress("publishing_joint_timeline_H264_AAC")
    with (root / "encode.log").open("xb") as log:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=log,
            stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        try:
            for frame in rgb:
                process.stdin.write(
                    frame.mul(255).round().to(torch.uint8).numpy().tobytes()
                )
            process.stdin.close()
            code = process.wait(timeout=180)
            if code:
                raise RuntimeError(f"Joint delivery ffmpeg failed ({code}); retain encode.log")
        except BrokenPipeError as error:
            # A crashed encoder must not be confused with invalid model output.
            try:
                code = process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                code = None
            report["encoder_failure"] = {"exit_code": code, "error": str(error)}
            raise RuntimeError(f"Joint delivery ffmpeg closed its input ({code}); retain encode.log") from error
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(destination),
            "-f",
            "null",
            "-",
        ],
        check=True,
        capture_output=True,
        timeout=120,
    )
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_streams",
                "-show_format",
                "-of",
                "json",
                str(destination),
            ],
            text=True,
        )
    )
    v = next(s for s in probe["streams"] if s["codec_type"] == "video")
    a = next(s for s in probe["streams"] if s["codec_type"] == "audio")
    assert (
        v["width"],
        v["height"],
        int(v["nb_frames"]),
        v["r_frame_rate"],
        v["codec_name"],
        v["pix_fmt"],
    ) == (864, 480, pf, "24/1", "h264", "yuv420p")
    assert (
        float(v["duration"]) == float(a["duration"]) == seconds
        and int(a["sample_rate"]) == 32000
        and a["channels"] == 2
    )
    samples = subprocess.check_output(
        [
            "ffmpeg",
            "-nostdin",
            "-v",
            "error",
            "-xerror",
            "-i",
            str(destination),
            "-map",
            "0:a:0",
            "-af",
            f"atrim=end_sample={ps}",
            "-c:a",
            "pcm_f32le",
            "-f",
            "f32le",
            "-",
        ],
        timeout=90,
    )
    decoded_pcm = np.frombuffer(samples, dtype="<f4").reshape(-1, 2)
    assert decoded_pcm.shape == (ps, 2) and np.isfinite(decoded_pcm).all()
    report.update(
        status="native_Tao_stream_decode_pass_pending_review",
        stages=stages,
        video_filter=video_filter,
        audio_filter=audio_filter,
        timing=timing,
        video_sha256=sha(destination),
        native_audio_sha256=sha(root / "native-audio.wav"),
        delivery_pcm_rms=np.sqrt(
            np.mean(decoded_pcm.astype("float64") ** 2, axis=0)
        ).tolist(),
        delivery_pcm_peak=float(np.abs(decoded_pcm).max()),
        strict_full_AV_decode=True,
        source_audio_repeated=False,
        sampler_forwards=0,
        human_qualified=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    path = parser.parse_args().request
    root = path.parent
    request = json.loads(path.read_text(encoding="utf8"))
    report = dict(status="incomplete", human_qualified=False)

    def progress(stage):
        row = dict(stage=stage, time=time.time())
        (root / "worker-live.json").write_text(json.dumps(row))
        print(json.dumps(row), flush=True)

    try:
        for name, expected in request["identities"].items():
            assert sha(name) == expected, name
        run(request, root, report, progress)
        for name, expected in request["identities"].items():
            assert sha(name) == expected, name
        report["postflight_pass"] = True
    except BaseException as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        raise
    finally:
        (root / "report.json").write_text(json.dumps(report, indent=2), encoding="utf8")


if __name__ == "__main__":
    main()
