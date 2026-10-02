"""Actual Core Conv VAE decode of new refined latents, retaining original H3 AAC."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time

from backend_files import sha


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    path = parser.parse_args().request
    root = path.parent
    request = json.loads(path.read_text())
    report = {"status": "incomplete", "scope": __doc__}
    vae = torch = None
    handle = None
    def progress(stage):
        value = {"stage": stage, "time": time.time()}
        (root / "worker-live.json").write_text(json.dumps(value))
        print(json.dumps(value), flush=True)
    try:
        assert request["schema"] == "t8-ltx-refined-decode-v1"
        for filename, digest in request["identities"].items():
            assert sha(filename) == digest, filename
        sys.path.insert(0, request["core"])
        import comfy.cli_args
        comfy.cli_args.args.cpu = True  # explicit isolated VAE CUDA device only below
        import torch as torch_module
        torch = torch_module
        import comfy.sd
        import comfy.utils
        from safetensors.torch import load_file, save_file
        from PIL import Image, ImageDraw
        from ltx_refined_media import deliver
        torch.set_num_threads(2)
        assert str(torch.cuda.get_device_properties(0).uuid).removeprefix("GPU-").lower() == request["gpu_uuid"].removeprefix("GPU-").lower()
        z = load_file(request["latent"], device="cpu")["video"]
        geometry = request['geometry']
        frames, width, height = (geometry[k] for k in ('frames', 'width', 'height'))
        if geometry['fps'] != 24 or (frames - 1) % 8 or not 9 <= frames <= 192:
            raise ValueError('Prepared decode requires exact8n+1 frames at24fps, up to8s')
        assert z.shape == (1, 128, (frames - 1) // 8 + 1, height // 32, width // 32) and z.dtype == torch.bfloat16 and torch.isfinite(z).all()
        progress("loading_existing_Core_LTX_Conv_VAE")
        state, metadata = comfy.utils.load_torch_file(request["vae"], return_metadata=True)
        vae = comfy.sd.VAE(sd=state, metadata=metadata, device=torch.device("cuda:0"), dtype=torch.bfloat16)
        del state
        vae.throw_exception_if_invalid()
        assert vae.latent_channels == 128
        calls = []
        handle = vae.first_stage_model.decoder.register_forward_pre_hook(lambda _m, args: calls.append(list(args[0].shape)))
        progress("decoding_prepared_refined_frames_tiled")
        start = time.perf_counter()
        with torch.inference_mode():
            pixels = vae.decode_tiled(z, tile_x=16, tile_y=16, overlap=4, tile_t=8, overlap_t=2).cpu()
        torch.cuda.synchronize()
        handle.remove()
        handle = None
        assert pixels.shape == (1, frames, height, width, 3) and torch.isfinite(pixels).all() and calls
        report["decode_seconds"] = time.perf_counter() - start
        save_file({"pixels": pixels.contiguous()}, str(root / "refined-rgb.safetensors"))
        # Release only this isolated model before CPU encoding/media audit.
        vae.first_stage_model.to_empty(device="meta")
        vae = None
        gc.collect()
        torch.cuda.empty_cache()
        progress("mux_original_AAC_and_verify_packets")
        movie = root / "h3-learned2x-ltx-refined-original-audio.mp4"
        report["media"] = deliver(pixels[0], request["original_video"], movie, frame_count=frames)
        contact = Image.new("RGB", (1536, 556), "#121a24")
        draw = ImageDraw.Draw(contact)
        for n, index in enumerate(round(i * (frames - 1) / 5) for i in range(6)):
            x, y = n % 3 * 512, n // 3 * 278
            draw.text((x + 5, y + 4), f"Refined frame {index} - human review pending", fill="white")
            image = Image.fromarray(pixels[0, index].mul(255).round().to(torch.uint8).numpy())
            contact.paste(image.resize((512, 256)), (x, y + 22))
        contact.save(root / "contact.png")
        for filename, digest in request["identities"].items():
            assert sha(filename) == digest, filename
        report.update(status="real_H3_x2_LTX_refined_media_pass_pending_review", decoder_calls=calls,
            video_sha256=sha(movie), rgb_sha256=sha(root / "refined-rgb.safetensors"),
            original_audio_retained=True, ltx_generated_audio_discarded=True,
            limits="Actual three-update upstream joint refiner from dequantized INT8 base in BF16, original audio packets retained. Tiled Core VAE; no original-BF16/Spark/untiled/quality/speed parity claim.")
    except BaseException as exc:
        report.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        active_error = sys.exc_info()[0] is not None
        try:
            if handle is not None:
                handle.remove()
            if vae is not None:
                vae.first_stage_model.to_empty(device="meta")
            gc.collect()
            if torch is not None:
                torch.cuda.empty_cache()
        except BaseException as exc:
            report["cleanup_error"] = f"{type(exc).__name__}: {exc}"
            if not active_error:
                report.update(status="failed", error=report["cleanup_error"])
                raise
        finally:
            (root / "report.json").write_text(json.dumps(report, indent=2))
            print(json.dumps({"status": report["status"]}), flush=True)


if __name__ == "__main__":
    main()
