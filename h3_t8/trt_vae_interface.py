"""Standard H3 VAE tensor interface around an explicit, scoped backend lease.

No TensorRT import, implicit compilation or node registration. The lease factory
must handle Comfy memory reservation before allocating an engine, validate its
identity/profile, and release resources on exit. This module is not that loader.
"""
import threading

from .trt_vae_contract import output_bytes, output_shape, positive_int, required_tile_shapes
from .trt_vae_decode import _spatial_decode, decode_raw_latent, pixel_statistics
from .trt_vae_encode import encode_rgb, required_encode_tiles


class StageInterface:
    def __init__(self, owner, native_stage):
        self.owner, self.native_stage = owner, native_stage

    def __getattr__(self, name):
        return getattr(self.native_stage, name)

    def _finalize_pixels(self, part):
        # Expose the same RGB contract even on Core revisions where finalize
        # lived in VAE.process_output rather than this stage method.
        mean, std = pixel_statistics(self.native_stage.pixel_mean, self.native_stage.pixel_std)
        return (part.float()*std.to(part.device)+mean.to(part.device)).clamp(0,1)

    def _adaptive_decode(self, z):
        """Raw, pre-clamp seven-token chunk required by bounded outpaint decode."""
        import torch
        if not isinstance(z, torch.Tensor) or z.ndim != 5 or z.shape[:3] != (1, 24, 7):
            raise ValueError("TRT raw decode requires one complete seven-token chunk")
        self.owner._check_budget((1, 24, 7, *z.shape[-2:]))
        raw_float_bytes = 1 * 3 * 28 * z.shape[-2] * z.shape[-1] * 16 * 16 * 4
        if raw_float_bytes > self.owner.max_output_bytes:
            raise ValueError("Raw28-frame CPU RGB exceeds the explicit output byte budget")
        if not z.is_floating_point() or not bool(torch.isfinite(z).all()):
            raise ValueError("Raw decoder requires finite floating-point latent")
        with self.owner._operation("raw_decode"):
            shapes = required_tile_shapes(tuple(z.shape))
            with self.owner.open_backend("decoder", shapes) as decode:
                result = _spatial_decode(z.half(), decode, self.owner.check).cpu()
            return result


class H3VAEInterface:
    """VAE-compatible encode/decode and raw-stage routes; allocation is injected."""
    def __init__(self, native_vae, open_backend, *, max_output_bytes, encode_backend="native", check=lambda: None):
        import torch
        if encode_backend not in ("native", "trt"):
            raise ValueError("Encoder must be explicitly native or trt")
        positive_int(max_output_bytes, "max_output_bytes")
        stage = native_vae.first_stage_model
        if tuple(stage.latents_mean.shape) != (24,) or tuple(stage.latents_std.shape) != (24,):
            raise ValueError("Expected native H3 latent normalization")
        self.native_vae, self.open_backend = native_vae, open_backend
        self.max_output_bytes, self.encode_backend, self.check = max_output_bytes, encode_backend, check
        self.first_stage_model = StageInterface(self, stage)
        self.output_device, self.vae_dtype = torch.device("cpu"), torch.float16
        self._lock = threading.Lock()
        self.last_report = {"status": "not_executed"}

    def __getattr__(self, name):
        return getattr(self.native_vae, name)

    def _operation(self, name):
        from contextlib import contextmanager
        @contextmanager
        def operation():
            if not self._lock.acquire(blocking=False):
                raise RuntimeError("This VAE is already executing")
            try:
                self.check()
                yield
                self.check()
                self.last_report = {"status": "complete", "operation": name,
                                    "scope": "Explicit scoped backend; no automatic runtime failure fallback"}
            except BaseException as error:
                self.last_report = {"status": "failed", "operation": name, "error": str(error)}
                raise
            finally:
                self._lock.release()
        return operation()

    def _check_budget(self, shape):
        if output_bytes(shape) > self.max_output_bytes:
            raise ValueError("Decoded CPU RGB exceeds the explicit output byte budget")
        return output_shape(shape)

    def vae_output_dtype(self):
        import torch
        return torch.float32

    def prepare_decode(self, sample_shape, memory_required=None):
        # Actual allocation is deferred to the scoped backend, not this preflight.
        self._check_budget(tuple(sample_shape))
        self.check()
        return 1  # One independent video per backend lease, as in Core's batch hint.

    def decode(self, samples, vae_options=None):
        import torch
        if vae_options:
            raise ValueError("This TRT engine does not implement extra native VAE decode options")
        if not isinstance(samples, torch.Tensor):
            raise ValueError("Standard VAE decode requires a tensor, not a LATENT dictionary")
        self._check_budget(tuple(samples.shape))
        if not samples.is_floating_point() or not bool(torch.isfinite(samples).all()):
            raise ValueError("Decoder requires finite floating-point latent")
        with self._operation("decode"):
            z = samples.detach().to(device="cpu", dtype=torch.float16)
            mean = self.first_stage_model.latents_mean.to(z).view(1, 24, 1, 1, 1)
            std = self.first_stage_model.latents_std.to(z).view(1, 24, 1, 1, 1)
            pixel_mean, pixel_std = pixel_statistics(self.first_stage_model.pixel_mean, self.first_stage_model.pixel_std)
            with self.open_backend("decoder", required_tile_shapes(tuple(z.shape))) as decoder:
                rgb = decode_raw_latent(z * std + mean, decoder, max_output_bytes=self.max_output_bytes, check=self.check,
                                        pixel_mean=pixel_mean, pixel_std=pixel_std)
            # Core VAE returns B,T,H,W,C; VAEDecode/audio_ops flatten to IMAGE.
            # Flattening here loses independent-video boundaries for direct callers.
            return rgb.movedim(1, -1)

    def encode(self, pixels):
        import torch
        if self.encode_backend == "native":
            with self._operation("native_encode"):
                return self.native_vae.encode(pixels)
        if not isinstance(pixels, torch.Tensor) or pixels.ndim not in (4, 5) or pixels.shape[-1] != 3:
            raise ValueError("TRT encoder expects IMAGE [frames,H,W,3] or [batch,frames,H,W,3]")
        if not pixels.is_floating_point() or not bool(torch.isfinite(pixels).all()) or not bool(((pixels >= 0) & (pixels <= 1)).all()):
            raise ValueError("Encoder expects finite IMAGE pixels in [0,1]")
        with self._operation("trt_encode"):
            batches = pixels.detach().cpu().unsqueeze(0) if pixels.ndim == 4 else pixels.detach().cpu()
            # Core's generic crop on 5D input also crops time by the spatial ratio.
            # Apply its spatial crop independently per video; never shorten time.
            image = torch.stack([self.native_vae.vae_encode_crop_pixels(video) for video in batches])
            raw = (image.movedim(-1, 1) * 2 - 1).half()
            # A real T1 profile is required. Do not silently pad a still to17frames.
            shapes = required_encode_tiles(tuple(raw.shape), single_frame_mode="native")
            with self.open_backend("encoder", shapes) as encoder:
                return encode_rgb(raw, encoder, latents_mean=self.first_stage_model.latents_mean,
                                  latents_std=self.first_stage_model.latents_std,
                                  max_output_bytes=self.max_output_bytes, check=self.check, single_frame_mode="native")

    def decode_tiled(self, samples, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        # Native H3 handles its own fixed256px overlap tiling; same alias semantics.
        return self.decode(samples)

    def encode_tiled(self, pixels, tile_x=None, tile_y=None, overlap=None, tile_t=None, overlap_t=None):
        return self.encode(pixels)
