"""Bounded native-H3 source encoding and explicit isotropic canvas transforms.

No per-window independent source encode: immutable shot-global token chunks are
produced once and are intended to be shared by all overlapping sample windows.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .video_outpaint_plan import validate_outpaint_plan, _integer


def _rgb(frames, height, width):
    if not isinstance(frames, torch.Tensor) or frames.ndim != 4 or frames.shape[0] < 1 or frames.shape[1:3] != (height, width) or frames.shape[-1] < 3:
        raise ValueError("source reader must return IMAGE frames of the planned geometry")
    if frames.dtype == torch.uint8:
        return frames[..., :3].float() / 255
    rgb = frames[..., :3]
    if not rgb.is_floating_point() or not torch.isfinite(rgb).all() or torch.any((rgb < 0) | (rgb > 1)):
        raise ValueError("source RGB must be uint8 or normalized finite floats")
    return rgb.float()


def _grid(width, height, *, mapped_width, mapped_height, scale, offset, device, inverse):
    x = torch.arange(width, device=device, dtype=torch.float32) + 0.5
    y = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    if inverse:
        x, y = (x - offset[0]) / scale, (y - offset[1]) / scale
    else:
        x, y = x * scale + offset[0], y * scale + offset[1]
    xx = (x / mapped_width * 2 - 1)[None, :].expand(height, width)
    yy = (y / mapped_height * 2 - 1)[:, None].expand(height, width)
    return torch.stack((xx, yy), dim=-1)


def source_to_sampling_canvas(frames, plan):
    checked = validate_outpaint_plan(plan)
    source, model = checked["source"], checked["sampling"]
    rgb = _rgb(frames, source["height"], source["width"])
    scale = model["isotropic_scale"]
    left, top = checked["output"]["source_rect"][:2]
    offset = [model["offset"][0] + left * scale, model["offset"][1] + top * scale]
    grid = _grid(model["width"], model["height"], mapped_width=source["width"], mapped_height=source["height"],
                 scale=scale, offset=offset, device=rgb.device, inverse=True)
    return F.grid_sample(rgb.movedim(-1, 1), grid[None].expand(rgb.shape[0], -1, -1, -1),
                         mode="bilinear", padding_mode="border", align_corners=False).movedim(1, -1)


def sampling_to_output_canvas(frames, plan):
    checked = validate_outpaint_plan(plan)
    output, model = checked["output"], checked["sampling"]
    rgb = _rgb(frames, model["height"], model["width"])
    grid = _grid(output["width"], output["height"], mapped_width=model["width"], mapped_height=model["height"],
                 scale=model["isotropic_scale"], offset=model["offset"], device=rgb.device, inverse=False)
    return F.grid_sample(rgb.movedim(-1, 1), grid[None].expand(rgb.shape[0], -1, -1, -1),
                         mode="bilinear", padding_mode="border", align_corners=False).movedim(1, -1)


def iter_encode_outpaint_source(video_vae, read_frames, plan, *, shot_index=0, start_token=0, interrupt_check=None):
    """Yield (shot-global token start, CPU latent chunk, report), at most five tokens.

    H3 encodes independent 17-frame clips into five tokens and drops only the last
    three tokens of the full trajectory. Each public encode call gets 22 frames:
    the real/padded 17 plus five sacrificial tail frames. Its first five tokens
    retain the native chunk; only the final shot chunk is reduced to two. This
    avoids changing VAE attributes or its model-management lifecycle. It repeats
    one padded internal clip per call: bounded memory with a compute tradeoff.

    Grid-aligned native-scale sources use source-only encoding like the reference.
    Other geometries use an explicitly mapped edge-extended model canvas; they
    are not claimed numerically identical to the reference's cropped geometry.
    """
    checked = validate_outpaint_plan(plan)
    _integer(shot_index, "shot_index", 0, len(checked["shots"]) - 1)
    native = getattr(video_vae, "first_stage_model", None)
    if any(getattr(native, name, None) != value for name, value in {
        "clip_length": 17, "token_drop": 3, "tokens_chunk_size": 5, "vae_ratio": 16,
    }.items()):
        raise ValueError("source preparation requires the native H3 17-frame/5-token VAE contract")
    shot, source, sampling = checked["shots"][shot_index], checked["source"], checked["sampling"]
    rect = checked["output"]["source_rect"]
    source_only = sampling["isotropic_scale"] == 1 and sampling["offset"] == [0, 0] and all(
        value % 32 == 0 for value in (source["width"], source["height"], rect[0], rect[1])
    )
    total_tokens = (shot["aligned_frames"] - 5) // 17 * 5 + 2
    _integer(start_token, "start_token", 0, total_tokens-1)
    if start_token % 5:
        raise ValueError("source preparation resume must start at a five-token chunk boundary")
    chunks = (shot["aligned_frames"] + 16) // 17
    for chunk in range(start_token // 5, chunks):
        if interrupt_check:
            interrupt_check()
        start = shot["start"] + chunk * 17
        stop = min(start + 17, shot["stop"])
        # A shot can need an entirely held final clip when alignment extends beyond its last frame.
        read_start = min(start, shot["stop"] - 1)
        frames = read_frames(read_start, max(read_start + 1, stop))
        if frames.shape[0] != max(read_start + 1, stop) - read_start:
            raise ValueError("source reader returned the wrong frame count")
        rgb = _rgb(frames, source["height"], source["width"])
        rgb = rgb if source_only else source_to_sampling_canvas(rgb, checked)
        pixels = torch.cat((rgb, rgb[-1:].expand(22 - rgb.shape[0], -1, -1, -1)))
        encoded = video_vae.encode(pixels)
        eh, ew = pixels.shape[1] // 16, pixels.shape[2] // 16
        if not isinstance(encoded, torch.Tensor) or tuple(encoded.shape) != (1, 24, 7, eh, ew) or not torch.isfinite(encoded).all():
            raise ValueError("H3 source encoder violated the expected 22-frame/7-token shape or finiteness")
        count = min(5, total_tokens - chunk * 5)
        source_chunk = encoded[:, :, :count].detach().to(device="cpu", dtype=torch.float32)
        result = torch.zeros((1, 24, count, sampling["height"] // 16, sampling["width"] // 16))
        x0, y0, x1, y1 = sampling["source_lock_latent_box"]
        if source_only:
            result[:, :, :, y0:y1, x0:x1] = source_chunk
        else:
            result[:, :, :, y0:y1, x0:x1] = source_chunk[:, :, :, y0:y1, x0:x1]
        yield chunk * 5, result, {
            "plan_sha256": checked["plan_sha256"], "shot_index": shot_index,
            "token_start": chunk * 5, "token_stop": chunk * 5 + count,
            "read_start": read_start, "read_stop": max(read_start + 1, stop),
            "public_vae_input_frames": 22, "geometry_mode": "reference_source_only" if source_only else "isotropic_edge_canvas",
            "source_pixels_delivered_from_this_latent": False, "gpu_quality_verified": False,
        }


class SequentialOutpaintFrameReader:
    """CPU file reader retaining at most a requested 17-frame batch and one held frame.

    Open only after strict source inspection and revalidate file identity after
    preparation. Forward skips support resume; a previous final frame may be
    reread for edge-hold padding, but arbitrary backwards seeking is forbidden.
    """
    def __init__(self, source_path, plan, *, interrupt_check=None):
        self.path = source_path
        self.plan = validate_outpaint_plan(plan)
        self.interrupt_check = interrupt_check
        self.container = None
        self.iterator = None
        self.next_frame = 0
        self.last = None

    def __enter__(self):
        import av
        self.container = av.open(str(self.path), mode="r")
        try:
            if len(self.container.streams.video) != 1:
                raise ValueError("outpaint reader requires exactly one source video stream")
            stream = self.container.streams.video[0]
            stream.codec_context.thread_count = 1
            self.iterator = iter(self.container.decode(stream))
            return self
        except BaseException:
            self.container.close()
            self.container = None
            raise

    def __exit__(self, *_):
        if self.container is not None:
            self.container.close()
        self.container = self.iterator = self.last = None

    def __call__(self, start, stop):
        source = self.plan["source"]
        _integer(start, "read start", 0, source["frames"]-1)
        _integer(stop, "read stop", start+1, source["frames"])
        if stop-start > 17:
            raise ValueError("source reader accepts at most 17 frames per request")
        if self.container is None:
            raise RuntimeError("source frame reader is not open")
        if start < self.next_frame-1:
            raise ValueError("source reader does not allow arbitrary backwards reads")
        frames = [self.last] if start == self.next_frame-1 else []
        while self.next_frame < stop:
            if self.interrupt_check:
                self.interrupt_check()
            try:
                frame = next(self.iterator)
            except StopIteration as error:
                raise ValueError("source ended before the planned frame range") from error
            if (frame.width, frame.height) != (source["width"], source["height"]):
                raise ValueError("source geometry changed during preparation")
            if self.next_frame >= start:
                pixels = torch.from_numpy(frame.to_ndarray(format="rgb24"))
                self.last = pixels
                frames.append(pixels)
            self.next_frame += 1
        if len(frames) != stop-start:
            raise ValueError("source reader failed to satisfy the exact requested frame range")
        return torch.stack(frames)
