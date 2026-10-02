"""CPU-only H3 decoder geometry. No TensorRT import or implicit GPU operation."""

from dataclasses import dataclass


def positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


@dataclass(frozen=True)
class TemporalPlan:
    tokens: int
    padding: int
    windows: int
    frames: int
    padded_frames: int

    @property
    def tail_frames(self):
        return self.padded_frames - self.frames


def temporal_plan(tokens):
    """Match H3's 17-frame / 5-token schedule, NOT a uniform 4*T-3 rule."""
    positive_int(tokens, "tokens")
    if tokens == 1:
        # A real one-token decoder call, not seven repeated tokens.
        return TemporalPlan(1, 0, 1, 1, 1)
    padding = (-tokens - 3) % 5
    windows = (tokens + padding + 3) // 5 - 1
    if windows < 1:
        padding += 5
        windows += 1
    quotient, remainder = divmod(tokens, 5)
    frames = 17 * quotient + (0, 1, 5, 9, 13)[remainder]
    return TemporalPlan(tokens, padding, windows, frames, 17 * windows + 5)


@dataclass(frozen=True)
class TileAxis:
    starts: tuple[int, ...]
    lengths: tuple[int, ...]
    overlaps: tuple[int, ...]


def tile_axis(pixels):
    """Native H3 256px tiles, >=64px overlap, 16px lattice; no edge padding."""
    positive_int(pixels, "pixels")
    if pixels % 16:
        raise ValueError("Decoded spatial dimensions must be multiples of 16")
    if pixels <= 256:
        return TileAxis((0,), (pixels,), ())
    count = (pixels + 255) // 256
    while count * 256 - (count - 1) * 64 < pixels:
        count += 1
    overlaps = [64] * (count - 1)
    remaining = (count * 256 - sum(overlaps) - pixels) // 16
    for i in range(remaining):
        overlaps[i % len(overlaps)] += 16
    starts = [0]
    for overlap in overlaps:
        starts.append(starts[-1] + 256 - overlap)
    return TileAxis(tuple(starts), (256,) * count, tuple(overlaps))


def output_shape(latent_shape):
    shape = tuple(latent_shape)
    if len(shape) != 5:
        raise ValueError("Expected H3 latent [B,24,T,H,W]")
    for value in shape:
        positive_int(value, "latent dimension")
    batch, channels, tokens, height, width = shape
    if channels != 24:
        raise ValueError("H3 video latent must have 24 channels")
    return batch, 3, temporal_plan(tokens).frames, height * 16, width * 16


def output_bytes(latent_shape):
    size = 4  # float32 RGB, before any serialization / encoder buffers
    for dim in output_shape(latent_shape):
        size *= dim
    return size


def required_tile_shapes(latent_shape):
    """All single-batch raw backend calls required; no added spatial padding."""
    _, _, _, height, width = output_shape(latent_shape)
    tokens = 1 if latent_shape[2] == 1 else 7
    return tuple(sorted({(1, 24, tokens, h // 16, w // 16)
                         for h in tile_axis(height).lengths
                         for w in tile_axis(width).lengths}))


def validate_declared_input(declared_shape, required_shapes):
    """Reject ONNX static-shape conflicts BEFORE engine building/execution.

    Symbolic dimensions are not proof an engine supports them; the runtime must
    independently verify its optimization profile and resolved output shape.
    """
    declared = tuple(declared_shape)
    if len(declared) != 5:
        raise ValueError("Decoder ONNX must declare five input dimensions")
    for dim in declared:
        if not isinstance(dim, str) or not dim:
            positive_int(dim, "declared dimension")
    for shape in required_shapes:
        output_shape(shape)
        for axis, (fixed, requested) in enumerate(zip(declared, shape)):
            if isinstance(fixed, int) and fixed != requested:
                raise ValueError(f"ONNX axis {axis} is fixed at {fixed}, but needs {requested}; "
                                 "requires a matching export/profile, not implicit padding")
