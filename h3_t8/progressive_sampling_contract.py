"""Contracts for the opt-in H3 progressive first-sampling route.

No Comfy imports, model loading, global patches or external SelfLift source.
This module does not execute a sampler. The Euler/rectified-flow identities are
used as independent boundary oracles before connecting the runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math

import torch


SCHEMA = "t8.h3.progressive_first_sample.plan.v1"


def _integer(value, name, minimum=1):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_number(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    return float(value)


@dataclass(frozen=True)
class ProgressivePlan:
    schema: str
    target_width: int
    target_height: int
    low_width: int
    low_height: int
    requested_low_scale: float
    sigmas: tuple[float, ...]
    sigma_dtype: str
    low_evaluations: int
    high_evaluations: int
    video_shape: tuple[int, ...]
    audio_shape: tuple[int, ...]
    task: str

    @property
    def prediction_sigma(self):
        return self.sigmas[self.low_evaluations - 1]

    @property
    def resume_sigma(self):
        return self.sigmas[self.low_evaluations]

    @property
    def total_evaluations(self):
        return self.low_evaluations + self.high_evaluations

    def report(self):
        contract = asdict(self)
        encoded = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return {
            **contract,
            "plan_sha256": hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            "status": "planned_not_executed",
            "total_evaluations": self.total_evaluations,
            "prediction_sigma": self.prediction_sigma,
            "resume_sigma": self.resume_sigma,
            "video_spatial_token_ratio": self.low_width * self.low_height / (self.target_width * self.target_height),
            "audio_policy": "continue_native_sampler_space_not_completed_audio_lock",
            "quality": "unverified",
            "runtime_identity": "not_bound",
        }


def plan_progressive_first_sample(video, audio, sigmas, *, low_evaluations,
                                  low_scale=0.5, task="t2va", noise_mask=None):
    """Validate an empty joint AV template and a complete native Euler ladder.

    This is deliberately not an img2img/two-pass/partial-sampling contract. An
    I2VA reference belongs to conditioning, not to nonzero initial samples.
    Model, conditioning, learned weights and Core identity bind at runtime.
    """
    return _plan_progressive(video, audio, sigmas, low_evaluations=low_evaluations,
                             low_scale=low_scale, task=task, noise_mask=noise_mask, initialized=False)


def plan_progressive_initialized_sample(video, audio, sigmas, *, low_evaluations,
                                        low_scale=0.5, task='t2va'):
    """Explicit experimental initialization; a partial ladder may start below1.

    Mask normalization and native clean-source binding are runtime obligations.
    At sigma1 an unmasked initial latent is intentionally erased by native flow.
    This does not change the strict empty-first-sample contract above.
    """
    return _plan_progressive(video, audio, sigmas, low_evaluations=low_evaluations,
                             low_scale=low_scale, task=task, noise_mask=None, initialized=True)


def _plan_progressive(video, audio, sigmas, *, low_evaluations, low_scale, task,
                      noise_mask, initialized):
    if task not in ("t2va", "i2va"):
        raise ValueError("Initial progressive scope supports only t2va and i2va")
    if noise_mask is not None:
        raise ValueError("noise_mask is not supported by progressive first sampling")
    for tensor, prefix, ndim, name in ((video, (1, 24), 5, "video"), (audio, (1, 32, 2), 4, "audio")):
        if not isinstance(tensor, torch.Tensor) or tensor.ndim != ndim or tuple(tensor.shape[:len(prefix)]) != prefix:
            raise ValueError(f"Invalid batch-1 H3 {name} shape")
        if not tensor.is_floating_point() or any(n <= 0 for n in tensor.shape):
            raise ValueError(f"{name} must be a nonempty floating tensor")
        if initialized and not bool(torch.isfinite(tensor).all()):
            raise ValueError(f'{name} source must be finite')
        if not initialized and (not bool(torch.isfinite(tensor).all()) or bool(torch.count_nonzero(tensor))):
            raise ValueError(f"{name} must be a finite all-zero initial template, not a completed first pass")
    temporal = video.shape[2]
    if temporal < 2 or (temporal - 2) % 5:
        raise ValueError("video time must follow the H3 5n+2 latent grid")
    frames = 5 + ((temporal - 2) // 5) * 17
    expected_audio = round(frames * 40 / 24)
    if audio.shape[-1] != expected_audio:
        raise ValueError(f"audio time must match video: expected {expected_audio}")
    width, height = video.shape[-1] * 16, video.shape[-2] * 16
    if width % 32 or height % 32:
        raise ValueError("target canvas must be divisible by 32")
    scale = _finite_number(low_scale, "low_scale")
    if not 0.25 <= scale < 1:
        raise ValueError("low_scale must be in [0.25, 1); use native sampling for a baseline")
    low_width = max(32, round(width * scale / 32) * 32)
    low_height = max(32, round(height * scale / 32) * 32)
    if low_width >= width or low_height >= height:
        raise ValueError("Aligned low canvas must shrink both axes")
    if not isinstance(sigmas, torch.Tensor) or sigmas.ndim != 1 or not sigmas.is_floating_point():
        raise ValueError("sigmas must be a one-dimensional floating tensor")
    if not 3 <= sigmas.numel() <= 1001:
        raise ValueError("Complete progressive sampling requires 2 to 1000 evaluations")
    values = tuple(sigmas.detach().to(device="cpu", dtype=torch.float64).tolist())
    if not all(math.isfinite(x) for x in values):
        raise ValueError("sigmas must be finite")
    if initialized and (not 0 < values[0] <= 1 or values[-1] != 0):
        raise ValueError('Initialized sigmas must start in (0,1] and end at0')
    if not initialized and (values[0] != 1 or values[-1] != 0):
        raise ValueError("Complete first-sample sigmas must start at 1 and end at 0")
    if any(a <= b for a, b in zip(values, values[1:])):
        raise ValueError("sigmas must strictly descend")
    low = _integer(low_evaluations, "low_evaluations")
    total = len(values) - 1
    if low >= total:
        raise ValueError("At least one high-resolution evaluation must remain")
    return ProgressivePlan(SCHEMA, width, height, low_width, low_height, scale,
                           values, str(sigmas.dtype), low, total - low,
                           tuple(video.shape), tuple(audio.shape), task)


def _matching_tensors(left, right):
    if not isinstance(left, torch.Tensor) or not isinstance(right, torch.Tensor):
        raise ValueError("Boundary inputs must be tensors")
    if left.shape != right.shape or left.device != right.device or left.dtype != right.dtype:
        raise ValueError("Boundary tensors must share shape, device and dtype")
    if not left.is_floating_point() or not bool(torch.isfinite(left).all()) or not bool(torch.isfinite(right).all()):
        raise ValueError("Boundary tensors must be finite floating tensors")


def euler_sampler_space_step(state, prediction, sigma, sigma_next):
    """One Euler interval in native sampler space, not raw audio/VAE space."""
    _matching_tensors(state, prediction)
    sigma = _finite_number(sigma, "sigma")
    sigma_next = _finite_number(sigma_next, "sigma_next")
    if not 0 <= sigma_next < sigma <= 1:
        raise ValueError("Euler interval must satisfy 0 <= next < sigma <= 1")
    return state + (state - prediction) * ((sigma_next - sigma) / sigma)


def rectified_flow_state(clean, noise, sigma, *, noise_scale=1.0):
    """CONST flow marginal; caller supplies MODEL-space clean and matching noise."""
    _matching_tensors(clean, noise)
    sigma = _finite_number(sigma, "sigma")
    scale = _finite_number(noise_scale, "noise_scale")
    if not 0 <= sigma <= 1 or scale <= 0:
        raise ValueError("Flow requires sigma in [0,1] and positive noise_scale")
    return clean * (1 - sigma) + noise * (sigma * scale)


class EvaluationLedger:
    """Keep Euler callback counts distinct from actual DiT forward counts."""

    def __init__(self, plan: ProgressivePlan):
        self.expected = {"low": plan.low_evaluations, "high": plan.high_evaluations}
        self.callbacks = {"low": 0, "high": 0}
        self.forwards = {"low": 0, "high": 0}

    def record(self, stage, *, forward=False):
        if stage not in self.expected:
            raise ValueError("Unknown sampling stage")
        if stage == "high" and self.callbacks["low"] != self.expected["low"]:
            raise RuntimeError("High stage started before the low stage completed")
        if stage == "low" and (self.callbacks["high"] or self.forwards["high"]):
            raise RuntimeError("Cannot return to low stage")
        if forward:
            self.forwards[stage] += 1
        else:
            self.callbacks[stage] += 1
            if self.callbacks[stage] > self.expected[stage]:
                raise RuntimeError("Too many Euler callbacks")

    def finish(self, *, allow_incomplete_evidence=False):
        if self.callbacks != self.expected:
            raise RuntimeError("Euler callback counts do not match the plan")
        if any(self.forwards[s] < self.callbacks[s] for s in self.expected):
            if not allow_incomplete_evidence:
                raise RuntimeError("Missing actual model-forward evidence")
            from .patch_stack_policy import warn_patch_stack
            warn_patch_stack('Progressive forward observer was bypassed; completion is not forward-count verification')
        result = {"callbacks": dict(self.callbacks), "actual_forwards": dict(self.forwards),
                  "scope": "execution_counts_only_not_quality_or_speed"}
        if any(self.forwards[s] < self.callbacks[s] for s in self.expected):
            result['forward_evidence_complete'] = False
        return result
