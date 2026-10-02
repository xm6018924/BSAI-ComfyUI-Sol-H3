"""First-window candidates reuse the exact serial sampler and durable AV prefix.

This is the internal selection protocol, not a generated-image UI or acceptance.
A bounded store view stops the existing coordinator normally after window zero;
it neither fakes cancellation nor changes its sampler/noise implementation hash.
"""
from __future__ import annotations

import hashlib
import json

from .video_outpaint_plan import canonical
from .video_outpaint_sampling_runtime import sample_prepared_outpaint_windows


class _FirstWindowRun:
    """Expose a bounded schedule but retain the real store's completion status."""

    def __init__(self, store):
        self.store = store
        self.windows = store.windows[:1]
        self.plan, self.identity, self.path = store.plan, store.identity, store.path

    def snapshot(self):
        state = self.store.snapshot()
        if len(state["committed"]) > 1:
            raise ValueError("candidate generation requires at most the first committed window")
        return state

    def begin(self, *, resume=False):
        self.snapshot()
        start = self.store.begin(resume=resume)
        if start == 1:
            self.store.pause()
        return start

    def commit(self, index, video, audio):
        if index != 0:
            raise ValueError("candidate run may commit only window zero")
        self.store.commit(index, video, audio)
        self.store.pause()

    def load(self, index):
        return self.store.load(index)

    def context_for(self, index):
        return self.store.context_for(index)

    def mark_interrupted(self):
        self.store.mark_interrupted()


def _receipt(body):
    return {**body, "sha256": hashlib.sha256(canonical(body).encode()).hexdigest()}


def candidate_receipt(store):
    """Produce a source/settings/asset-bound candidate descriptor, not selection."""
    state = store.snapshot()
    if len(state["committed"]) != 1 or state["status"] not in {"paused", "sampled"}:
        raise ValueError("candidate requires a successfully paused first window")
    store.load(0)
    body = {"schema": "t8.h3.video_outpaint.candidate/v1", "identity": state["identity"],
            "prefix": state["committed"], "first_frame_index": 0,
            "preview_scope": "first_window_not_full_clip_acceptance"}
    return _receipt(json.loads(canonical(body)))


def _validate_candidate(candidate, store):
    if not isinstance(candidate, dict) or set(candidate) != {
        "schema", "identity", "prefix", "first_frame_index", "preview_scope", "sha256"
    }:
        raise ValueError("invalid candidate receipt")
    body = {k: v for k, v in candidate.items() if k != "sha256"}
    if (candidate != _receipt(body) or candidate["schema"] != "t8.h3.video_outpaint.candidate/v1"
            or type(candidate["first_frame_index"]) is not int or candidate["first_frame_index"] != 0
            or candidate["preview_scope"] != "first_window_not_full_clip_acceptance"
            or candidate["identity"] != store.identity):
        raise ValueError("candidate integrity or execution identity mismatch")
    state = store.snapshot()
    prefix = candidate["prefix"]
    if not isinstance(prefix, list) or len(prefix) != 1 or state["committed"][:1] != prefix:
        raise ValueError("selected candidate differs from the committed AV prefix")
    store.load(0)


def select_candidate(candidate, store):
    """Explicit selection; no seed change, latent copy, sampling or implicit choice."""
    _validate_candidate(candidate, store)
    return _receipt({"schema": "t8.h3.video_outpaint.selection/v1",
                     "candidate": json.loads(canonical(candidate))})


def sample_first_candidate(*, window_store, **execution):
    report = sample_prepared_outpaint_windows(window_store=_FirstWindowRun(window_store), **execution)
    candidate = candidate_receipt(window_store)
    return candidate, {**report, "awaiting_explicit_selection": True,
                       "preview_is_generated_image": False, "candidate_sha256": candidate["sha256"]}


def continue_selected_candidate(*, selection, window_store, verify_execution, **execution):
    if not isinstance(selection, dict) or set(selection) != {"schema", "candidate", "sha256"}:
        raise ValueError("an explicit candidate selection is required")
    frozen = json.loads(canonical(selection))
    body = {k: v for k, v in frozen.items() if k != "sha256"}
    if frozen != _receipt(body) or frozen["schema"] != "t8.h3.video_outpaint.selection/v1":
        raise ValueError("selection integrity mismatch")
    _validate_candidate(frozen["candidate"], window_store)
    if "resume" in execution:
        raise ValueError("selected continuation owns its explicit resume policy")

    def verify_selected():
        # Recheck under the serial runtime lease before every new commit. The
        # source/model providers are still verified by the original coordinator.
        _validate_candidate(frozen["candidate"], window_store)
        return verify_execution()

    report = sample_prepared_outpaint_windows(window_store=window_store,
        verify_execution=verify_selected, resume=True, **execution)
    return {**report, "selected_candidate_sha256": frozen["candidate"]["sha256"],
            "selected_prefix_reused": True, "perceptual_acceptance": False}
