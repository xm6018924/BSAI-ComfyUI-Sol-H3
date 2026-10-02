"""Import-free ordered Tao request contract for the pending public stream route.

Audio seeds identify the prepared teacher. The public video seed is bound only
by the controller, incrementing once per request; neither operation mutates the
input manifest. Tensor/teacher semantics are checked separately by the worker.
"""

from copy import deepcopy

from .prepared_identity import absolute_path

STREAM_ITEM_FIELDS = {"request_index", "text_features", "milestones", "audio_seed"}


def validate_stream_requests(items, *, asset_paths=None):
    if not isinstance(items, list) or not items:
        raise ValueError("Prepared stream needs a nonempty ordered request list")
    identities = (
        None if asset_paths is None else {absolute_path(p) for p in asset_paths}
    )
    for index, item in enumerate(items):
        if not isinstance(item, dict) or set(item) != STREAM_ITEM_FIELDS:
            raise ValueError(
                "Unknown or missing stream request fields; video seeds are controller bound"
            )
        if type(item["request_index"]) is not int or item["request_index"] != index:
            raise ValueError("Stream request indices must be contiguous from zero")
        if type(item["audio_seed"]) is not int or not 0 <= item["audio_seed"] < 2**64:
            raise ValueError("Prepared stream audio_seed must be unsigned64")
        for name in ("text_features", "milestones"):
            path = absolute_path(item[name])
            if identities is not None and path not in identities:
                raise ValueError(
                    f"Stream request {index} {name} is missing an asset identity"
                )
    return items


def bind_stream_video_seeds(items, seed):
    validate_stream_requests(items)
    if type(seed) is not int or not 0 <= seed < 2**64:
        raise ValueError("Stream video seed must be unsigned64")
    bound = deepcopy(items)
    for index, item in enumerate(bound):
        item["video_seed"] = (seed + index) % 2**64
    return bound
