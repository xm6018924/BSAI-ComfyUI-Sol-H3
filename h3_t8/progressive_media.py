"""Lazy global audio windows; per-segment inputs are already windowed.

Reuse the established long-video selector without resampling or changing
conditioning/latent math. References are not output-timeline source video.
"""

import inspect

from .long_video import build_long_video_conditioning
from .long_video_in_node_loop_advanced import _window_segment_audio
from .core import validate_audio


def validate_condition_options(options):
    owned = {'clip', 'video_vae', 'audio_vae', 'context', 'segment_index',
             'context_frames', 'prompt', 'width', 'height', 'length', 'return_details'}
    allowed = set(inspect.signature(build_long_video_conditioning).parameters) - owned
    if type(options) is not dict or any(type(key) is not str for key in options):
        raise ValueError('Progressive condition options must be a string-keyed dictionary')
    if set(options) - allowed:
        raise ValueError(f'Unknown or job-owned condition options: {sorted(set(options) - allowed)}')


def resolve_segment_options(shared, local, segment):
    """Local overrides are explicit, already selected windows; never slice twice."""
    validate_condition_options(shared)
    validate_condition_options(local)
    options = {**shared, **local}
    mode = options.get('audio_mode', 'native')
    if not isinstance(mode, str) or mode.lower() not in {
            'native', 'reference_only', 'lock_source', 'remix_source'}:
        raise ValueError('Unknown progressive audio mode')
    mode = mode.lower()
    if mode != 'native' and options.get('drive_audio') is None:
        raise ValueError(f'Audio mode {mode} requires drive_audio')
    for name in ('drive_audio', 'final_audio'):
        audio = options.get(name)
        if name in local:
            if audio is not None:
                validate_audio(audio, name)
        elif name in shared:
            options[name] = _window_segment_audio(audio, segment.plan, name=name, audio_mode=mode)
    # A global endpoint belongs only to the final segment. An explicit local
    # endpoint is intentional and remains subject to the conditioning guards.
    if 'last_frame' not in local and not segment.plan.is_final_segment:
        options.pop('last_frame', None)
    return options


def select_delivery_audio(options, generated_audio):
    """Same precedence as build_long_video_conditioning, without VAE re-encode."""
    validate_condition_options(options)
    mode = options.get('audio_mode', 'native')
    if not isinstance(mode, str) or mode.lower() not in {
            'native', 'reference_only', 'lock_source', 'remix_source'}:
        raise ValueError('Unknown progressive audio mode')
    explicit = options.get('final_audio')
    source = options.get('drive_audio')
    if mode.lower() != 'native' and source is None:
        raise ValueError(f'Audio mode {mode} requires drive_audio')
    if explicit is not None:
        selected, origin = explicit, 'final_audio'
    elif source is not None and mode.lower() in {'native', 'lock_source'}:
        selected, origin = source, 'drive_audio'
    else:
        selected, origin = generated_audio, 'generated_audio'
    if selected is not None:
        validate_audio(selected, origin)
    return selected, origin
