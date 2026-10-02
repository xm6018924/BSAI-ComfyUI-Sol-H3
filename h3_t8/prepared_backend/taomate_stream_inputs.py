"""Validate ordered prepared requests against native Base10 continuation plans.

CPU only. No conversion of a first-request teacher into a continuation teacher;
each request must have its own matching upstream artifact and saved milestones.
The existing one-request public execution contract is not changed here.
"""
from pathlib import Path


def load_stream_inputs(request, torch):
    from safetensors import safe_open
    from safetensors.torch import load_file
    from taomate_h3.inference.base10_teacher import (
        BASE10_TEACHER_STATE_NUMBERS, ExternalBase10TeacherArtifact)
    from taomate_h3.streaming.geometry import canonical_continuation_plan, direct_5s_plan

    items = request.get('stream_requests')
    if not isinstance(items, list) or not items:
        raise ValueError('Prepared stream requires an ordered nonempty request list')
    prepared = []
    for index, item in enumerate(items):
        if not isinstance(item, dict) or item.get('request_index') != index or type(item.get('request_index')) is not int:
            raise ValueError('Prepared stream request indices must be contiguous from zero')
        for name in ('audio_seed', 'video_seed'):
            if type(item.get(name)) is not int or not 0 <= item[name] < 2**64:
                raise ValueError('Prepared stream seeds must be unsigned64 integers')
        text = load_file(item['text_features'], device='cpu')
        hidden, tags = text.get('hidden'), text.get('tags')
        if (hidden is None or tags is None or hidden.ndim != 2 or hidden.shape[1] != 5120
                or hidden.shape[0] == 0 or hidden.dtype != torch.bfloat16
                or tags.shape != (hidden.shape[0],) or tags.dtype != torch.long
                or not torch.isfinite(hidden).all() or not torch.all(tags == 1)):
            raise ValueError('Prepared stream text must be finite raw BF165120 text-only features')
        with safe_open(item['text_features'], framework='pt', device='cpu') as source:
            prompt = (source.metadata() or {}).get('prompt')
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError('Prepared stream text has no source prompt metadata')
        plan = direct_5s_plan()
        if index:
            plan = canonical_continuation_plan(plan, request_index=index)
        count = plan.phases[-1].audio_latent_stop
        prepared.append(dict(item, text=text, prompt=prompt, audio_latent_count=count))

    teacher = ExternalBase10TeacherArtifact.open(Path(request['teacher']),
        prompts=[item['prompt'] for item in prepared],
        seeds=[item['audio_seed'] for item in prepared],
        consumer_geometry=dict(width=864, height=480, video_latent_h=30, video_latent_w=54))
    for index, item in enumerate(prepared):
        # These seeds are used by the prepared consumer's upstream noise builder.
        # An independently offset teacher seed must not silently change them.
        if teacher.audio_noise_seed(index) != item['audio_seed']:
            raise ValueError('Prepared consumer audio noise seed differs from teacher')
        count = item['audio_latent_count']
        external = teacher.load_request(torch, request_index=index,
            audio_latent_count=count, device='cpu')
        saved = load_file(item['milestones'], device='cpu')
        if set(saved) != {f'state_{number}' for number in BASE10_TEACHER_STATE_NUMBERS}:
            raise ValueError('Prepared stream saved milestones must contain exactly states3/6/9')
        for stage, number in enumerate(BASE10_TEACHER_STATE_NUMBERS):
            actual, expected = external['milestones'][stage], saved[f'state_{number}']
            if (actual.dtype != torch.float32 or expected.dtype != torch.float32
                    or expected.shape != (2 * count, 32) or not torch.isfinite(actual).all()
                    or not torch.equal(actual, expected)):
                raise ValueError('Prepared stream request milestones differ from its teacher')
        item['known_audio'] = saved['state_9'].reshape(2, count, 32).permute(0, 2, 1).contiguous()
        item['teacher_receipt'] = external['receipt']
    return teacher, tuple(prepared)
