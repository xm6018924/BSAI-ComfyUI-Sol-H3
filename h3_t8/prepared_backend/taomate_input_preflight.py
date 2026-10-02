"""CPU-only exact teacher/text/milestone binding, shared with generation."""
from pathlib import Path


def load_inputs(request, torch):
    from safetensors import safe_open
    from safetensors.torch import load_file
    from taomate_h3.inference.base10_teacher import ExternalBase10TeacherArtifact, BASE10_TEACHER_STATE_NUMBERS
    text = load_file(request['text_features'], device='cpu')
    if not {'hidden', 'tags'} <= set(text) or any(not torch.isfinite(t).all() for t in text.values()):
        raise ValueError('Prepared Tao text features are incomplete/nonfinite')
    with safe_open(request['text_features'], framework='pt', device='cpu') as source:
        prompt = (source.metadata() or {}).get('prompt')
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError('Prepared Tao text has no source prompt metadata')
    teacher = ExternalBase10TeacherArtifact.open(Path(request['teacher']), prompts=[prompt],
        seeds=[request['audio_seed']], consumer_geometry=dict(width=864, height=480, video_latent_h=30, video_latent_w=54))
    external = teacher.load_request(torch, request_index=0, audio_latent_count=207, device='cpu')
    saved = load_file(request['milestones'], device='cpu')
    for index, number in enumerate(BASE10_TEACHER_STATE_NUMBERS):
        expected = saved.get(f'state_{number}')
        actual = external['milestones'][index]
        if expected is None or not torch.equal(expected, actual) or not torch.isfinite(actual).all():
            raise ValueError('Prepared Tao teacher and saved Base10 milestones disagree')
    known = saved['state_9'].reshape(2, 207, 32).permute(0, 2, 1).contiguous()
    return text, prompt, teacher, known, external['receipt']
