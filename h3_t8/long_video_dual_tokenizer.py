"""Bind native MiniMax tokenizer vocabulary/settings to resumable job identity."""
from pathlib import Path

from .long_video_dual_identity import content_identity, _implementation
from .long_video_delivery import _sha256_file
from .patch_stack_policy import UnverifiedModelStack


def tokenizer_identity(clip):
    from comfy.text_encoders.minimax import MiniMaxH3Tokenizer, MiniMaxQwenSDTokenizer
    from transformers import Qwen2Tokenizer
    outer = clip.tokenizer
    if type(outer) is not MiniMaxH3Tokenizer or type(outer.qwen3vl_32b) is not MiniMaxQwenSDTokenizer:
        raise UnverifiedModelStack('Custom tokenizer lacks a portable vocabulary identity')
    inner = outer.qwen3vl_32b
    tokenizer = inner.tokenizer
    if type(tokenizer) is not Qwen2Tokenizer:
        raise UnverifiedModelStack('Tokenizer backend lacks a portable vocabulary identity adapter')
    if clip.use_clip_schedule or clip.apply_hooks_to_conds is not None:
        raise UnverifiedModelStack('Scheduled CLIP/hooks have no portable dual-stage identity adapter')
    for obj in (outer, inner, tokenizer):
        if any(callable(value) and not (obj is inner and key == 'tokenizer' and value is tokenizer)
               for key, value in vars(obj).items()):
            raise UnverifiedModelStack('Tokenizer contains an unverified instance-level execution override')
    settings = {key: value for key, value in vars(inner).items()
                if key not in {'tokenizer', 'inv_vocab', 'embedding_directory'}}
    # Textual inversion is resolved lazily. Bind file contents as well as the
    # ordered search paths; directory names alone would allow stale resume.
    directories = inner.embedding_directory or []
    if isinstance(directories, (str, Path)):
        directories = [directories]
    embeddings = []
    for directory in directories:
        root = Path(directory).resolve()
        embeddings.append({'root': str(root), 'files': {
            path.relative_to(root).as_posix(): _sha256_file(path)
            for path in sorted(root.rglob('*')) if path.is_file()}})
    backend = getattr(tokenizer, 'backend_tokenizer', None)
    if backend is not None:
        token_data = {'serialized_backend': backend.to_str()}
    else:
        token_data = {'vocab': tokenizer.get_vocab(),
                      'bpe_ranks': sorted((list(key), value) for key, value in tokenizer.bpe_ranks.items()),
                      'pattern': tokenizer.pat.pattern}
    return {'implementation': [_implementation(type(obj)) for obj in (outer, inner, tokenizer)],
            'data': content_identity(token_data), 'settings': content_identity(settings),
            'special_tokens': content_identity(tokenizer.special_tokens_map),
            'options': content_identity(clip.tokenizer_options), 'layer_idx': clip.layer_idx,
            'embeddings': embeddings}
