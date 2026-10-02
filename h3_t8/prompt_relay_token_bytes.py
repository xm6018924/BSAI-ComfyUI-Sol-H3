"""Lossless offsets for verified byte-level tokenizers, never replacement decode.

Token IDs still come from the connected native H3 encoder. Fast tokenizer
character offsets cannot distinguish two tokens splitting one UTF-8 character.
"""
import json
from collections.abc import Mapping


def byte_decoder(tokenizer):
    legacy = getattr(tokenizer, "byte_decoder", None)
    if isinstance(legacy, Mapping) and legacy:
        return legacy
    backend = getattr(tokenizer, "backend_tokenizer", None)
    if backend is None:
        backend = getattr(tokenizer, "_tokenizer", None)
    decoder = getattr(backend, "decoder", None)
    if decoder is None:
        raise ValueError("Tokenizer has neither a byte decoder nor a verified ByteLevel backend")
    try:
        state = json.loads(decoder.__getstate__())
    except (TypeError, ValueError, AttributeError) as error:
        raise ValueError("Cannot inspect tokenizer byte decoder") from error
    if state.get("type") != "ByteLevel":
        raise ValueError("Prompt Relay requires lossless ByteLevel decoding; other decoder types are unsupported")
    # GPT/Qwen ByteLevel bijection. No Unicode normalization or UTF-8 replacement.
    preserved = list(range(33, 127)) + list(range(161, 173)) + list(range(174, 256))
    characters = list(preserved)
    extra = 0
    for value in range(256):
        if value not in preserved:
            preserved.append(value)
            characters.append(256 + extra)
            extra += 1
    return dict(zip(map(chr, characters), preserved))


def supports_byte_tokens(tokenizer):
    if not callable(getattr(tokenizer, "convert_ids_to_tokens", None)):
        return False
    try:
        byte_decoder(tokenizer)
        return True
    except ValueError:
        return False


def token_byte_offsets(prompt, token_ids, tokenizer):
    decoder = byte_decoder(tokenizer)
    added = getattr(tokenizer, "added_tokens_decoder", {})
    decoded, offsets = bytearray(), []
    for token_id in token_ids:
        token_id = int(token_id)
        token = tokenizer.convert_ids_to_tokens(token_id)
        if not isinstance(token, str) or not token:
            raise RuntimeError(f"Prompt Relay invalid token representation for ID {token_id}")
        special = added.get(token_id) if isinstance(added, Mapping) else None
        if special is not None:
            content = getattr(special, "content", str(special))
            if content != token:
                raise RuntimeError("Prompt Relay added-token content does not match its ID")
            piece = token.encode("utf-8")
        else:
            try:
                piece = bytes(decoder[character] for character in token)
            except (KeyError, TypeError, ValueError) as error:
                raise RuntimeError(f"Prompt Relay could not reconstruct byte offsets for token {token!r}") from error
        start = len(decoded)
        decoded.extend(piece)
        offsets.append((start, len(decoded)))
    if bytes(decoded) != prompt.encode("utf-8"):
        raise RuntimeError("Prompt Relay tokenizer byte reconstruction does not match the compiled prompt")
    return offsets
