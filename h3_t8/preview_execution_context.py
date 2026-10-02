"""Read-only sampling-stage labels, scoped to the current execution context."""
from contextlib import contextmanager
from contextvars import ContextVar

CONTEXT = ContextVar('t8_sampling_preview_context_v1', default=None)


@contextmanager
def preview_scope(**labels):
    previous = CONTEXT.get() or {}
    token = CONTEXT.set({**previous, **labels})
    try:
        yield
    finally:
        CONTEXT.reset(token)
