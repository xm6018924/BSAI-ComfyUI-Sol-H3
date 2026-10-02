"""Single-stream ownership/lifecycle; never alters upstream cache or sampling math."""
from threading import Lock


class OwnedRequestState:
    def __init__(self):
        self.state = 'ready'
        self.records = []
        self.cleanup_errors = []
        self._model = None
        self._lock = Lock()

    def execute(self, runtime, model, call, release, interrupt=lambda: None):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Streaming owner already has an active request')
        try:
            if self.state != 'ready':
                raise RuntimeError(f'Streaming owner is {self.state}; create a new runtime for a new task')
            if self._model is not None and self._model is not model:
                raise ValueError('Cannot change the model owner within a retained stream')
            if runtime._request_index != len(self.records) or len(runtime.executions) != len(self.records):
                raise RuntimeError('Streaming request history changed outside its owner')
            self._model = model
            self.state = 'running'
            before = {name: getattr(runtime, name) for name in (
                '_request_index', '_native_frame_offset', '_video_latent_offset', '_audio_latent_offset')}
            try:
                interrupt()
                result = call()
                interrupt()
                if runtime._running:
                    raise RuntimeError('Upstream request did not exit its running scope')
                if runtime._request_index != before['_request_index'] + 1 or len(runtime.executions) != len(self.records) + 1:
                    raise RuntimeError('Upstream did not commit exactly one request')
                execution = runtime.executions[-1]
                for start, count in (('native_frame_offset', 'published_native_frames'),
                                     ('video_latent_offset', 'published_video_latents'),
                                     ('audio_latent_offset', 'published_audio_latents_per_channel')):
                    if getattr(execution, start) != before['_' + start]:
                        raise RuntimeError('Upstream request starts at the wrong global offset')
                    delta = getattr(execution, count)
                    if type(delta) is not int or delta <= 0 or getattr(runtime, '_' + start) != before['_' + start] + delta:
                        raise RuntimeError('Upstream request has an invalid published timeline')
                self.records.append({'request_index': before['_request_index'],
                    'before': before, 'after': {name: getattr(runtime, name) for name in before}})
                self.state = 'ready'
                return result
            except BaseException as error:
                # Upstream can commit some phases before raising. No rollback to
                # an apparently reusable state: close this owner permanently.
                self.state = 'failed'
                try:
                    release()
                except BaseException as cleanup_error:
                    detail = f'Owned streaming cleanup also failed: {type(cleanup_error).__name__}: {cleanup_error}'
                    self.cleanup_errors.append(detail)
                    add_note = getattr(error, 'add_note', None)
                    if callable(add_note):
                        add_note(detail)
                finally:
                    self._model = None
                raise
        finally:
            self._lock.release()

    def close(self, release):
        if not self._lock.acquire(blocking=False):
            raise RuntimeError('Cannot close an active streaming request')
        try:
            if self.state == 'closed':
                return
            # Mark before cleanup: a cleanup error must never reopen the stream.
            self.state = 'closed'
            try:
                release()
            finally:
                self._model = None
        finally:
            self._lock.release()
