"""Bind actual paired standalone Relay inputs across LOW checkpoint replay."""

import json

from .progressive_relay import detach_relay_input, strip_paired_conditioning, _fresh_delegate
from .progressive_continuation_runtime import _input_identity
from .long_video_dual_identity import _implementation
from .h3_core_compat import plain_attention_backend
from .patch_stack_policy import model_identity_matches


class ProgressiveRelayCheckpointBinding:
    def __init__(self, low, high, positive, negative, sampler):
        self.models = low, high
        self.positive, self.negative, self.sampler = positive, negative, sampler
        self._snapshot = self._capture()

    @property
    def identity(self):
        return json.loads(json.dumps(self._snapshot))

    def _capture(self):
        from .progressive_checkpoint import native_model_identity, canonical
        phases = []
        contracts = []
        for model in self.models:
            base, contract = detach_relay_input(model)
            contracts.append(contract)
            route = None
            if contract is not None:
                # Authentication above checks the actual factory-owned wrapper,
                # paired binding, and installed override, not an input SHA label.
                backend = _fresh_delegate(contract['attention_backend'])
                plain = contract['source_plain_override']
                route = dict(binding=contract['binding'], query_chunk_rows=contract['query_chunk_rows'],
                    core_hashes=contract['core_hashes'],
                    backend=backend.report() if backend is not None else None,
                    plain=None if plain is None else dict(name=plain_attention_backend(plain),
                                                         implementation=_implementation(plain)))
            phases.append(dict(model=native_model_identity(base, self.sampler), relay=route))
        low, high = contracts
        if low is None or (high is not None and high['binding'] != low['binding']):
            raise ValueError('Checkpoint Relay requires paired LOW and compatible HIGH bindings')
        inputs = dict(positive=strip_paired_conditioning(self.positive, low, required=True),
                      negative=strip_paired_conditioning(self.negative, low, required=False))
        return json.loads(canonical(dict(phases=phases, inputs=_input_identity(inputs))))

    def verify(self):
        if not model_identity_matches(self._snapshot, self._capture()):
            raise ValueError('Progressive checkpoint Relay inputs changed')
        return self.identity
