"""Experimental host-resident cache bridge; external pinned Tao implementation required.

No registration, model loading or global patching. A consumer must hold one layer
lease while evaluating attention and must not retain its tensors after exit.
"""
from contextlib import contextmanager
from dataclasses import replace

import torch
from taomate_bounded_retention import bounded_cache_class


class HostResidentCleanCache:
    def __init__(self, upstream, compute_contract):
        self.upstream = upstream
        self.contract = compute_contract
        self.host = bounded_cache_class(upstream)(replace(compute_contract, device_type='cpu'))
        self._leased_layer = None
        self._leased_pair = None
        self.transferred_bytes = 0
        self.peak_layer_bytes = 0

    @property
    def committed_blocks(self):
        return self.host.committed_blocks

    @property
    def history_tokens(self):
        return self.host.history_tokens

    @property
    def history_audio_tokens(self):
        return self.host.history_audio_tokens

    @property
    def history_video_tokens(self):
        return self.host.history_video_tokens

    @property
    def clean_commit_active(self):
        return self.host.clean_commit_active

    @contextmanager
    def layer(self, name, device):
        if self._leased_layer is not None:
            raise RuntimeError('Only one layer cache lease may be active')
        device = torch.device(device)
        if device.type != self.contract.device_type:
            raise ValueError('Requested device does not match compute contract')
        pair = self.host.history(name)
        self._leased_layer = name
        try:
            if pair is not None:
                self._leased_pair = self.upstream.AVKV(
                    pair.key.to(device=device, copy=True), pair.value.to(device=device, copy=True))
                size = sum(x.numel() * x.element_size() for x in (pair.key, pair.value))
                self.transferred_bytes += size
                self.peak_layer_bytes = max(size, self.peak_layer_bytes)
            yield
        finally:
            self._leased_pair = None
            self._leased_layer = None

    def history(self, name):
        if name != self._leased_layer:
            raise RuntimeError('Attention must borrow the matching layer explicitly')
        return self._leased_pair

    def _idle_lease(self):
        if self._leased_layer is not None:
            raise RuntimeError('Cache transaction cannot change during layer evaluation')

    def begin_clean_commit(self, index):
        self._idle_lease()
        return self.host.begin_clean_commit(index)

    def stage(self, name, key, value, tags, mask):
        if name != self._leased_layer:
            raise RuntimeError('Staging must occur inside its layer lease')
        self.upstream._validate_av_pair(self.upstream.AVKV(key, value), self.contract, layer_name=name)
        if tags.ndim != 1 or mask.ndim != 1 or tags.numel() != key.shape[0] or mask.numel() != key.shape[0]:
            raise ValueError('Packed metadata length mismatch')
        indices = mask.bool().nonzero().flatten()
        # Select on the compute device: do not copy unrelated text/reference rows.
        selected_tags = tags.index_select(0, indices).to('cpu', copy=True)
        current_key = key.index_select(0, indices).detach().to('cpu', copy=True)
        current_value = value.index_select(0, indices).detach().to('cpu', copy=True)
        if self.host._staged_tags is not None and tuple(selected_tags.tolist()) != self.host._staged_tags:
            raise ValueError('Layer commit media tags changed')
        self.host.stage(name, current_key, current_value, selected_tags, torch.ones_like(selected_tags, dtype=torch.bool))
        self.transferred_bytes += sum(x.numel() * x.element_size() for x in (current_key, current_value))

    def commit(self):
        self._idle_lease()
        return self.host.commit()

    def rollback(self):
        self._idle_lease()
        return self.host.rollback()

    def retain_sink_and_recent_commits(self):
        self._idle_lease()
        return self.host.retain_sink_and_recent_commits()

    def drop_audio_history(self):
        self._idle_lease()
        return self.host.drop_audio_history()

    def clear(self):
        self._idle_lease()
        return self.host.clear()
