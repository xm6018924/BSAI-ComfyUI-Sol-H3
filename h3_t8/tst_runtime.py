"""Internal stage-local TST query owner; never replaces an attention backend.

The containing native MODEL owner must open forward() on the full sigma clock
and call transform() before Relay chunks queries. This module installs no hooks.
"""

from contextlib import contextmanager
import copy
import math
import threading

import torch

from .tst_math import PROFILE, correct_video_queries, effective_strength, _integer

TST_RUNTIME_KEY = 't8_owned_tst_query_runtime_v1'


def transform_owned_queries(q, k, heads, transformer_options, *, skip_reshape, skip_output_reshape):
    """Explicit router seam, before any Relay query split. Disabled is identity."""
    runtime = transformer_options.get(TST_RUNTIME_KEY)
    if runtime is None:
        return q
    if type(runtime) is not TSTQueryRuntime or not skip_reshape or skip_output_reshape:
        raise RuntimeError('TST requires its authenticated native query owner/layout')
    return runtime.transform(q, k, heads, transformer_options)


class TSTQueryRuntime:
    def __init__(self, full_sigmas, *, stage_start, stage_end, layer_count, frames,
                 spatial_tokens, mode='report_only', tau=.2, max_workspace_mib=256):
        if mode not in ('report_only', 'apply_exp'):
            raise ValueError('Disabled TST must not install a runtime')
        if not isinstance(full_sigmas, (list, tuple)) or len(full_sigmas) < 3:
            raise ValueError('TST needs a full native schedule with at least two evaluations')
        if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v)
               for v in full_sigmas):
            raise ValueError('TST schedule values must be finite numbers')
        values = tuple(float(v) for v in full_sigmas)
        if not 0 < values[0] <= 1 or values[-1] != 0 or any(a <= b for a, b in zip(values, values[1:])):
            raise ValueError('TST schedule must strictly decrease from (0,1] to zero')
        _integer(stage_start, 'stage_start')
        _integer(stage_end, 'stage_end', 1)
        if not stage_start < stage_end <= len(values) - 1:
            raise ValueError('TST stage must be an interval in the full schedule')
        _integer(frames, 'frames', 2)
        _integer(spatial_tokens, 'spatial_tokens', 1)
        _integer(max_workspace_mib, 'max_workspace_mib', 1)
        effective_strength(tau, layer_index=0, layer_count=layer_count,
                           step_index=0, total_steps=len(values) - 1)
        self.config = dict(profile=PROFILE, full_sigmas=values, stage_start=stage_start,
            stage_end=stage_end, layer_count=layer_count, frames=frames, spatial_tokens=spatial_tokens,
            mode=mode, tau=float(tau), max_workspace_mib=max_workspace_mib)
        self._original_config = copy.deepcopy(self.config)
        self._next = stage_start
        self._active = None
        self._records = []
        self.failed = False

    def _verify(self):
        if self.config != self._original_config:
            raise RuntimeError('TST runtime configuration changed after binding')
        if self.failed:
            raise RuntimeError('Failed TST runtime requires a fresh execution owner')

    @contextmanager
    def forward(self, sigma_video):
        self._verify()
        if self._active is not None or self._next >= self.config['stage_end']:
            raise RuntimeError('TST forward is nested or outside its assigned stage')
        expected = self.config['full_sigmas'][self._next]
        if (isinstance(sigma_video, bool) or not isinstance(sigma_video, (int, float)) or
                not math.isfinite(sigma_video) or not math.isclose(sigma_video, expected, rel_tol=2e-6, abs_tol=2e-6)):
            raise RuntimeError('TST forward sigma differs from the full schedule')
        self._active = dict(thread=threading.get_ident(), layers=set(), seq_len=None, layout=None,
                            applied_layers=0, gamma_min=math.inf, gamma_max=-math.inf, max_estimated_tensor_bytes=0)
        try:
            yield self
            self._verify()
            state = self._active
            missing = sorted(set(range(self.config['layer_count'])) - state['layers'])
            if missing:
                from .patch_stack_policy import warn_patch_stack
                warn_patch_stack(f'TST query processing was bypassed at layers {missing}; coverage is unverified')
            self._records.append(dict(step_index=self._next, sigma_video=float(sigma_video),
                query_transform_calls=len(state['layers']), seq_len=state['seq_len'],
                query_coverage_verified=not missing, bypassed_layers=missing,
                gamma_min=state['gamma_min'] if state['layers'] else None,
                gamma_max=state['gamma_max'] if state['layers'] else None,
                **{key: state[key] for key in ('applied_layers', 'max_estimated_tensor_bytes')}))
            self._next += 1
        except BaseException:
            self.failed = True
            raise
        finally:
            self._active = None

    def transform(self, q, k, heads, transformer_options):
        self._verify()
        state = self._active
        if state is None or state['thread'] != threading.get_ident():
            raise RuntimeError('TST query transform requires its active owning forward')
        if not isinstance(transformer_options, dict):
            raise RuntimeError('TST requires native transformer options')
        layout = transformer_options.get('minimax_h3_layout')
        if layout is None or type(getattr(layout, 'seq_len', None)) is not int:
            raise RuntimeError('TST requires the actual native packed layout')
        if not isinstance(q, torch.Tensor) or q.ndim != 4:
            raise RuntimeError('TST requires native B,H,S,D queries')
        if q.shape[-2] != layout.seq_len:
            # Token-refiner attention is outside the main packed AV sequence.
            # It must not advance the layer clock or be query-scaled.
            if 0 < q.shape[-2] < layout.seq_len:
                return q
            raise RuntimeError('TST query length exceeds the actual packed layout')
        spans = list(getattr(layout, 'segments', ()))
        cursor, videos = 0, []
        for start, end, kind in spans:
            if type(start) is not int or type(end) is not int or start != cursor or not start < end <= layout.seq_len:
                raise RuntimeError('TST packed layout spans are not contiguous')
            if kind == 'video':
                videos.append((start, end))
            cursor = end
        if cursor != layout.seq_len or len(videos) != 1:
            raise RuntimeError('TST requires one exact target video segment')
        start, end = videos[0]
        if end - start != self.config['frames'] * self.config['spatial_tokens']:
            raise RuntimeError('TST video span differs from its stage geometry')
        layer = transformer_options.get('block_index')
        if type(layer) is not int or not 0 <= layer < self.config['layer_count'] or layer in state['layers']:
            raise RuntimeError('TST received a missing, duplicate or invalid Core block_index')
        if type(heads) is not int or q.shape[1] != heads:
            raise RuntimeError('TST head count differs from native queries')
        signature = (layout.seq_len, tuple(spans), tuple(q.shape), q.dtype, q.device)
        if state['layout'] is not None and state['layout'] != signature:
            raise RuntimeError('TST packed geometry changed within a forward')
        output, report = correct_video_queries(q, k, video_start=start,
            frames=self.config['frames'], spatial_tokens=self.config['spatial_tokens'],
            mode=self.config['mode'], tau=self.config['tau'], layer_index=layer,
            layer_count=self.config['layer_count'], step_index=self._next,
            total_steps=len(self.config['full_sigmas']) - 1,
            max_workspace_mib=self.config['max_workspace_mib'])
        state['layers'].add(layer)
        state['seq_len'] = layout.seq_len
        state['layout'] = signature
        state['applied_layers'] += int(report['applied'])
        state['gamma_min'] = min(state['gamma_min'], float(report['gamma'].min()))
        state['gamma_max'] = max(state['gamma_max'], float(report['gamma'].max()))
        state['max_estimated_tensor_bytes'] = max(state['max_estimated_tensor_bytes'], report['estimated_tensor_bytes'])
        return output

    def snapshot(self, *, require_complete=False):
        self._verify()
        if self._active is not None:
            raise RuntimeError('Cannot finalize TST evidence inside a forward')
        complete = self._next == self.config['stage_end']
        if require_complete and not complete:
            raise RuntimeError('TST assigned stage has not completed')
        return dict(config=copy.deepcopy(self.config), forwards=copy.deepcopy(self._records),
            completed=complete, model_or_backend_calls_verified=False,
            scope='query_transform_execution_only_not_model_backend_or_quality_qualification')
