"""Append-only independent temporal Query correction; disabled by default."""

import json

from comfy_api.latest import io


class MiniMaxH3TSTModelEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TSTModelEXPT8',
            display_name='MiniMax H3 TST Temporal Query (EXP/T8)',
            category='T8/MiniMax H3/Performance/Experimental', is_experimental=True,
            description=('Standalone MODEL patch. Connect after LoRA, backend, and Relay/EAV configuration.'
                'Native Euler, CFG1; the full SIGMAS must match the sampler. This is not a quality guarantee; full-weight final-film validation is not complete.'
                'Signed Query correction via frame-mean proxy; this is not the current output-gain algorithm of the author code.'
                'LOW and HIGH can be configured separately; HIGH reuses the position on the full schedule and does not time from 0.'),
            inputs=[io.Model.Input('model'),
                io.Sigmas.Input('sigmas', tooltip='Full native video sigma schedule; HIGH uses only its corresponding interval. Do not pass a schedule with only the last 4 steps as if it were a full 8-step schedule.'),
                io.Combo.Input('mode', options=['disabled', 'report_only', 'apply_exp'], default='disabled',
                    tooltip='disabled installs no patch; report_only measures without modifying Query; apply_exp actually corrects.'),
                io.Float.Input('tau', default=.2, min=0., max=2., step=.01,
                    tooltip='Correction strength; 0 still computes diagnostics. Larger is not necessarily sharper; compare on fixed footage.'),
                io.Int.Input('max_workspace_mib', default=256, min=1, max=32768, advanced=True,
                    tooltip='Explicit temporary tensor budget; requests exceeding it are rejected. This is not a whole-GPU VRAM cap or an OOM guarantee.')],
            outputs=[io.Model.Output('model'), io.String.Output('configuration_json')])

    @classmethod
    def execute(cls, **kwargs):
        from .tst_model import build_tst_model
        model, report = build_tst_model(**kwargs)
        return io.NodeOutput(model, json.dumps(report, ensure_ascii=False, allow_nan=False))
