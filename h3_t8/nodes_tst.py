"""Append-only independent temporal Query correction; disabled by default."""

import json

from comfy_api.latest import io


class MiniMaxH3TSTModelEXPT8(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(node_id='MiniMaxH3TSTModelEXPT8',
            display_name='MiniMax H3 TST Temporal Query / 时序Query修正 (EXP/T8)',
            category='T8/MiniMax H3/Performance/Experimental', is_experimental=True,
            description=('独立MODEL补丁。接在LoRA、后端、Relay/EAV配置之后。'
                '原生Euler、CFG1；完整SIGMAS必须与采样器一致。不是画质保证，未完成完整权重成片验收。'
                '帧均值代理的有符号Query修正，非作者代码当前的输出增益算法。'
                'LOW/HIGH可分别配置；HIGH沿用完整时间表的位置，不从0计时。'),
            inputs=[io.Model.Input('model'),
                io.Sigmas.Input('sigmas', tooltip='完整原生视频sigma表；HIGH只使用其对应区间，不能传只有后4步的表冒充完整8步。'),
                io.Combo.Input('mode', options=['disabled', 'report_only', 'apply_exp'], default='disabled',
                    tooltip='disabled不安装补丁；report_only测量不改Query；apply_exp实际修正。'),
                io.Float.Input('tau', default=.2, min=0., max=2., step=.01,
                    tooltip='修正强度；0仍计算诊断。增大不等于更清晰，需固定素材对照。'),
                io.Int.Input('max_workspace_mib', default=256, min=1, max=32768, advanced=True,
                    tooltip='显式临时tensor预算，超过会拒绝；不是整卡显存上限或无OOM保证。')],
            outputs=[io.Model.Output('model'), io.String.Output('configuration_json')])

    @classmethod
    def execute(cls, **kwargs):
        from .tst_model import build_tst_model
        model, report = build_tst_model(**kwargs)
        return io.NodeOutput(model, json.dumps(report, ensure_ascii=False, allow_nan=False))
