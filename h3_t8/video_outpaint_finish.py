"""Shared, parameter-bound geometry -> color -> exact-source finishing."""
from __future__ import annotations

import hashlib

from .video_outpaint_color import color_match_outpaint_frames
from .video_outpaint_plan import canonical, _finite, _integer, validate_outpaint_plan
from .video_outpaint_pixel_receipt import validate_source_mode


def geometry_settings(geometry_align=False, alignment_band_pixels=64, alignment_max_displacement=8.0):
    if not isinstance(geometry_align, bool):
        raise ValueError('geometry_align must be boolean')
    return {'geometry_align':geometry_align,
            'alignment_band_pixels':_integer(alignment_band_pixels,'alignment_band_pixels',8,512),
            'alignment_max_displacement':_finite(alignment_max_displacement,'alignment_max_displacement',0,16)}


def _digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def finish_outpaint_frames(source,candidate,plan,*,start_frame=0,state=None,enabled=True,
                           geometry_align=False,alignment_band_pixels=64,alignment_max_displacement=8.0,
                           source_mode="preserve_source",
                           **color_settings):
    """Disabled geometry preserves the legacy image/state contract bit-for-bit."""
    settings=geometry_settings(geometry_align,alignment_band_pixels,alignment_max_displacement)
    mode=validate_source_mode(source_mode)
    if not isinstance(enabled,bool):
        raise ValueError('enabled must be boolean')
    if mode == "joint_decode":
        checked=validate_outpaint_plan(plan)
        binding=_digest({'plan':checked['plan_sha256'],'source_mode':mode,
                         'requested_color':enabled,'geometry':settings})
        color_state=None
        if state is not None:
            if not isinstance(state,dict):
                raise ValueError('invalid joint decode state')
            body={k:v for k,v in state.items() if k!='state_sha256'}
            if (set(body)!={'schema','binding','next_frame','color'} or
                    state.get('state_sha256')!=_digest(body) or body['schema']!='t8.h3.outpaint.joint_decode/v1' or
                    body['binding']!=binding or body['next_frame']!=start_frame):
                raise ValueError('joint decode source/settings/sequence mismatch')
            color_state=body['color']
        # Reuse strict shape/range/color-setting/shot-state validation, but never
        # deliver the pasted result. Joint decode bypasses *both* source-edge
        # geometry and tint correction, which would reintroduce a source seam.
        _,color_state,report=color_match_outpaint_frames(source,candidate,checked,
            start_frame=start_frame,state=color_state,enabled=False,**color_settings)
        continuation={'schema':'t8.h3.outpaint.joint_decode/v1','binding':binding,
                      'next_frame':start_frame+len(source),'color':color_state}
        continuation['state_sha256']=_digest(continuation)
        report.update(algorithm='joint_vae_decode_without_source_pasteback_v1',
            source_exact_before_encoding=False,source_reconstructed=True,source_mode=mode,
            source_postprocessing_bypassed=True,requested_color_match=enabled,
            geometry_settings=settings,state_sha256=continuation['state_sha256'])
        return candidate.clone(),continuation,report
    if not geometry_align:
        return color_match_outpaint_frames(source,candidate,plan,start_frame=start_frame,state=state,
                                          enabled=enabled,**color_settings)
    checked=validate_outpaint_plan(plan)
    binding=_digest({'plan':checked['plan_sha256'],'geometry':settings})
    color_state=None
    if state is not None:
        if not isinstance(state,dict):
            raise ValueError('invalid finishing state')
        body={k:v for k,v in state.items() if k!='state_sha256'}
        if (set(body)!={'schema','binding','next_frame','color'} or
                state.get('state_sha256')!=_digest(body) or body['schema']!='t8.h3.outpaint.finish/v1' or
                body['binding']!=binding or body['next_frame']!=start_frame):
            raise ValueError('finishing source/settings/sequence mismatch')
        color_state=body['color']
    from .video_outpaint_alignment import register_outpaint_candidate
    aligned,alignment=register_outpaint_candidate(source,candidate,checked,start_frame=start_frame,
        band_pixels=settings['alignment_band_pixels'],max_displacement=settings['alignment_max_displacement'])
    result,color_state,report=color_match_outpaint_frames(source,aligned,checked,start_frame=start_frame,
        state=color_state,enabled=enabled,**color_settings)
    continuation={'schema':'t8.h3.outpaint.finish/v1','binding':binding,
                  'next_frame':start_frame+len(source),'color':color_state}
    continuation['state_sha256']=_digest(continuation)
    report.update(algorithm='source_geometry_then_paired_color_exact_pasteback_v1',
                  geometry_settings=settings,geometry_alignment=alignment,
                  state_sha256=continuation['state_sha256'])
    return result,continuation,report
