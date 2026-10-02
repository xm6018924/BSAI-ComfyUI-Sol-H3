"""Opt-in source-guided geometry registration for an outpaint candidate.

This is not color matching. Correspondences are measured inside the original
source rectangle and extrapolated only a bounded distance into generated pixels.
No model is downloaded; OpenCV is imported only when this experiment is called.
The returned candidate still requires exact RGB source pasteback. This module is
not wired into released nodes until learned-video and human validation pass.
"""
from __future__ import annotations

import numpy as np
import torch

from .video_outpaint_composite import composite_outpaint_frames
from .video_outpaint_plan import _finite, _integer, validate_outpaint_plan


def _boundary_line(flow, confidence, *, side, maximum):
    """Near-border estimates only; discard inconsistent/large correspondences."""
    if side in (0, 2):
        flow, confidence = flow.transpose(1, 0, 2), confidence.T
    if side in (2, 3):
        flow, confidence = flow[::-1], confidence[::-1]
    depth = min(12, flow.shape[0]-1)
    local, error = flow[1:depth+1], confidence[1:depth+1]
    valid = (error < 1.5) & (np.linalg.norm(local, axis=-1) <= maximum)
    weights = valid / (np.arange(1, depth+1, dtype=np.float32)[:, None]+1)
    line = (local*weights[..., None]).sum(axis=0) / weights.sum(axis=0)[:, None].clip(.001)
    reliable = valid.sum(axis=0) >= max(2, depth//3)
    if reliable.mean() < .5:
        return np.zeros_like(line, dtype=np.float32), {'applied': False, 'reason': 'insufficient_consistent_matches'}
    for axis in range(2):
        line[:, axis] = np.interp(np.arange(len(line)), np.where(reliable)[0], line[reliable, axis])
    # Local smoothing reduces spiky mesh distortion; no temporal averaging/lag.
    padded = np.pad(line, ((2, 2), (0, 0)), mode='edge')
    line = sum(padded[i:i+len(line)]*weight for i, weight in enumerate((1, 4, 6, 4, 1)))/16
    smoothing = 0
    # A correspondence can be individually consistent yet vary too abruptly
    # between adjacent pixels to define a non-folding extension. Regularize that
    # spatial field before extrapolation, not by switching the whole frame off.
    for _ in range(24):
        if np.linalg.norm(np.diff(line, axis=0),axis=-1).max(initial=0) <= .35:
            break
        padded = np.pad(line, ((2,2),(0,0)), mode='edge')
        line = sum(padded[i:i+len(line)]*weight for i,weight in enumerate((1,4,6,4,1)))/16
        smoothing += 1
    return line.astype(np.float32), {'applied': True, 'reliable_fraction': float(reliable.mean()),
                                     'max_displacement': float(np.linalg.norm(line, axis=-1).max()),
                                     'spatial_smoothing_passes':smoothing}


def _mesh(flow, confidence, rect, width, height, band, maximum):
    """Blend side extrapolations at corners, with zero support beyond the band."""
    x0, y0, x1, y1 = rect
    yy, xx = np.mgrid[:height, :width].astype(np.float32)
    displacement = np.zeros((height, width, 2), dtype=np.float32)
    total = np.zeros((height, width), dtype=np.float32)
    stats = []
    for side, margin in enumerate((x0, y0, width-x1, height-y1)):
        if not margin:
            stats.append({'applied': False, 'reason': 'no_expansion'})
            continue
        line, report = _boundary_line(flow, confidence, side=side, maximum=maximum)
        stats.append(report)
        if not report['applied']:
            continue
        distance = (x0-.5-xx, y0-.5-yy, xx-x1+.5, yy-y1+.5)[side]
        # Corner directions are averaged BEFORE fading. Fading each side and
        # clamping the weight sum to one creates a discontinuity at side joins.
        weight = distance.clip(0)
        tangent = yy-y0 if side in (0, 2) else xx-x0
        indices = tangent.clip(0, len(line)-1).astype(np.int32)
        displacement += line[indices]*weight[..., None]
        total += weight
    displacement /= np.maximum(total, .001)[..., None]
    outside_x = np.maximum(x0-.5-xx, xx-x1+.5).clip(0)
    outside_y = np.maximum(y0-.5-yy, yy-y1+.5).clip(0)
    t = (np.hypot(outside_x,outside_y)/band).clip(0,1)
    displacement *= (1-3*t*t+2*t*t*t)[...,None]
    # Reject folded/inverted mappings rather than returning torn image geometry.
    dx = np.gradient(displacement, axis=1)
    dy = np.gradient(displacement, axis=0)
    determinant = (1+dx[...,0])*(1+dy[...,1])-dy[...,0]*dx[...,1]
    exterior = np.ones((height, width), dtype=bool)
    exterior[y0:y1, x0:x1] = False
    # Source-edge jump is intentional: source will be pasted verbatim. Check the
    # generated interior of the mesh, not a derivative crossing the ownership edge.
    exterior[max(0,y0-1):min(height,y1+1), max(0,x0-1):min(width,x1+1)] = False
    if np.any(determinant[exterior] < .2):
        return np.zeros_like(displacement), stats, 'rejected_folded_mesh'
    return displacement, stats, 'bounded_exterior_registration'


def register_outpaint_candidate(source, candidate, plan, *, start_frame=0,
                                band_pixels=64, max_displacement=8.0):
    """Return registered candidate + report; no source/candidate input mutation.

    Source-area pixels in the returned candidate remain the original decoded
    candidate (needed for independent color estimation). Compositing source RGB
    is deliberately a separate ownership step and must not be bypassed.
    """
    checked = validate_outpaint_plan(plan)
    # Reuse the existing strict shape/dtype/device/range/sequence contract.
    composite_outpaint_frames(source, candidate, checked, start_frame=start_frame)
    _integer(band_pixels, 'band_pixels', 8, 512)
    maximum = _finite(max_displacement, 'max_displacement', 0, 16)
    if source.dtype != torch.uint8 and not source.is_floating_point():
        raise ValueError('registration requires uint8 or normalized floating RGB')
    if source.is_floating_point() and any(torch.any((t[..., :3] < 0) | (t[..., :3] > 1)) for t in (source, candidate)):
        raise ValueError('registration floating RGB must be normalized')
    result = candidate.clone()
    rect = checked['output']['source_rect']
    x0, y0, x1, y1 = rect
    if min(source.shape[1:3]) < 32 or maximum == 0 or not checked['output']['has_outpaint']:
        return result, {'algorithm': 'not_applied', 'reason': 'small_source_or_disabled', 'perceptual_acceptance': False}
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError('Experimental outpaint registration needs OpenCV (opencv-python); it does not download a model') from error
    if not hasattr(cv2, 'DISOpticalFlow_create'):
        raise RuntimeError('OpenCV build lacks DISOpticalFlow_create')
    normalizer = 255 if source.dtype == torch.uint8 else 1
    rows = []
    height, width = candidate.shape[1:3]
    yy, xx = np.mgrid[:y1-y0, :x1-x0].astype(np.float32)
    cy, cx = np.mgrid[:height, :width].astype(np.float32)
    for i in range(len(source)):
        original = source[i, ..., :3].detach().cpu().float().numpy()/normalizer
        decoded = candidate[i, ..., :3].detach().cpu().float().numpy()/normalizer
        a = cv2.cvtColor(np.round(original*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(np.round(decoded[y0:y1,x0:x1]*255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        if min(float(a.std()), float(b.std())) < 2:
            rows.append({'frame': start_frame+i, 'status': 'insufficient_texture'})
            continue
        estimator = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        estimator.setGradientDescentIterations(40)
        estimator.setVariationalRefinementIterations(10)
        forward = estimator.calc(a, b, None)
        reverse = estimator.calc(b, a, None)
        inverse = cv2.remap(reverse, xx+forward[...,0], yy+forward[...,1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        error = np.linalg.norm(forward+inverse, axis=-1)
        displacement, sides, status = _mesh(forward, error, rect, width, height, band_pixels, maximum)
        remapped = cv2.remap(decoded, cx+displacement[...,0], cy+displacement[...,1],
                             cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        active = np.any(displacement != 0, axis=-1)
        active[y0:y1,x0:x1] = False
        converted = torch.from_numpy(remapped.copy()).to(device=candidate.device)
        converted = (converted*255).round().to(candidate.dtype) if candidate.dtype == torch.uint8 else converted.to(candidate.dtype)
        mask = torch.from_numpy(active).to(device=candidate.device)
        result[i, ..., :3] = torch.where(mask[...,None], converted, result[i, ..., :3])
        rows.append({'frame':start_frame+i, 'status':status, 'sides':sides, 'changed_pixels':int(active.sum())})
    return result, {'algorithm':'bidirectional_dis_bounded_exterior_mesh_v1', 'frames':rows,
                    'band_pixels':band_pixels, 'max_displacement':maximum,
                    'source_candidate_interior_unchanged':torch.equal(result[:,y0:y1,x0:x1], candidate[:,y0:y1,x0:x1]),
                    'perceptual_acceptance':False, 'color_correction':False}
