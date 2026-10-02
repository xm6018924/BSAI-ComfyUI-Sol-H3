"""Independent Meridian material/warp stages. Does not touch long-video internals."""

import base64
import hashlib
import io as bytes_io
from pathlib import Path
import sys

import numpy as np

from .meridian_checkpoint_io import canonical_identity, file_sha, part_header
from .meridian_media import image_u8, video_window
from .meridian_plan import bucket, canonical_plan, source_map
from .meridian_sources import FILES, REVISION, omega_imports, source_module
from .meridian_stage_store import StageStore
from .meridian_resources import owned_offload, note_cleanup_failure


def check():
    import comfy.model_management as mm

    mm.throw_exception_if_processing_interrupted()


def cancelled():
    check()
    return False


def tensor_sha(tensor):
    """Hash actual selected RGB bytes, not a source filename or mtime."""
    digest = hashlib.sha256()
    for frame in tensor:
        check()
        digest.update(frame.contiguous().numpy().tobytes())
    return digest.hexdigest()


def file_identity(path):
    path = Path(path).resolve(strict=True)
    return dict(
        path=str(path),
        bytes=path.stat().st_size,
        sha256=file_sha(path, cancelled=cancelled),
    )


def file_descriptor(path):
    """Not a content identity. Full payload SHA is required by its consuming stage."""
    path = Path(path).resolve(strict=True)
    stat = path.stat()
    return dict(
        path=str(path),
        bytes=stat.st_size,
        mtime_ns=stat.st_mtime_ns,
        content_verification="deferred_to_consuming_stage_not_cache_identity",
    )


def validate_checkpoint(path):
    info = part_header(path)
    meta = info["metadata"]
    if (
        meta.get("dmd_merged") != "true"
        or meta.get("t8_recipe") != "dmd_grid4_forward3_shift3_3"
    ):
        raise ValueError(
            "Select converted Meridian merged-DMD ConvRot INT8, not an ordinary H3 checkpoint"
        )
    int8 = [spec for spec in info["header"].values() if spec["dtype"] == "I8"]
    if len(int8) != 250:
        raise ValueError(
            "Meridian native checkpoint requires its250 ConvRot INT8 layers"
        )
    return meta


def config(
    model_path,
    vae_path,
    source,
    omega_repo,
    omega_checkpoint,
    cache_root,
    cache_gib=20.0,
    model=None,
    video_vae=None,
):
    from .meridian_sources import resolve_directory, resolve_file
    import folder_paths

    models = Path(folder_paths.models_dir)
    source = resolve_directory(
        source, "MERIDIAN_SOURCE_DIR", models / "meridian/source"
    )
    # Validate thin adapters now, but import geometry only at material execution.
    sources = {
        name: file_sha(source / "recam" / f"{name}.py", cancelled=cancelled)
        for name in FILES
    }
    if sources != FILES:
        raise ValueError(
            f"Use Meridian source revision {REVISION}, then separately audit any upstream update"
        )
    checkpoint = resolve_file(
        model_path,
        "MERIDIAN_MODEL",
        models / "meridian/meridian_dmd_int8_convrot_comfy.safetensors",
    )
    metadata = validate_checkpoint(checkpoint)
    omega_repo = resolve_directory(
        omega_repo, "VGGT_OMEGA_DIR", models / "meridian/vggt-omega"
    )
    omega = resolve_file(
        omega_checkpoint,
        "VGGT_OMEGA_CKPT",
        omega_repo / "checkpoints/vggt_omega_1b_512.pt",
    )
    if not (omega_repo / "vggt_omega/__init__.py").is_file():
        raise ValueError("Correct authorized Omega source required, not ordinary VGGT")
    # Bind all Python implementation files, not merely the public models entry.
    omega_sources = {
        str(p.relative_to(omega_repo)): file_sha(p, cancelled=cancelled)
        for p in sorted((omega_repo / "vggt_omega").rglob("*.py"))
    }
    if not omega_sources:
        raise ValueError("Omega source directory is empty")
    own_vae = None
    if video_vae is None:
        own_vae = resolve_file(
            vae_path,
            "MERIDIAN_VIDEO_VAE",
            models / "vae/minimax_h3_video_vae_fp16.safetensors",
        )
    elif vae_path:
        raise ValueError(
            "Use either standard video_vae socket or VAE file override, not both"
        )
    root = (
        Path(
            cache_root or Path(folder_paths.get_temp_directory()) / "t8-meridian-cache"
        ).resolve()
        / "stages-v1"
    )
    if not np.isfinite(cache_gib) or cache_gib < 0:
        raise ValueError("Nonnegative finite cache GiB required")
    warnings = []
    if model is not None:
        warnings.append(
            "External MODEL patches preserved; converted-base identity cannot be certified. Sampling disk cache disabled."
        )
    if video_vae is not None:
        warnings.append(
            "External VAE preserved and delegated unchanged. VAE disk cache disabled; dynamic residency is caller-owned."
        )
    project = Path(__file__).resolve().parents[1]
    core = project.parents[1]
    implementation = {
        str(p.relative_to(project)): file_sha(p, cancelled=cancelled)
        for p in sorted((project / "h3_t8").glob("meridian*.py"))
    }
    # This adapter calls the original scheduler. Bind its real implementation too;
    # the guessed minimax/conditioning.py does not exist in this Core revision.
    implementation["h3_t8/sampling.py"] = file_sha(
        project / "h3_t8/sampling.py", cancelled=cancelled
    )
    implementation.update(
        {
            str(p.relative_to(core)): file_sha(p, cancelled=cancelled)
            for p in [
                core / "comfy/sd.py",
                core / "comfy/ldm/minimax/model.py",
                core / "comfy/ldm/minimax/vae.py",
                core / "comfy/ops.py",
                core / "comfy/model_patcher.py",
                core / "comfy/model_management.py",
                core / "comfy/sample.py",
                core / "comfy/model_base.py",
                core / "comfy/model_sampling.py",
                core / "comfy/samplers.py",
                core / "comfy/k_diffusion/sampling.py",
                core / "comfy/conds.py",
                core / "comfy/latent_formats.py",
                core / "comfy/nested_tensor.py",
                core / "comfy/patcher_extension.py",
                core / "comfy/ldm/modules/attention.py",
            ]
            if p.is_file()
        }
    )
    return dict(
        schema="t8.meridian.runtime.v1",
        checkpoint=file_descriptor(checkpoint),
        metadata=metadata,
        vae=file_descriptor(own_vae) if own_vae else None,
        source=str(source),
        source_files=sources,
        source_revision=REVISION,
        omega=file_descriptor(omega),
        omega_repo=str(omega_repo),
        omega_sources=omega_sources,
        cache_root=str(root),
        budget=int(cache_gib * 1024**3),
        model=model,
        video_vae=video_vae,
        warnings=warnings,
        implementation=implementation,
    )


def store(runtime):
    return StageStore(runtime["cache_root"], runtime["budget"], cancelled)


def roi(full, geometry):
    """Training1280 letterbox, nearest log-aspect ROI crop, not stretching."""
    import torch

    geo, (h, w) = geometry, full.shape[1:3]
    canvas, condition = bucket(w, h)
    scale = geo.FULL / max(h, w)
    cw, ch = round(w * scale), round(h * scale)
    ox, oy = (geo.FULL - cw) // 2, (geo.FULL - ch) // 2
    padded = torch.zeros(len(full), geo.FULL, geo.FULL, 3, dtype=torch.uint8)
    padded[:, oy : oy + ch, ox : ox + cw] = geo.resize_u8(full, (ch, cw))
    aspect = canvas[0] / canvas[1]
    bw, bh = (cw, round(cw / aspect)) if cw / ch <= aspect else (round(ch * aspect), ch)
    box = [ox + (cw - bw) // 2, oy + (ch - bh) // 2, bw, bh, canvas[0] / bw]
    return padded, box, list(canvas), list(condition), [ox, oy, cw, ch]


def gauge(geometry, box):
    """Fixed first-frame central10% depth unit, finite valid-picture fallback."""
    import torch

    x, y, w, h, _ = box
    px, py = (x + w * 0.5) * 0.4, (y + h * 0.5) * 0.4
    d, k = geometry["depth"][0], geometry["keep"][0]
    ya, yb = max(0, round(py - h * 0.02)), min(512, round(py + h * 0.02))
    xa, xb = max(0, round(px - w * 0.02)), min(512, round(px + w * 0.02))
    sub = d[ya:yb, xa:xb][k[ya:yb, xa:xb]]
    if sub.numel() < 20:
        picture = torch.zeros_like(k)
        picture[
            max(0, round(y * 0.4)) : min(512, round((y + h) * 0.4)),
            max(0, round(x * 0.4)) : min(512, round((x + w) * 0.4)),
        ] = True
        sub = d[k & picture & (d > 0)]
    zm = float(sub.median()) if sub.numel() else float("nan")
    if not np.isfinite(zm) or zm <= 0:
        raise ValueError(
            "No positive usable depth in selected picture; geometry preparation failed"
        )
    K = geometry["intr"][0]
    pivot = [
        (px - float(K[0, 2])) / float(K[0, 0]),
        (py - float(K[1, 2])) / float(K[1, 1]),
        1.0,
    ]
    return zm, pivot


def prepare(runtime, image=None, video=None, start_frame=0, window_frames=73):
    import torch

    if (image is None) == (video is None):
        raise ValueError("Connect exactly one IMAGE or VIDEO material")
    if (
        type(start_frame) is not int
        or start_frame < 0
        or type(window_frames) is not int
        or window_frames < 1
    ):
        raise ValueError("Invalid material window")
    geo = source_module(runtime["source"], "geometry", check)
    if image is not None:
        if start_frame != 0:
            raise ValueError("Still IMAGE has only source frame0")
        pixels, clock = (
            image_u8(image),
            dict(normalized_fps="24", selection="one_still_frame"),
        )
        kind, end = "image", 0
    else:
        pixels, clock = video_window(video, start_frame, window_frames, check)
        kind, end = "video", start_frame + window_frames - 1
    full, box, canvas, condition, letterbox = roi(pixels, geo)
    # Only geometry needs Omega. Main34GB model and VAE are not read by previews.
    omega_identity = file_identity(runtime["omega"]["path"])
    contract = dict(
        schema="t8.meridian.geometry.v1",
        material_sha256=tensor_sha(pixels),
        kind=kind,
        start=start_frame,
        end=end,
        shape=list(pixels.shape),
        clock=clock,
        omega=omega_identity,
        omega_sources=runtime["omega_sources"],
        algorithm=runtime["source_files"]["geometry"],
        implementation=runtime["implementation"],
        box=box,
        canvas=canvas,
    )
    del pixels

    def operation():
        import comfy.model_management as mm
        import comfy.model_patcher

        check()
        with omega_imports(runtime["omega_repo"]):
            from vggt_omega.models import VGGTOmega

            model = VGGTOmega().eval()
            weights = torch.load(
                runtime["omega"]["path"],
                map_location="cpu",
                weights_only=True,
                mmap=True,
            )
            loaded = model.load_state_dict(weights, strict=True)
            del weights
            if loaded.missing_keys or loaded.unexpected_keys:
                raise ValueError("Omega512 strict state mismatch")
            device = mm.get_torch_device()
            patcher = comfy.model_patcher.ModelPatcher(
                model, load_device=device, offload_device=torch.device("cpu")
            )
            try:
                mm.load_models_gpu([patcher], force_full_load=True)
                check()
                with torch.inference_mode():
                    result = geo.reconstruct(model, geo.to_input(full).to(device))
                check()
                result = {name: value.detach().cpu() for name, value in result.items()}
                if file_identity(runtime["omega"]["path"]) != omega_identity:
                    raise ValueError(
                        "Omega weight file changed during geometry preparation"
                    )
                if not all(torch.isfinite(value).all() for value in result.values()):
                    raise ValueError("Omega returned nonfinite geometry")
                zm, pivot = gauge(result, box)
                return dict(
                    geometry=result,
                    full=full,
                    box=box,
                    canvas=canvas,
                    condition_canvas=condition,
                    letterbox=letterbox,
                    zm=zm,
                    pivot=pivot,
                )
            finally:
                # Only our own geometry patcher; no unload_all_models or global interrupt.
                pending = sys.exc_info()[1]
                try:
                    owned_offload(
                        lambda: patcher.unpatch_model(torch.device("cpu")), "Omega"
                    )
                except BaseException as cleanup_error:
                    if pending is None:
                        raise
                    note_cleanup_failure(pending, cleanup_error, "Omega")

    value, receipt = store(runtime).run("geometry", contract, operation)
    material = dict(
        value,
        identity=canonical_identity(contract),
        start=start_frame,
        end=end,
        kind=kind,
        runtime=runtime,
        video=video,
        receipt=receipt,
        clock=clock,
    )
    return material


def editor_payload(material):
    """Bounded subsampled points/thumbnails. This is geometry, NOT a generated preview."""
    import torch
    from PIL import Image

    geo = source_module(material["runtime"]["source"], "geometry", check)
    geometry, full = material["geometry"], material["full"]
    W = geometry["c2w"][0]
    Wi = torch.linalg.inv(W)
    frames = np.unique(
        np.linspace(0, len(full) - 1, min(8, len(full))).round().astype(int)
    ).tolist()
    clouds = []
    for i in frames:
        check()
        # 512-space is enough for interactive picking, never used as final warp.
        points = geo.unproject(
            geometry["depth"][i : i + 1],
            geometry["extr"][i : i + 1],
            geometry["intr"][i : i + 1],
        )[0]
        valid = (
            geometry["keep"][i]
            & (geometry["depth"][i] > 0)
            & (geometry["depth"][i] < 10 * material["zm"])
        )
        pts = points[valid]
        colors = geo.resize_u8(full[i : i + 1], (512, 512))[0][valid]
        if len(pts) > 1024:
            sample = torch.linspace(0, len(pts) - 1, 1024).round().long()
            pts, colors = pts[sample], colors[sample]
        pts = (pts @ Wi[:3, :3].T + Wi[:3, 3]) / material["zm"]
        image = Image.fromarray(full[i].numpy())
        image.thumbnail((320, 320))
        buffer = bytes_io.BytesIO()
        image.save(buffer, format="JPEG", quality=78)
        clouds.append(
            dict(
                src=material["start"] + i,
                points=pts.tolist(),
                colors=colors.tolist(),
                thumbnail="data:image/jpeg;base64,"
                + base64.b64encode(buffer.getvalue()).decode(),
            )
        )
    return dict(
        schema="t8.meridian.editor.v1",
        identity=material["identity"],
        start=material["start"],
        end=material["end"],
        kind=material["kind"],
        pivot=material["pivot"],
        box=material["box"],
        canvas=material["canvas"],
        condition_canvas=material["condition_canvas"],
        clouds=clouds,
        source_poses=(Wi[None] @ geometry["c2w"]).tolist(),
        source_intrinsics=geometry["intr"].tolist(),
        label="Subsampled geometry preview only; final warp uses full1280 depth splats; ROI crop, not stretch",
    )


def warp_material(material, plan):
    import torch

    plan = canonical_plan(plan, material)
    runtime = material["runtime"]
    geo = source_module(runtime["source"], "geometry", check)
    path = source_module(runtime["source"], "path", check)
    indices = [s - material["start"] for s in source_map(plan)]
    contract = dict(
        schema="t8.meridian.warp.v1",
        geometry=material["identity"],
        plan=plan,
        source_files=runtime["source_files"],
        implementation=runtime["implementation"],
    )

    def operation():
        check()
        geometry = material["geometry"]
        # Keep camera interpolation independent of all source-time breakpoints.
        cameras, _, focal, _ = path.plan_path(
            [dict(k, src=0) for k in plan["camera_keys"]],
            plan["frames"],
            material["zm"],
        )
        if plan["mode"] == "source_camera":
            poses = geometry["c2w"][indices]
        else:
            poses = geometry["c2w"][0:1] @ torch.as_tensor(
                cameras, dtype=geometry["c2w"].dtype
            )
        images, covers, sources = [], [], []
        x, y, w, h, f = material["box"]
        canvas, condition = material["canvas"], material["condition_canvas"]
        # One source at a time: no73-frame expanded1280 point-cloud allocation.
        for i, j in enumerate(indices):
            check()
            S = {k: v[j : j + 1] for k, v in geometry.items()}
            K = S["intr"].clone()
            K[:, :2, :2] *= focal[i]
            rendered, covered = geo.warp(
                S,
                torch.linalg.inv(poses[i : i + 1]),
                K,
                material["full"][j : j + 1],
                material["box"],
                canvas,
            )
            images.append(geo.resize_u8(rendered, condition[::-1]))
            covers.append(float(covered.float().mean()))
            sources.append(
                geo.resize_u8(
                    material["full"][j : j + 1, y : y + h, x : x + w], condition[::-1]
                )
            )
        check()
        return dict(
            condition=torch.cat(sources),
            render=torch.cat(images),
            coverage=covers,
            poses=poses.cpu(),
            canvas=canvas,
            condition_canvas=condition,
            plan=plan,
        )

    value, receipt = store(runtime).run("warp", contract, operation)
    return dict(
        value, identity=canonical_identity(contract), receipt=receipt, material=material
    )
