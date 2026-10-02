"""Same-origin director services.

D1 owns the persistent project/asset contract. D2a–D2c add explicit,
validated native-generation routes that use Core's normal in-process queue.
D3 either compiles the four compatible H3 route patches or hands the user an
exact allow-listed native workflow; it never disguises a handoff as a queue.
"""

from __future__ import annotations

import asyncio
from functools import wraps
from pathlib import Path
import uuid

from .director_project import (
    ProjectConflict,
    ProjectStore,
    compile_project,
    contained,
    identity,
    new_project,
    validate_project,
)
from .director_generation import (
    build_director_generation_prompt,
    cancel_director_prompt,
    director_model_catalog,
    director_job_status,
    queue_director_prompt,
)
from .director_capabilities import inspect_director_capabilities
from .director_d3 import export_d3_package, handoff_d3_route, inspect_d3_routes

PREFIX = "/minimax_h3_t8/director"
_REGISTERED = False


def get_store():
    import folder_paths

    return ProjectStore(
        folder_paths.get_user_directory(), folder_paths.get_input_directory()
    )


def register_director_routes():
    global _REGISTERED
    if _REGISTERED:
        return True
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        # CPU/schema tools do not necessarily have the Core server module initialized.
        return False
    server = getattr(PromptServer, "instance", None)
    if server is None:
        return False
    routes = server.routes

    def guarded(function):
        @wraps(function)
        async def call(request):
            try:
                return await function(request)
            except ProjectConflict as error:
                return web.json_response(
                    {"error": str(error), "code": "revision_conflict"}, status=409
                )
            except FileNotFoundError:
                return web.json_response(
                    {"error": "项目或素材不存在，请重连；当前草稿不会被覆盖"},
                    status=404,
                )
            except (ValueError, KeyError, TypeError) as error:
                return web.json_response({"error": str(error)}, status=400)
            except Exception as error:
                import logging

                logging.exception("Director service failed")
                return web.json_response(
                    {
                        "error": f"保存／读取失败，可保留草稿后重试：{type(error).__name__}"
                    },
                    status=500,
                )

        return call

    @routes.get(PREFIX + "/projects")
    @guarded
    async def projects(_request):
        return web.json_response(
            {"projects": await asyncio.to_thread(get_store().list)}
        )

    @routes.get(PREFIX + "/projects/{project_id}")
    @guarded
    async def load(request):
        project = await asyncio.to_thread(
            get_store().load, request.match_info["project_id"]
        )
        return web.json_response(project)

    @routes.post(PREFIX + "/projects/{project_id}")
    @guarded
    async def save(request):
        body = await request.json()
        if identity(request.match_info["project_id"]) != body["project"]["id"]:
            raise ValueError("项目路径与内容身份不一致")
        result = await asyncio.to_thread(
            get_store().save, body["project"], body["expected_revision"]
        )
        return web.json_response(result)

    @routes.post(PREFIX + "/compile")
    @guarded
    async def compile(request):
        body = await request.json()
        result = await asyncio.to_thread(compile_project, body["project"], get_store())
        return web.json_response(result)

    @routes.post(PREFIX + "/validate")
    @guarded
    async def validate(request):
        body = await request.json()
        return web.json_response({"project": validate_project(body["project"])})

    @routes.post(PREFIX + "/export")
    @guarded
    async def export(request):
        body = await request.json()
        result = await asyncio.to_thread(compile_project, body["project"], get_store())
        if not result["ready"]:
            return web.json_response(
                {
                    "error": "预检未通过，保留项目但不能导出可执行预检图",
                    "report": result,
                },
                status=422,
            )
        return web.json_response(
            {
                **export_preflight_workflow(body["project"], body["shot_id"]),
                "report": result,
            }
        )

    @routes.post(PREFIX + "/generate")
    @guarded
    async def generate(request):
        """Queue the validated D2a–D2c recipe selected by the current shot."""
        body = await request.json()
        built = await asyncio.to_thread(
            build_director_generation_prompt,
            body["project"],
            body["shot_id"],
            get_store(),
            seed=int(body.get("seed", 26091901)),
        )
        prompt_id = await queue_director_prompt(
            built["prompt"], body.get("client_id")
        )
        return web.json_response(
            {
                "prompt_id": prompt_id,
                "recipe": built["recipe"],
                "d3_routes": built.get("d3_routes", []),
                "seed": built["seed"],
                "turbo_lora": built["turbo_lora"],
                "report": built["report"],
            },
            status=202,
        )

    @routes.post(PREFIX + "/d3/compile")
    @guarded
    async def d3_compile(request):
        """Compile the selected D3 graph without submitting it to Core."""
        body = await request.json()
        built = await asyncio.to_thread(
            build_director_generation_prompt,
            body["project"],
            body["shot_id"],
            get_store(),
            seed=int(body.get("seed", 26091901)),
        )
        return web.json_response(
            {
                "schema": "t8.minimax_h3.director_d3_compiled_graph.v1",
                "recipe": built["recipe"],
                "d3_routes": built.get("d3_routes", []),
                "seed": built["seed"],
                "nodes": {
                    str(node_id): {
                        "class_type": node.get("class_type"),
                        "inputs": node.get("inputs", {}),
                    }
                    for node_id, node in built["prompt"].items()
                },
                "report": built["report"],
                "warning": "只编译图，不排队、不加载模型、不代表 GPU 或感知质量通过。",
            }
        )

    @routes.get(PREFIX + "/jobs/{prompt_id}")
    @guarded
    async def job(request):
        return web.json_response(
            director_job_status(request.match_info["prompt_id"])
        )

    @routes.post(PREFIX + "/jobs/{prompt_id}/cancel")
    @guarded
    async def cancel(request):
        return web.json_response(
            cancel_director_prompt(request.match_info["prompt_id"])
        )

    @routes.post(PREFIX + "/assets")
    @guarded
    async def upload(request):
        store = get_store()
        reader = await request.multipart()
        part = await reader.next()
        if part is None or part.name != "file" or not part.filename:
            raise ValueError("上传必须包含 file")
        asset_id = str(uuid.uuid4())
        suffix = Path(part.filename).suffix.lower()
        if suffix not in {
            ".png",
            ".jpg",
            ".jpeg",
            ".webp",
            ".bmp",
            ".gif",
            ".mp4",
            ".mov",
            ".mkv",
            ".webm",
            ".wav",
            ".mp3",
            ".flac",
            ".ogg",
            ".m4a",
            ".aac",
        }:
            raise ValueError("请选择标准图片、视频或音频文件")
        path = contained(store.input_root, f"t8_director/{asset_id}/source{suffix}")
        path.parent.mkdir(parents=True, exist_ok=False)
        size = 0
        try:
            with path.open("xb") as stream:
                while chunk := await part.read_chunk(1024 * 1024):
                    size += len(chunk)
                    if size > 1024 * 1024 * 1024:
                        raise ValueError(
                            "单素材上传上限1GiB，请分段或压缩后重试；未截断保存"
                        )
                    await asyncio.to_thread(stream.write, chunk)
            asset = await asyncio.to_thread(
                store.register_asset, path, asset_id, part.filename
            )
            return web.json_response(asset, status=201)
        except BaseException:
            path.unlink(missing_ok=True)  # Only this incomplete, server-created upload.
            path.parent.rmdir()
            raise

    @routes.get(PREFIX + "/assets/{asset_id}")
    @guarded
    async def asset(request):
        store = get_store()
        asset = await asyncio.to_thread(store.asset, request.match_info["asset_id"])
        return web.FileResponse(contained(store.input_root, asset["server_path"]))

    @routes.get(PREFIX + "/assets/{asset_id}/input-preview")
    @guarded
    async def input_preview(request):
        store = get_store()
        prepared = await asyncio.to_thread(
            store.prepare_image,
            request.match_info["asset_id"],
            int(request.query["width"]),
            int(request.query["height"]),
        )
        return web.FileResponse(contained(store.input_root, prepared["server_path"]))

    @routes.get(PREFIX + "/ui")
    async def ui(_request):
        return web.FileResponse(
            Path(__file__).resolve().parents[1] / "web" / "director" / "index.html"
        )

    @routes.get(PREFIX + "/session.mjs")
    async def session(_request):
        return web.FileResponse(
            Path(__file__).resolve().parents[1] / "web" / "director" / "session.mjs"
        )

    @routes.get(PREFIX + "/default")
    async def default(_request):
        return web.json_response(new_project())

    @routes.get(PREFIX + "/capabilities")
    @guarded
    async def capabilities(_request):
        """Report D3 native entry points without pretending to queue them."""
        import nodes

        return web.json_response(
            inspect_director_capabilities(nodes.NODE_CLASS_MAPPINGS.keys())
        )

    @routes.get(PREFIX + "/models")
    @guarded
    async def models(_request):
        """Return only installed H3-compatible models for the Director selectors."""
        return web.json_response(director_model_catalog())

    @routes.get(PREFIX + "/d3/routes")
    @guarded
    async def d3_routes(_request):
        """List D3 native hand-off routes without touching a project or queue."""
        import nodes

        return web.json_response(inspect_d3_routes(node_ids=nodes.NODE_CLASS_MAPPINGS.keys()))

    @routes.post(PREFIX + "/d3/preflight")
    @guarded
    async def d3_preflight(request):
        """Preflight one saved Director shot before handing it to a D3 route."""
        body = await request.json()
        return web.json_response(
            await asyncio.to_thread(
                inspect_d3_routes,
                body.get("project"),
                body.get("shot_id"),
                get_store(),
                capability=body.get("capability"),
            )
        )

    @routes.post(PREFIX + "/d3/handoff")
    @guarded
    async def d3_handoff(request):
        """Return an exact allow-listed native workflow/README without queuing."""
        body = await request.json()
        result = await asyncio.to_thread(
            handoff_d3_route,
            str(body.get("capability", "")),
            body.get("file"),
        )
        return web.json_response(result)

    @routes.post(PREFIX + "/d3/package")
    @guarded
    async def d3_package(request):
        """Export the current project plus an exact native route hand-off."""
        body = await request.json()
        result = await asyncio.to_thread(
            export_d3_package,
            str(body.get("capability", "")),
            body.get("project"),
            body.get("shot_id"),
            get_store(),
        )
        return web.json_response(result)

    _REGISTERED = True
    return True


def export_preflight_workflow(project, shot_id):
    """A real native CPU contract graph, not a mislabeled runnable GPU recipe."""
    from .director_project import canonical, validate_project

    project = validate_project(project)
    if shot_id not in {s["id"] for s in project["doc"]["shots"]}:
        raise ValueError("镜头身份不存在")
    values = [canonical(project), shot_id]
    api = {
        "1": {
            "class_type": "MiniMaxH3DirectorProjectT8",
            "inputs": {"project_json": values[0], "shot_id": shot_id},
            "_meta": {"title": "曜石导演台 · D1 CPU预检（不生成）"},
        }
    }
    workflow = {
        "id": str(uuid.uuid4()),
        "version": 0.4,
        "last_node_id": 1,
        "last_link_id": 0,
        "nodes": [
            {
                "id": 1,
                "type": "MiniMaxH3DirectorProjectT8",
                "pos": [160, 140],
                "size": [520, 260],
                "flags": {},
                "order": 0,
                "mode": 0,
                "inputs": [],
                "outputs": [
                    {"name": name, "type": kind, "links": None}
                    for name, kind in (
                        ("compiled_prompt", "STRING"),
                        ("width", "INT"),
                        ("height", "INT"),
                        ("length", "INT"),
                        ("media_map_json", "STRING"),
                        ("report_json", "STRING"),
                    )
                ],
                "properties": {
                    "Node name for S&R": "MiniMaxH3DirectorProjectT8",
                    "cnr_id": "minimax-h3-audio-t8",
                },
                "widgets_values": values,
            }
        ],
        "links": [],
        "groups": [],
        "config": {},
        "extra": {
            "t8_director": {
                "project_id": project["id"],
                "scope": "D1 CPU preflight only; not generation",
            }
        },
    }
    return {"workflow": workflow, "api_snapshot": api}
