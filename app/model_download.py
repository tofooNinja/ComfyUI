"""Server-side model downloads for the frontend's "Missing Models" flow.

POST   /models/download            {url, folder, filename, overwrite?} -> {task_id, ...}
GET    /models/download            list of all tasks (this process only)
GET    /models/download/{task_id}  task status: created|running|completed|failed

Files are streamed to ``<folder path>/<filename>.part`` and renamed on success.
Only https downloads from an allowlist of model hosts are accepted; the target
must be a registered model folder and the filename a bare basename whose
extension the folder accepts.
"""

import asyncio
import logging
import os
import uuid
from urllib.parse import urlparse

import aiohttp
from aiohttp import web

import folder_paths

ALLOWED_HOSTS = frozenset({"huggingface.co", "civitai.com"})
# Folders under folder_paths that are not model directories.
FOLDER_BLACKLIST = frozenset({"custom_nodes", "configs", "datasets"})
CHUNK_SIZE = 1024 * 1024


class ModelDownloadError(ValueError):
    pass


def validate_download_request(url: str, folder: str, filename: str) -> str:
    """Validate a request and return the absolute destination path."""
    if not isinstance(url, str) or not isinstance(folder, str) or not isinstance(filename, str):
        raise ModelDownloadError("url, folder and filename must be strings")

    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname is None:
        raise ModelDownloadError("url must be https")
    if parsed.hostname.lower() not in ALLOWED_HOSTS:
        raise ModelDownloadError(f"downloads are only allowed from: {', '.join(sorted(ALLOWED_HOSTS))}")

    if folder in FOLDER_BLACKLIST or folder not in folder_paths.folder_names_and_paths:
        raise ModelDownloadError(f"unknown model folder: {folder}")
    dirs, extensions = folder_paths.folder_names_and_paths[folder]
    if not dirs or "folder" in extensions:
        raise ModelDownloadError(f"folder does not accept file downloads: {folder}")

    if (
        not filename
        or filename in (".", "..")
        or os.path.basename(filename) != filename
        or "/" in filename
        or "\\" in filename
        or "\0" in filename
        or filename.startswith(".")
    ):
        raise ModelDownloadError("filename must be a bare file name")
    if extensions and not any(filename.lower().endswith(ext) for ext in extensions if ext):
        raise ModelDownloadError(f"file extension not allowed for folder {folder}")

    target_dir = os.path.abspath(dirs[0])
    destination = os.path.abspath(os.path.join(target_dir, filename))
    if os.path.dirname(destination) != target_dir:
        raise ModelDownloadError("invalid destination")
    return destination


class ModelDownloadManager:
    def __init__(self) -> None:
        self.tasks: dict[str, dict] = {}
        self.active_paths: set[str] = set()

    def add_routes(self, routes):
        @routes.post("/models/download")
        async def start_download(request):
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "invalid JSON body"}, status=400)
            if not isinstance(body, dict):
                return web.json_response({"error": "invalid JSON body"}, status=400)

            url = body.get("url")
            folder = body.get("folder")
            filename = body.get("filename")
            overwrite = body.get("overwrite", False) is True
            try:
                destination = validate_download_request(url, folder, filename)
            except ModelDownloadError as e:
                return web.json_response({"error": str(e)}, status=400)

            if destination in self.active_paths:
                return web.json_response({"error": "a download to this path is already running"}, status=409)
            if os.path.exists(destination) and not overwrite:
                return web.json_response({"error": "file already exists"}, status=409)

            task_id = str(uuid.uuid4())
            task = {
                "task_id": task_id,
                "status": "created",
                "url": url,
                "folder": folder,
                "filename": filename,
                "downloaded": 0,
                "total": None,
                "progress": 0.0,
                "error": None,
            }
            self.tasks[task_id] = task
            self.active_paths.add(destination)
            asyncio.create_task(self._download(task, destination))
            return web.json_response(task, status=202)

        @routes.get("/models/download")
        async def list_downloads(request):
            return web.json_response(list(self.tasks.values()))

        @routes.get("/models/download/{task_id}")
        async def get_download(request):
            task = self.tasks.get(request.match_info["task_id"])
            if task is None:
                return web.json_response({"error": "unknown task"}, status=404)
            return web.json_response(task)

    async def _download(self, task: dict, destination: str) -> None:
        part_path = destination + ".part"
        headers = {}
        hf_token = os.environ.get("HF_TOKEN")
        if hf_token and urlparse(task["url"]).hostname.lower() == "huggingface.co":
            headers["Authorization"] = f"Bearer {hf_token}"

        task["status"] = "running"
        logging.info("Downloading model %s -> %s", task["url"], destination)
        try:
            os.makedirs(os.path.dirname(destination), exist_ok=True)
            timeout = aiohttp.ClientTimeout(total=None, sock_connect=30, sock_read=120)
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(task["url"]) as resp:
                    if resp.status != 200:
                        raise ModelDownloadError(f"server returned HTTP {resp.status}")
                    if resp.content_length is not None:
                        task["total"] = resp.content_length
                    with open(part_path, "wb") as f:
                        async for chunk in resp.content.iter_chunked(CHUNK_SIZE):
                            f.write(chunk)
                            task["downloaded"] += len(chunk)
                            if task["total"]:
                                task["progress"] = min(task["downloaded"] / task["total"], 1.0)
            os.replace(part_path, destination)
            task["progress"] = 1.0
            task["status"] = "completed"
            logging.info("Model download finished: %s (%d bytes)", destination, task["downloaded"])
        except Exception as e:
            task["status"] = "failed"
            task["error"] = str(e)
            logging.warning("Model download failed for %s: %s", task["url"], e)
            try:
                os.remove(part_path)
            except OSError:
                pass
        finally:
            self.active_paths.discard(destination)
