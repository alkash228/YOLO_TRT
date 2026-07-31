"""Container restart + human-readable admin helpers."""
from __future__ import annotations

import http.client
import os
import socket
import threading
import time
from typing import Any


STATUS_RU = {
    "queued": "В очереди",
    "running": "Идёт обработка",
    "done": "Готово",
    "cancelled": "Отменена",
    "error": "Ошибка",
    "ready": "Готов",
    "building_engines": "Сборка TensorRT…",
    "starting": "Запуск…",
    "stopped": "Остановлен",
    "gpu_missing": "Нет GPU / CUDA",
}

PHASE_RU = {
    "queued": "Ожидание",
    "staging": "Подготовка файла",
    "preload": "Декод / preload",
    "inference": "Инференс (YOLO)",
    "pass2": "Pass2 / tracklets",
    "finalize": "Запись результатов",
    "done": "Завершено",
    "error": "Ошибка",
    "cancelled": "Отмена",
}


def status_ru(code: str) -> str:
    return STATUS_RU.get(str(code), str(code))


def phase_ru(code: str) -> str:
    return PHASE_RU.get(str(code).lower(), str(code))


def fmt_sec(sec: float | None) -> str:
    if sec is None:
        return "—"
    s = float(sec)
    if s < 0:
        return "—"
    if s < 60:
        return f"{s:.1f} с"
    m = int(s // 60)
    r = s - m * 60
    if m < 60:
        return f"{m} мин {r:.0f} с"
    h = m // 60
    m = m % 60
    return f"{h} ч {m} мин"


def fmt_bytes(n: int | float | None) -> str:
    if n is None:
        return "—"
    x = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(x) < 1024 or unit == "TB":
            if unit == "B":
                return f"{int(x)} {unit}"
            return f"{x:.2f} {unit}"
        x /= 1024.0
    return f"{x:.2f} TB"


def docker_available() -> bool:
    sock = os.environ.get("DOCKER_SOCK", "/var/run/docker.sock")
    return os.path.exists(sock)


def _unix_http_connection(timeout: float = 60.0) -> http.client.HTTPConnection:
    sock_path = os.environ.get("DOCKER_SOCK", "/var/run/docker.sock")
    if not os.path.exists(sock_path):
        raise RuntimeError(
            "Docker socket не смонтирован. В docker-compose добавь volume: "
            "/var/run/docker.sock:/var/run/docker.sock"
        )

    class UnixHTTPConnection(http.client.HTTPConnection):
        def __init__(self, unix_path: str, timeout: float = 60.0) -> None:
            super().__init__("localhost", timeout=timeout)
            self._unix_path = unix_path

        def connect(self) -> None:
            self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.sock.settimeout(self.timeout)
            self.sock.connect(self._unix_path)

    return UnixHTTPConnection(sock_path, timeout=timeout)


def _docker_request(method: str, path: str, body: bytes | None = None) -> tuple[int, str]:
    conn = _unix_http_connection(timeout=60.0)
    headers = {"Content-Type": "application/json", "Host": "localhost"}
    conn.request(method, path, body=body, headers=headers)
    resp = conn.getresponse()
    data = resp.read().decode("utf-8", errors="replace")
    status = int(resp.status)
    conn.close()
    return status, data


def _docker_request_raw(method: str, path: str) -> tuple[int, bytes]:
    conn = _unix_http_connection(timeout=120.0)
    headers = {"Host": "localhost", "Accept": "application/vnd.docker.raw-stream"}
    conn.request(method, path, headers=headers)
    resp = conn.getresponse()
    data = resp.read()
    status = int(resp.status)
    conn.close()
    return status, data


def _demux_docker_logs(payload: bytes) -> str:
    """Decode Docker multiplexed stdout/stderr stream into plain text."""
    if not payload:
        return ""

    def _plain() -> str:
        return payload.decode("utf-8", errors="replace")

    # Multiplexed frames: [type][0 0 0][size BE uint32][payload…]
    looks_mux = len(payload) >= 8 and payload[0] in (0, 1, 2) and payload[1:4] == b"\x00\x00\x00"
    if not looks_mux:
        return _plain()

    out = bytearray()
    i = 0
    n = len(payload)
    while i + 8 <= n:
        if payload[i] not in (0, 1, 2) or payload[i + 1 : i + 4] != b"\x00\x00\x00":
            return _plain() if not out else bytes(out).decode("utf-8", errors="replace")
        size = int.from_bytes(payload[i + 4 : i + 8], "big")
        i += 8
        if size < 0 or i + size > n:
            return _plain() if not out else bytes(out).decode("utf-8", errors="replace")
        out.extend(payload[i : i + size])
        i += size
    return bytes(out).decode("utf-8", errors="replace")


def fetch_container_logs(
    *,
    tail: int = 2000,
    timestamps: bool = True,
    stdout: bool = True,
    stderr: bool = True,
) -> tuple[str, str]:
    """
    Fetch Docker container logs as plain text.

    Returns (filename_suggestion, text).
    """
    name = container_name()
    tail = max(1, min(int(tail), 50000))
    q = (
        f"/containers/{name}/logs"
        f"?stdout={'1' if stdout else '0'}"
        f"&stderr={'1' if stderr else '0'}"
        f"&timestamps={'1' if timestamps else '0'}"
        f"&tail={tail}"
    )
    code, raw = _docker_request_raw("GET", q)
    if code >= 300:
        raise RuntimeError(
            f"Docker logs failed HTTP {code}: {raw[:400].decode('utf-8', errors='replace')}"
        )
    text = _demux_docker_logs(raw)
    stamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    filename = f"{name}_logs_{stamp}.txt"
    header = (
        f"# container={name}\n"
        f"# fetched_utc={time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())}\n"
        f"# tail={tail} timestamps={timestamps}\n"
        f"# ---\n"
    )
    return filename, header + text


def container_name() -> str:
    return (
        os.environ.get("YOLO_DRT_CONTAINER_NAME", "").strip()
        or os.environ.get("HOSTNAME", "").strip()
        or "yolo-drt-api"
    )


def restart_container(*, mode: str = "docker") -> dict[str, Any]:
    """
    Restart the Docker container.

    mode=docker → Docker Engine API restart (needs docker.sock).
    mode=exit → process exit; compose restart: unless-stopped recreates container.
    """
    name = container_name()
    mode = (mode or "docker").strip().lower()
    if mode == "exit":
        def _die() -> None:
            time.sleep(0.4)
            os._exit(1)

        threading.Thread(target=_die, name="container-exit-restart", daemon=True).start()
        return {
            "ok": True,
            "mode": "exit",
            "container": name,
            "message": (
                "Процесс завершится через ~0.4с; Docker (restart: unless-stopped) "
                "поднимет контейнер заново."
            ),
        }

    # docker
    try:
        code, body = _docker_request("POST", f"/containers/{name}/restart?t=5")
    except Exception as exc:
        # Fallback to exit-restart if socket missing
        if "socket" in str(exc).lower() or "не смонтирован" in str(exc):
            return restart_container(mode="exit") | {
                "fallback_from": "docker",
                "docker_error": str(exc),
            }
        raise
    if code >= 300:
        raise RuntimeError(f"Docker restart failed HTTP {code}: {body[:500]}")
    return {
        "ok": True,
        "mode": "docker",
        "container": name,
        "message": f"Контейнер «{name}» перезапускается через Docker Engine.",
        "docker_status": code,
    }


def disk_info(path: str) -> dict[str, Any]:
    try:
        st = os.statvfs(path)
        total = st.f_frsize * st.f_blocks
        free = st.f_frsize * st.f_bavail
        return {
            "path": path,
            "total_bytes": total,
            "free_bytes": free,
            "total_human": fmt_bytes(total),
            "free_human": fmt_bytes(free),
        }
    except (AttributeError, OSError):
        # Windows host test without Docker
        try:
            import shutil

            usage = shutil.disk_usage(path)
            return {
                "path": path,
                "total_bytes": usage.total,
                "free_bytes": usage.free,
                "total_human": fmt_bytes(usage.total),
                "free_human": fmt_bytes(usage.free),
            }
        except OSError as exc:
            return {"path": path, "error": str(exc)}
