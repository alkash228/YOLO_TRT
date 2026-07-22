"""LAN / listen URL helpers for API and WEB_app startup banners."""
from __future__ import annotations

import socket
from typing import Iterable


def lan_ipv4_addresses() -> list[str]:
    """Non-loopback IPv4 addresses of this machine."""
    ips: list[str] = []
    seen: set[str] = set()

    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as sock:
            sock.connect(("8.8.8.8", 80))
            ip = sock.getsockname()[0]
            if ip and not ip.startswith("127.") and ip not in seen:
                seen.add(ip)
                ips.append(ip)
    except OSError:
        pass

    try:
        hostname = socket.gethostname()
        for info in socket.getaddrinfo(hostname, None, socket.AF_INET):
            ip = info[4][0]
            if ip.startswith("127.") or ip in seen:
                continue
            seen.add(ip)
            ips.append(ip)
    except OSError:
        pass

    return ips


def listen_urls(host: str, port: int, path: str = "") -> list[str]:
    """Build HTTP URLs clients can open (localhost + LAN when bound to all interfaces)."""
    if path and not path.startswith("/"):
        path = f"/{path}"
    urls: list[str] = []

    if host in ("0.0.0.0", "::", ""):
        urls.append(f"http://127.0.0.1:{port}{path}")
        for ip in lan_ipv4_addresses():
            urls.append(f"http://{ip}:{port}{path}")
    elif host in ("127.0.0.1", "localhost", "::1"):
        urls.append(f"http://127.0.0.1:{port}{path}")
        for ip in lan_ipv4_addresses():
            urls.append(f"http://{ip}:{port}{path}")
    else:
        urls.append(f"http://{host}:{port}{path}")

    return urls


def print_listen_banner(
    *,
    service: str,
    host: str,
    port: int,
    path: str = "",
    extra_lines: Iterable[str] | None = None,
) -> None:
    urls = listen_urls(host, port, path)
    line = "=" * 62
    print(f"\n{line}")
    print(f"  {service}")
    print(f"  Bind: {host}:{port}")
    print(line)
    print("  Открыть в браузере или с другого ПК в локальной сети:")
    for url in urls:
        print(f"    -> {url}")
    if extra_lines:
        for text in extra_lines:
            print(f"  {text}")
    print(f"{line}\n")
