"""Resource-owned HTTP client for bounded dataset downloads."""

from __future__ import annotations

import socket
from typing import Final

import httpx2

LIMITS: Final = httpx2.Limits(
    max_connections=200,
    max_keepalive_connections=40,
    keepalive_expiry=30.0,
)
TIMEOUT: Final = httpx2.Timeout(connect=5.0, read=30.0, write=10.0, pool=10.0)
SOCKET_OPTIONS: Final = [(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)]


def create_client() -> httpx2.Client:
    """Create the sole configured HTTP client used by data acquisition."""
    transport = httpx2.HTTPTransport(
        http2=True,
        retries=3,
        limits=LIMITS,
        socket_options=SOCKET_OPTIONS,
    )
    return httpx2.Client(
        transport=transport,
        timeout=TIMEOUT,
        follow_redirects=True,
        headers={"User-Agent": "bigdb-mvp/0.1 bounded-dataset-ingest"},
    )
