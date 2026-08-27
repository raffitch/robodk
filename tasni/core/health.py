"""Lightweight reachability probes for the dashboard.

These must NOT launch RoboDK or disturb a running capture — they are bare TCP
connect tests with a short timeout, nothing more.
"""
from __future__ import annotations

import ipaddress
import socket

# RoboDK's API server listens here by default (seen in its startup banner).
ROBODK_API_PORT = 20500


def connection_route(host: str) -> str:
    """Human-readable route type for a configured IP/hostname."""
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return "IP/hostname"
    if address in ipaddress.ip_network("100.64.0.0/10"):
        return "Tailscale"
    if address.is_private or address.is_link_local or address.is_loopback:
        return "Direct/LAN"
    return "Public IP"


def tcp_probe(host: str, port: int, timeout: float = 0.6) -> bool:
    """True if a TCP connection to ``host:port`` succeeds within ``timeout``."""
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False
