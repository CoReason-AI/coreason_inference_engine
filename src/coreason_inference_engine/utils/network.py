# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import asyncio
import ipaddress
import socket
from urllib.parse import urlparse


class SSRFValidationError(Exception):
    """Raised when a URL resolves to a restricted IP address (e.g., loopback, private, metadata)."""


async def validate_url_for_ssrf(url: str) -> str:
    """
    Validates a URL to ensure it does not resolve to an internal/private IP address,
    preventing Server-Side Request Forgery (SSRF) attacks.
    Returns the first validated safe IP address string to prevent DNS rebinding (TOCTOU).

    Args:
        url: The full URL to validate.

    Returns:
        The validated safe IP address.

    Raises:
        SSRFValidationError: If the URL resolves to a restricted IP space (loopback, private, etc.).
        ValueError: If the URL is malformed or missing a hostname.
    """
    parsed = urlparse(url)
    hostname = parsed.hostname

    if not hostname:
        raise ValueError(f"Invalid URL: '{url}' lacks a valid hostname.")

    loop = asyncio.get_running_loop()

    try:
        # Resolve the hostname. Using AF_UNSPEC gets both IPv4 and IPv6 if available.
        addr_info = await loop.getaddrinfo(hostname, None)
    except socket.gaierror as e:
        # If the domain doesn't exist, we can't fetch it, so it's technically 'safe' from SSRF,
        # but we let the caller know it failed resolution.
        raise ValueError(f"Could not resolve hostname '{hostname}': {e}") from e

    # Check all resolved IPs
    safe_ip = None
    for result in addr_info:
        # result is a tuple: (family, type, proto, canonname, sockaddr)
        # For IPv4, sockaddr is (address, port)
        # For IPv6, sockaddr is (address, port, flow info, scope id)
        sockaddr = result[4]
        ip_str = sockaddr[0]

        try:
            ip_obj = ipaddress.ip_address(ip_str)
        except ValueError:
            # Should not happen with getaddrinfo results, but handle just in case
            continue

        if getattr(ip_obj, "ipv4_mapped", None):
            ipv4_mapped = getattr(ip_obj, "ipv4_mapped", None)
            if ipv4_mapped:
                if ipv4_mapped.is_loopback:
                    raise SSRFValidationError(f"URL resolves to loopback IP via IPv4-mapped IPv6: {ip_str}")
                if ipv4_mapped.is_unspecified:
                    raise SSRFValidationError(f"URL resolves to unspecified IP via IPv4-mapped IPv6: {ip_str}")
                if ipv4_mapped.is_link_local:
                    raise SSRFValidationError(f"URL resolves to link-local IP via IPv4-mapped IPv6: {ip_str}")
                if ipv4_mapped.is_reserved:
                    raise SSRFValidationError(f"URL resolves to reserved IP via IPv4-mapped IPv6: {ip_str}")
                if ipv4_mapped.is_private:
                    raise SSRFValidationError(f"URL resolves to private IP via IPv4-mapped IPv6: {ip_str}")
                if ipv4_mapped.is_multicast:
                    raise SSRFValidationError(f"URL resolves to multicast IP via IPv4-mapped IPv6: {ip_str}")

        if ip_obj.is_unspecified:
            raise SSRFValidationError(f"URL resolves to unspecified IP: {ip_str}")

        # Spec: mathematically reject any resolution to loopback (127.0.0.0/8), private networks
        # (10.0.0.0/8, 192.168.0.0/16), or cloud metadata endpoints (169.254.169.254)
        if ip_obj.is_loopback:
            raise SSRFValidationError(f"URL resolves to loopback IP: {ip_str}")

        if ip_obj.version == 4:
            if ip_obj.is_link_local:
                raise SSRFValidationError(f"URL resolves to link-local IP: {ip_str} (Cloud Metadata risk)")
            if ip_obj.is_reserved:
                raise SSRFValidationError(f"URL resolves to reserved IP: {ip_str}")
            if ip_obj.is_private:
                raise SSRFValidationError(f"URL resolves to private IP: {ip_str}")
            if ip_obj.is_multicast:
                raise SSRFValidationError(f"URL resolves to multicast IP: {ip_str}")
        elif ip_obj.version == 6:
            if ip_obj.is_link_local:
                raise SSRFValidationError(f"URL resolves to link-local IP: {ip_str} (Cloud Metadata risk)")
            if ip_obj.is_reserved:  # pragma: no cover
                raise SSRFValidationError(f"URL resolves to reserved IP: {ip_str}")
            # Address case sensitivity with lower()
            if ip_obj.is_private and ip_obj.exploded.lower().startswith(("fc", "fd")):
                # 2001:db8:: is documentation, not truly a private network SSRF target, but ipaddress marks it private.
                # However, unique local addresses (fc00::/7) are IPv6 private.
                raise SSRFValidationError(f"URL resolves to private IP: {ip_str}")
            if ip_obj.is_multicast:
                raise SSRFValidationError(f"URL resolves to multicast IP: {ip_str}")

        if not safe_ip:
            safe_ip = ip_str

    # Pragma no cover here because we can't easily mock an empty getaddrinfo response that doesn't trigger gaierror
    if not safe_ip:  # pragma: no cover
        raise ValueError(f"No valid IP address found for '{hostname}'.")
    return safe_ip
