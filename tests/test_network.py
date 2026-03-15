# Copyright (c) 2026 CoReason, Inc.
#
# This software is proprietary and dual-licensed.
# Licensed under the Prosperity Public License 3.0 (the "License").
# A copy of the license is available at https://prosperitylicense.com/versions/3.0.0
# For details, see the LICENSE file.
# Commercial use beyond a 30-day trial requires a separate license.

import socket
from typing import Any

import pytest

from coreason_inference_engine.utils.network import SSRFValidationError, validate_url_for_ssrf


def mock_getaddrinfo(
    hostname: str,
    port: int,  # noqa: ARG001
    family: int = 0,  # noqa: ARG001
    type: int = 0,  # noqa: A002, ARG001
    proto: int = 0,  # noqa: ARG001
    flags: int = 0,  # noqa: ARG001
) -> list[tuple[int, int, int, str, Any]]:
    """
    Mock socket.getaddrinfo to return predefined IPs for specific hostnames.
    Returns: list of (family, type, proto, canonname, sockaddr)
    """
    # Define mapping of hostnames to IPs
    dns_records = {
        "example.com": ["93.184.216.34"],  # public IP
        "localhost": ["127.0.0.1"],
        "local.test": ["127.0.0.1"],
        "metadata.aws": ["169.254.169.254"],  # AWS instance metadata
        "metadata.gcp": ["169.254.169.254"],  # GCP instance metadata
        "private-a": ["10.0.0.1"],
        "private-b": ["172.16.0.1"],
        "private-c": ["192.168.1.1"],
        "multicast.test": ["224.0.0.1"],
        "ipv6-local": ["::1"],
        "ipv6-link": ["fe80::1"],
        "ipv6-public": ["2001:db8::1"],
        "reserved.test": ["240.0.0.1"],  # Class E reserved
    }

    # If the hostname is already an IP address, mock should return it directly
    try:
        import ipaddress

        ipaddress.ip_address(hostname)
        if ":" in hostname:
            return [(int(socket.AF_INET6), int(socket.SOCK_STREAM), 6, "", (hostname, 80, 0, 0))]
        return [(int(socket.AF_INET), int(socket.SOCK_STREAM), 6, "", (hostname, 80))]
    except ValueError:
        pass

    if hostname in dns_records:
        results: list[tuple[int, int, int, str, Any]] = []
        for ip in dns_records[hostname]:
            if ":" in ip:  # IPv6
                results.append((int(socket.AF_INET6), int(socket.SOCK_STREAM), 6, "", (ip, 80, 0, 0)))
            else:  # IPv4
                results.append((int(socket.AF_INET), int(socket.SOCK_STREAM), 6, "", (ip, 80)))
        return results

    # Raise gaierror if not found
    raise socket.gaierror(socket.EAI_NONAME, f"Name or service not known: {hostname}")


async def mock_async_getaddrinfo(
    hostname: str,
    port: int,
    family: int = 0,
    type: int = 0,  # noqa: A002
    proto: int = 0,
    flags: int = 0,
) -> list[tuple[int, int, int, str, Any]]:
    return mock_getaddrinfo(hostname, port, family, type, proto, flags)


@pytest.fixture
def patch_dns(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket, "getaddrinfo", mock_getaddrinfo)


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_public(patch_dns: None) -> None:  # noqa: ARG001
    # Should not raise any exceptions and return safe ip
    assert await validate_url_for_ssrf("https://example.com/api/v1/data") == "93.184.216.34"
    assert await validate_url_for_ssrf("http://example.com") == "93.184.216.34"
    assert await validate_url_for_ssrf("https://[2001:db8::1]/path") == "2001:db8::1"


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_loopback(patch_dns: None) -> None:  # noqa: ARG001
    with pytest.raises(SSRFValidationError, match="loopback IP"):
        await validate_url_for_ssrf("http://localhost:8080/admin")
    with pytest.raises(SSRFValidationError, match="loopback IP"):
        await validate_url_for_ssrf("http://127.0.0.1/")
    with pytest.raises(SSRFValidationError, match="loopback IP"):
        await validate_url_for_ssrf("http://local.test/status")
    with pytest.raises(SSRFValidationError, match="loopback IP"):
        await validate_url_for_ssrf("http://[::1]/")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_private(patch_dns: None) -> None:  # noqa: ARG001
    with pytest.raises(SSRFValidationError, match="private IP"):
        await validate_url_for_ssrf("http://10.0.0.1/config")
    with pytest.raises(SSRFValidationError, match="private IP"):
        await validate_url_for_ssrf("http://192.168.1.1/")
    with pytest.raises(SSRFValidationError, match="private IP"):
        await validate_url_for_ssrf("http://172.16.0.1/")
    with pytest.raises(SSRFValidationError, match="private IP"):
        await validate_url_for_ssrf("https://private-a/")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_metadata(patch_dns: None) -> None:  # noqa: ARG001
    with pytest.raises(SSRFValidationError, match="link-local IP"):
        await validate_url_for_ssrf("http://169.254.169.254/latest/meta-data/")
    with pytest.raises(SSRFValidationError, match="link-local IP"):
        await validate_url_for_ssrf("http://metadata.aws/")
    with pytest.raises(SSRFValidationError, match="link-local IP"):
        await validate_url_for_ssrf("http://metadata.gcp/")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_multicast(patch_dns: None) -> None:  # noqa: ARG001
    with pytest.raises(SSRFValidationError, match="multicast IP"):
        await validate_url_for_ssrf("http://224.0.0.1/")
    with pytest.raises(SSRFValidationError, match="multicast IP"):
        await validate_url_for_ssrf("http://multicast.test/")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_reserved(patch_dns: None) -> None:  # noqa: ARG001
    with pytest.raises(SSRFValidationError, match="reserved IP"):
        await validate_url_for_ssrf("http://240.0.0.1/")
    with pytest.raises(SSRFValidationError, match="reserved IP"):
        await validate_url_for_ssrf("http://reserved.test/")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_invalid_url() -> None:
    with pytest.raises(ValueError, match="lacks a valid hostname"):
        await validate_url_for_ssrf("not_a_url")
    with pytest.raises(ValueError, match="lacks a valid hostname"):
        await validate_url_for_ssrf("http:///path/only")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_ipaddress_value_error(patch_dns: None, monkeypatch: pytest.MonkeyPatch) -> None:  # noqa: ARG001
    # Mock getaddrinfo to return something that causes ipaddress.ip_address to raise ValueError
    def mock_bad_getaddrinfo(
        hostname: str,  # noqa: ARG001
        port: int,  # noqa: ARG001
        family: int = 0,  # noqa: ARG001
        type: int = 0,  # noqa: A002, ARG001
        proto: int = 0,  # noqa: ARG001
        flags: int = 0,  # noqa: ARG001
    ) -> list[tuple[int, int, int, str, Any]]:
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("not_an_ip", 80))]

    monkeypatch.setattr(socket, "getaddrinfo", mock_bad_getaddrinfo)
    # It should raise ValueError because there's no valid IP address found
    with pytest.raises(ValueError, match="No valid IP address found"):
        await validate_url_for_ssrf("http://example.com")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_ipv6_private(patch_dns: None) -> None:  # noqa: ARG001
    with pytest.raises(SSRFValidationError, match="private IP"):
        await validate_url_for_ssrf("http://[fc00::1]/")
    with pytest.raises(SSRFValidationError, match="private IP"):
        await validate_url_for_ssrf("http://[fd00::1]/")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_ipv6_multicast(patch_dns: None) -> None:  # noqa: ARG001
    with pytest.raises(SSRFValidationError, match="multicast IP"):
        await validate_url_for_ssrf("http://[ff02::1]/")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_ipv6_reserved(patch_dns: None) -> None:  # noqa: ARG001
    # An example of IPv6 reserved space.
    # Let's test using an explicit link-local IPv6 and maybe a reserved.
    # IP address fe80:: is link local.
    with pytest.raises(SSRFValidationError, match="link-local IP"):
        await validate_url_for_ssrf("http://[fe80::1]/")

    # Mock a reserved IPv6 address. ::/128 is unspecified, which is reserved.
    with pytest.raises(SSRFValidationError, match="unspecified IP"):
        await validate_url_for_ssrf("http://[::0]/")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_unresolvable(patch_dns: None) -> None:  # noqa: ARG001
    with pytest.raises(ValueError, match="Could not resolve hostname"):
        await validate_url_for_ssrf("http://nonexistent.domain.test/")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_ipv4_mapped_ipv6(patch_dns: None) -> None:  # noqa: ARG001
    with pytest.raises(SSRFValidationError, match="loopback IP via IPv4-mapped IPv6"):
        await validate_url_for_ssrf("http://[::ffff:127.0.0.1]/")
    with pytest.raises(SSRFValidationError, match="private IP via IPv4-mapped IPv6"):
        await validate_url_for_ssrf("http://[::ffff:10.0.0.1]/")
    with pytest.raises(SSRFValidationError, match="link-local IP via IPv4-mapped IPv6"):
        await validate_url_for_ssrf("http://[::ffff:169.254.169.254]/")
    with pytest.raises(SSRFValidationError, match="unspecified IP via IPv4-mapped IPv6"):
        await validate_url_for_ssrf("http://[::ffff:0.0.0.0]/")
    with pytest.raises(SSRFValidationError, match="reserved IP via IPv4-mapped IPv6"):
        await validate_url_for_ssrf("http://[::ffff:240.0.0.1]/")
    with pytest.raises(SSRFValidationError, match="multicast IP via IPv4-mapped IPv6"):
        await validate_url_for_ssrf("http://[::ffff:224.0.0.1]/")


@pytest.mark.asyncio
async def test_validate_url_for_ssrf_unspecified(patch_dns: None) -> None:  # noqa: ARG001
    with pytest.raises(SSRFValidationError, match="unspecified IP"):
        await validate_url_for_ssrf("http://0.0.0.0/")
