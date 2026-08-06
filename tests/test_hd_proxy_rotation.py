"""When YouTube rate-limits the IP, retrying from that IP is not a plan.

Measured on job 229233891e: 42 of 42 YouTube sources fell back to 360p on a PO-token/403
rejection, and the pipeline's own two recovery sweeps recovered 0/42 across roughly ninety minutes.
Nothing in the setup was wrong — the POT server answered on 4416, yt-dlp was the current release,
and the identical command succeeded later completely unchanged. YouTube was rate-limiting that
exit address, and the render then proceeds in SD; one 12-minute video has already shipped entirely
at 360p that way.

So the HD path can change exit node. Nodes come from VIDLORE_HD_PROXIES, each call takes the next
one so a blocked node is not hammered, and an empty setting means a direct connection — the
previous behaviour, byte for byte.

Measured against the exact video that failed, through the pipeline's own HD command: without
cookies 4 of 7 nodes returned 720p and 3 hit "Sign in to confirm you're not a bot"; WITH the
cookies this path already sends, all 7 returned 720p. Afterwards both previously-403 videos probed
clean through `probe_max_height` — 720p and 1080p.
"""
from __future__ import annotations

import inspect

import pytest

from vidlore.clipstudio import hd_download as H


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    monkeypatch.delenv("VIDLORE_HD_PROXIES", raising=False)
    H._PROXY_ROTATION[0] = 0
    yield
    H._PROXY_ROTATION[0] = 0


# ------------------------------------------------------------------ unset means unchanged
def test_no_proxies_configured_is_a_direct_connection():
    assert H._proxy_pool() == []
    assert H._proxy_args() == []


def test_blank_and_whitespace_entries_are_ignored(monkeypatch):
    monkeypatch.setenv("VIDLORE_HD_PROXIES", "  ,, \n , ")
    assert H._proxy_pool() == []
    assert H._proxy_args() == []


# ------------------------------------------------------------------ the formats a user will paste
def test_host_port_user_pass_becomes_an_authenticated_url(monkeypatch):
    monkeypatch.setenv("VIDLORE_HD_PROXIES", "1.2.3.4:12323:bob:secret")
    assert H._proxy_pool() == ["http://bob:secret@1.2.3.4:12323"]


def test_host_port_without_credentials_is_accepted(monkeypatch):
    monkeypatch.setenv("VIDLORE_HD_PROXIES", "1.2.3.4:8080")
    assert H._proxy_pool() == ["http://1.2.3.4:8080"]


def test_a_full_url_is_passed_through_untouched(monkeypatch):
    monkeypatch.setenv("VIDLORE_HD_PROXIES", "socks5://user:pw@10.0.0.1:1080")
    assert H._proxy_pool() == ["socks5://user:pw@10.0.0.1:1080"]


@pytest.mark.parametrize("sep", [",", "\n", ",\n"])
def test_entries_may_be_separated_by_commas_or_newlines(monkeypatch, sep):
    monkeypatch.setenv("VIDLORE_HD_PROXIES", f"1.1.1.1:1:a:b{sep}2.2.2.2:2:c:d")
    assert len(H._proxy_pool()) == 2


def test_a_malformed_entry_is_dropped_not_guessed(monkeypatch):
    monkeypatch.setenv("VIDLORE_HD_PROXIES", "not-a-proxy, 1.2.3.4:9:u:p:extra, 5.6.7.8:80:u:p")
    assert H._proxy_pool() == ["http://u:p@5.6.7.8:80"]


# ------------------------------------------------------------------ rotation
def test_each_call_takes_the_next_node(monkeypatch):
    """A blocked node must not be hammered — that is the whole point."""
    monkeypatch.setenv("VIDLORE_HD_PROXIES", "1.1.1.1:1:a:b,2.2.2.2:2:c:d,3.3.3.3:3:e:f")
    got = [H._proxy_args()[1] for _ in range(6)]
    assert got[:3] == ["http://a:b@1.1.1.1:1", "http://c:d@2.2.2.2:2", "http://e:f@3.3.3.3:3"]
    assert got[3:] == got[:3], "and then it wraps"


def test_a_single_node_still_works(monkeypatch):
    monkeypatch.setenv("VIDLORE_HD_PROXIES", "1.1.1.1:1:a:b")
    assert H._proxy_args() == H._proxy_args() == ["--proxy", "http://a:b@1.1.1.1:1"]


# ------------------------------------------------------------------ both HD commands use it
def test_the_download_command_rotates():
    src = inspect.getsource(H.download_hd)
    assert "*_proxy_args()," in src


def test_the_probe_command_rotates_too():
    """A probe that reads SD would mark a 1080p upload as SD before a byte is fetched."""
    src = inspect.getsource(H.probe_max_height)
    assert "*_proxy_args()," in src


def test_the_proxy_is_passed_as_yt_dlp_s_own_flag():
    assert H._proxy_args.__doc__
    src = inspect.getsource(H._proxy_args)
    assert '"--proxy"' in src
