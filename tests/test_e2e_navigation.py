"""E2E smoke tests using HTTP to verify Streamlit app launches and serves content.

These tests verify that:
1. The Streamlit app launches successfully as a background process
2. The HTTP server responds with a valid page

No browser dependency is required — uses only urllib for HTTP requests.
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from collections.abc import Generator

# Project root
PROJECT_ROOT = Path(__file__).parent.parent
HOMEPY = PROJECT_ROOT / "Home.py"


def _wait_for_server(
    base_url: str,
    process: subprocess.Popen,
    max_retries: int = 30,
) -> None:
    """Poll the server URL until it responds or max_retries is exhausted."""
    import urllib.error
    import urllib.request

    for _ in range(max_retries):
        try:
            urllib.request.urlopen(base_url, timeout=1)
            return
        except urllib.error.URLError:
            time.sleep(1)
    process.terminate()
    pytest.fail("Streamlit server failed to start")


@pytest.fixture(scope="module")
def streamlit_server() -> Generator[tuple[subprocess.Popen, int], None, None]:
    """Start Streamlit server in the background and yield the process and port."""
    # Find an available port
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        s.listen(1)
        port = s.getsockname()[1]

    # Start Streamlit
    env = os.environ.copy()
    env["STREAMLIT_SERVER_PORT"] = str(port)
    env["STREAMLIT_SERVER_HEADLESS"] = "true"
    env["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"

    process = subprocess.Popen(
        ["python", "-m", "streamlit", "run", str(HOMEPY)],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    # Wait for server to be ready
    base_url = f"http://localhost:{port}"
    _wait_for_server(base_url, process)

    yield process, port

    # Cleanup
    process.terminate()
    process.wait(timeout=5)


def test_given_streamlit_app_launches_then_start_without_immediate_errors(
    streamlit_server: tuple[subprocess.Popen, int],
) -> None:
    """Verify the Streamlit process stays running after launch."""
    process, _port = streamlit_server
    time.sleep(5)
    assert process.poll() is None, "Streamlit process should be running"


def test_given_app_running_then_serves_home_page(
    streamlit_server: tuple[subprocess.Popen, int],
) -> None:
    """Verify the Streamlit server responds with HTTP 200 and contains expected content.

    Uses only urllib — no browser dependency required.
    """
    import urllib.request

    _, port = streamlit_server
    base_url = f"http://localhost:{port}"

    response = urllib.request.urlopen(base_url, timeout=10)
    assert response.status == 200, f"Expected HTTP 200, got {response.status}"

    html = response.read().decode("utf-8")
    # Streamlit renders a client-side app; the initial HTML is a thin shell
    # containing the framework bootstrap and a root mount point.
    assert '<div id="root"></div>' in html, (
        "Expected Streamlit root mount point in HTML"
    )
    assert "streamlit" in html.lower(), (
        "Expected Streamlit framework references in HTML"
    )
