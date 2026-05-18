"""E2E tests using Playwright to verify Streamlit app navigation.

These tests verify that:
1. The Streamlit app launches successfully
2. The custom sidebar navigation is visible
"""

from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from playwright.sync_api import expect, sync_playwright

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
    import urllib.request

    for _ in range(max_retries):
        try:
            urllib.request.urlopen(base_url, timeout=1)
            return
        except Exception:
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
    process, _port = streamlit_server
    time.sleep(5)
    assert process.poll() is None, "Streamlit process should be running"


def test_given_sidebar_navigation_visible_then_be_visible(
    streamlit_server: tuple[subprocess.Popen, int],
) -> None:
    _, port = streamlit_server
    base_url = f"http://localhost:{port}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()

        page.goto(base_url)

        # Wait for the main content to load (streamlit renders client-side)
        page.wait_for_selector('[data-testid="stSidebar"]', timeout=20000)

        # Check that sidebar exists
        sidebar = page.locator('[data-testid="stSidebar"]')
        expect(sidebar).to_be_visible()

        browser.close()
