"""E2E tests using Playwright to verify Streamlit app navigation.

These tests verify that:
1. The Streamlit app launches successfully
2. The custom sidebar navigation is visible
"""

from __future__ import annotations

import os
import subprocess
import time
from collections.abc import Generator
from pathlib import Path

import pytest
from playwright.sync_api import expect, sync_playwright


# Project root
PROJECT_ROOT = Path(__file__).parent.parent
HOMEPY = PROJECT_ROOT / "Home.py"


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
    max_retries = 30
    base_url = f"http://localhost:{port}"
    for _ in range(max_retries):
        try:
            import urllib.request

            urllib.request.urlopen(base_url, timeout=1)
            break
        except Exception:
            time.sleep(1)
    else:
        process.terminate()
        pytest.fail("Streamlit server failed to start")

    yield process, port

    # Cleanup
    process.terminate()
    process.wait(timeout=5)


def test_streamlit_app_launches(streamlit_server: tuple[subprocess.Popen, int]) -> None:
    """Verify Streamlit process started without immediate errors."""
    process, port = streamlit_server
    time.sleep(5)
    assert process.poll() is None, "Streamlit process should be running"


def test_sidebar_navigation_visible(
    streamlit_server: tuple[subprocess.Popen, int],
) -> None:
    """Verify sidebar navigation element is visible."""
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
