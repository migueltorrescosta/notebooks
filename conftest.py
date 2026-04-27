"""Root conftest — ensure src is on the Python path for all tests."""

from __future__ import annotations

import sys
from pathlib import Path

# Add notebooks/ root so "from src import ..." works from tests/ and pages/
_root = Path(__file__).parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
