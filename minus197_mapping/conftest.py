"""
conftest.py
-----------
Adds the project root to sys.path so all package imports resolve
correctly when running pytest from any working directory.
"""
import sys
from pathlib import Path

# Project root = directory containing this file
ROOT = Path(__file__).parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
