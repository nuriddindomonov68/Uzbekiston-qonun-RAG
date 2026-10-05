"""
conftest.py — ensures the project root is on sys.path for all pytest tests,
so that `from app.xxx import yyy` works without installing the package.
"""
import sys
import os

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
