"""
conftest.py - Root pytest configuration.
Sets GROQ_API_KEY env var for all tests.
"""
import os
import pytest

# Set required env vars before any imports
os.environ.setdefault("GROQ_API_KEY", "test-key-not-real")
