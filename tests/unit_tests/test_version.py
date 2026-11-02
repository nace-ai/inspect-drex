"""The package version, the installed metadata and the CHANGELOG agree."""

from __future__ import annotations

import re
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import pytest

import inspect_drex

ROOT = Path(__file__).resolve().parents[2]


def test_changelog_leads_with_the_package_version():
    headings = re.findall(r"^## (\S+)", (ROOT / "CHANGELOG.md").read_text(), re.MULTILINE)
    assert headings[0] == inspect_drex.__version__


def test_installed_metadata_matches_the_module():
    try:
        installed = version("inspect-drex")
    except PackageNotFoundError:
        pytest.skip("inspect-drex is not installed")
    assert installed == inspect_drex.__version__


def test_the_user_agent_names_the_version():
    expected = f"inspect-drex/{inspect_drex.__version__}"
    assert expected == inspect_drex.CLIENT_NAME
