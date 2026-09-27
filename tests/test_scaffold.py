"""Package smoke tests."""

import callcenter


def test_package_imports():
    assert callcenter.__version__ == "0.1.0"
