"""Test basic package imports and version information."""

import pytest


def test_package_import():
    """Test that the main package can be imported."""
    import connectome_gnn
    assert connectome_gnn is not None


def test_version_available():
    """Test that version information is accessible."""
    import connectome_gnn
    assert hasattr(connectome_gnn, "__version__")
    assert isinstance(connectome_gnn.__version__, str)
    assert len(connectome_gnn.__version__) > 0


def test_author_info():
    """Test that author information is available."""
    import connectome_gnn
    assert hasattr(connectome_gnn, "__author__")
    assert hasattr(connectome_gnn, "__email__")
    assert isinstance(connectome_gnn.__author__, str)
    assert isinstance(connectome_gnn.__email__, str)