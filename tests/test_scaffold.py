"""Scaffold sanity tests — proves the environment and package wiring work.

Behavioural tests live alongside the code they cover (test_chunkers.py,
test_fusion.py, test_dedup.py, test_generate.py, test_evals.py, test_api.py).
"""

from pathlib import Path


def test_package_importable():
    import rag

    assert rag.__version__


def test_settings_defaults_load_without_env_file():
    from rag.config import Settings

    s = Settings(_env_file=None)
    assert s.rrf_k == 60
    assert 0 < s.dense_weight <= 1
    assert s.chunk_overlap < s.chunk_size
    assert s.chroma_dir == Path(".chroma")
