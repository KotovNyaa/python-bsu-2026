"""Тесты точки входа пайплайна."""

from lab.main import run


def test_run() -> None:
    res = run()
    assert "stats" in res
    assert "diff" in res
    assert "top_genes" in res
    assert not res["stats"].empty
    assert not res["diff"].empty
