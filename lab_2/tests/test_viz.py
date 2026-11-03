"""Тесты модуля визуализации."""

import pandas as pd

from lab.viz import box, corr, decoy, volcano


def test_box(tmp_path) -> None:
    df = pd.DataFrame(
        {
            "Category": ["PTB", "FTB", "PTB"],
            "M1": [1.0, 2.0, 1.5],
            "M2": [3.0, 4.0, 3.5],
        }
    )
    out = box(df, ["M1", "M2"], path=tmp_path / "b.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_corr(tmp_path) -> None:
    df = pd.DataFrame({"M1": [1.0, 2.0, 3.0], "M2": [2.0, 4.0, 6.0]})
    out = corr(df, ["M1", "M2"], path=tmp_path / "c.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_volcano(tmp_path) -> None:
    df = pd.DataFrame({"p_value": [0.01, 0.5, 0.001], "log2_fc": [2.0, 0.1, -2.5]})
    out = volcano(df, path=tmp_path / "v.png")
    assert out.is_file()
    assert out.stat().st_size > 0


def test_decoy(tmp_path) -> None:
    df = pd.DataFrame(
        {"LibraryIntensity": [10.0, 100.0, 50.0, 5.0], "decoy": [0, 0, 1, 1]}
    )
    out = decoy(df, path=tmp_path / "d.png")
    assert out.is_file()
    assert out.stat().st_size > 0
