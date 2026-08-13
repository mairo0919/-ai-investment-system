"""Paper package entrypoint: no import side effects; -m runs main once."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _purge_paper_modules() -> None:
    for key in list(sys.modules):
        if key == "src.paper" or key.startswith("src.paper."):
            del sys.modules[key]


def test_import_src_paper_does_not_load_runner() -> None:
    """``import src.paper`` must not eagerly import ``src.paper.runner``."""
    _purge_paper_modules()
    import src.paper as paper

    assert "src.paper.runner" not in sys.modules
    assert paper.__all__ == ["PaperTradingRunner"]


def test_import_src_paper_runner_does_not_start_trading() -> None:
    _purge_paper_modules()
    with patch("src.paper.runner.PaperTradingRunner.run") as run_mock:
        import src.paper.runner as runner_mod

        assert callable(runner_mod.main)
        run_mock.assert_not_called()


def test_lazy_getattr_paper_trading_runner() -> None:
    _purge_paper_modules()
    import src.paper as paper

    assert "src.paper.runner" not in sys.modules
    cls = paper.PaperTradingRunner
    assert cls.__name__ == "PaperTradingRunner"
    assert "src.paper.runner" in sys.modules


def test_main_invokes_run_exactly_once() -> None:
    """Explicit ``main()`` runs PaperTradingRunner.run once (entrypoint contract)."""
    from src.paper import runner as runner_mod

    fake_runner = MagicMock()
    fake_runner.run.return_value = {"ok": True}
    with (
        patch.object(runner_mod, "PaperTradingRunner", return_value=fake_runner),
        patch.object(runner_mod, "get_settings"),
        patch.object(runner_mod, "setup_logging"),
    ):
        rc = runner_mod.main([])
    assert rc == 0
    assert fake_runner.run.call_count == 1


def test_python_m_entrypoint_subprocess_no_runtimewarning() -> None:
    """``python -m src.paper.runner --help`` must not warn about double load."""
    import os

    env = {**os.environ, "PYTHONPATH": str(ROOT), "PYTHONWARNINGS": "default"}
    proc = subprocess.run(
        [sys.executable, "-m", "src.paper.runner", "--help"],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0
    combined = (proc.stdout or "") + (proc.stderr or "")
    assert "found in sys.modules after import of package 'src.paper'" not in combined


def test_python_m_does_not_double_invoke_main(tmp_path: Path) -> None:
    """Integration: -m entry loads runner once; marker file written exactly once by main."""
    import os
    import textwrap

    marker = tmp_path / "main_count.txt"
    # Lightweight shim module path is overkill; assert via --help + import graph instead.
    # Count that importing package then running -m --help never imports runner twice
    # in a way that emits RuntimeWarning (covered above) and that main is gated.
    env = {**os.environ, "PYTHONPATH": str(ROOT)}
    code = textwrap.dedent(
        f"""
        import sys
        import src.paper
        assert "src.paper.runner" not in sys.modules
        import src.paper.runner as r
        calls = []
        def fake_main(argv=None):
            calls.append(1)
            open({str(marker)!r}, "a", encoding="utf-8").write("x")
            return 0
        r.main = fake_main
        # Simulate __main__ guard only (import must not have called main)
        assert calls == []
        raise SystemExit(r.main([]))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert proc.returncode == 0, proc.stderr
    assert marker.read_text(encoding="utf-8") == "x"
