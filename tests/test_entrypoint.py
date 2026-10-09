"""tests/test_entrypoint.py -- python -m rlv2 picks the right stage module and hands over argv.

Spec: C:/Users/user/.claude/plans/virtual-sauteeing-crown.md Layer 3 (task 8).
NO GOVERNING SPEC in the repo's .spec tree; basis is the operator-approved restructure plan.
"""
import importlib.util
import sys

import pytest

from rlv2 import __main__ as entry


def _targets():
    for name, target in entry.STAGES.items():
        if isinstance(target, dict):
            for sub, mod in target.items():
                yield f"{name} {sub}", mod
        else:
            yield name, target


@pytest.mark.parametrize("label,module", list(_targets()))
def test_every_stage_names_a_module_that_exists(label, module):
    assert importlib.util.find_spec(module) is not None, f"stage '{label}' -> {module} does not resolve"


@pytest.mark.parametrize("argv,expected", [
    (["cpt", "--epochs", "2", "--out", "x"], ("rlv2.train.cpt.train", ["--epochs", "2", "--out", "x"])),
    (["data", "entigraph"], ("rlv2.data.entigraph", [])),
    (["hpo", "rl", "--seed", "7"], ("rlv2.train.hpo.rl", ["--seed", "7"])),
])
def test_dispatch_runs_the_module_as_main_with_the_remaining_argv(monkeypatch, argv, expected):
    seen = {}

    def fake_run_module(name, run_name=None, alter_sys=False):
        seen.update(name=name, run_name=run_name, argv=list(sys.argv))

    monkeypatch.setattr(entry.runpy, "run_module", fake_run_module)
    assert entry.main(argv) == 0
    module, rest = expected
    assert seen["name"] == module
    assert seen["run_name"] == "__main__"
    assert seen["argv"] == [module] + rest


@pytest.mark.parametrize("argv", [["nonsense"], ["hpo"], ["hpo", "nope"], ["data", "nope"]])
def test_unknown_stage_or_missing_substage_exits_2_without_running_anything(monkeypatch, argv):
    monkeypatch.setattr(entry.runpy, "run_module", lambda *a, **k: pytest.fail("must not run a module"))
    assert entry.main(argv) == 2


def test_no_args_prints_usage_and_exits_0(capsys):
    assert entry.main([]) == 0
    out = capsys.readouterr().out
    assert "usage: python -m rlv2" in out and "cpt" in out and "sft" in out and "rl" in out
