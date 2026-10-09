"""tests/test_ast_outline.py -- the README code outline is generated from the AST and must not go stale.

Spec: C:/Users/user/.claude/plans/virtual-sauteeing-crown.md Layer 3 (task 9).
NO GOVERNING SPEC in the repo's .spec tree; basis is the operator-approved restructure plan.
"""
import importlib.util
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location("ast_outline", os.path.join(ROOT, "docs", "ast_outline.py"))
ao = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ao)


def _write(tmp_path, body):
    p = tmp_path / "m.py"
    p.write_text(body, encoding="utf-8")
    return str(p)


def test_public_names_are_listed_with_first_docstring_line_and_private_are_counted(tmp_path):
    path = _write(tmp_path, '"""m.py -- the summary.\n\nmore"""\n'
                            'class Budget:\n    """Rolling thing.\n\n    detail"""\n\n'
                            'def cap(x):\n    """Clamp x."""\n\n'
                            'def nodoc(x):\n    return x\n\n'
                            'def _a():\n    pass\n\ndef _b():\n    pass\n')
    summary, n_lines, public, private = ao.outline_module(path)
    assert summary == "the summary."
    assert public == [("class", "Budget", "Rolling thing."), ("def", "cap", "Clamp x."), ("def", "nodoc", "")]
    assert private == 2
    assert n_lines == 19


def test_methods_and_nested_functions_are_not_top_level_entries(tmp_path):
    path = _write(tmp_path, 'class A:\n    def m(self):\n        pass\n\ndef f():\n    def inner():\n        pass\n')
    _, _, public, private = ao.outline_module(path)
    assert [name for _, name, _ in public] == ["A", "f"]
    assert private == 0


def test_first_line_strips_only_a_leading_path_prefix_and_truncates():
    assert ao.first_line("rlv2/x/y.py -- does the thing\nmore") == "does the thing"
    assert ao.first_line("keeps the filter -- and the rest of this sentence") == "keeps the filter -- and the rest of this sentence"
    long = "w " * 80
    out = ao.first_line(long, width=20)
    assert len(out) <= 20 and out.endswith("...")
    assert ao.first_line(None) == ""


def test_the_readme_block_is_current():
    assert ao.main(["--check"]) == 0, "README code outline is stale: run python docs/ast_outline.py"
