"""AST and structural tests enforcing ELMAS.md engineering rules."""

import ast
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_ROOT = PROJECT_ROOT / "src"

_FORBIDDEN_DIRS = {"utils", "helpers", "misc", "common"}

_ALLOWED_LAYER_IMPORTS: dict[str, set[str]] = {
    "core": set(),
    "ingestion": {"core"},
    "features": {"core", "ingestion"},
    "evaluation": {"core"},
    "modeling": {"core", "features", "evaluation"},
    "visualization": {"core", "evaluation"},
}

_CONTROL_NODES = (
    ast.If,
    ast.For,
    ast.AsyncFor,
    ast.While,
    ast.With,
    ast.AsyncWith,
    ast.Try,
)


def _iter_source_files() -> list[Path]:
    files = sorted(SRC_ROOT.rglob("*.py"))
    files.append(PROJECT_ROOT / "main.py")
    return files


def _max_control_depth(node: ast.AST, current_depth: int = 0) -> int:
    max_depth = current_depth
    for child in ast.iter_child_nodes(node):
        if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        next_depth = (
            current_depth + 1 if isinstance(child, _CONTROL_NODES) else current_depth
        )
        max_depth = max(max_depth, _max_control_depth(child, next_depth))
    return max_depth


def test_no_forbidden_utility_directories() -> None:
    """Reject generic catch-all directories under src/."""
    found_dirs = {
        path.name.lower() for path in SRC_ROOT.rglob("*") if path.is_dir()
    }
    assert found_dirs.isdisjoint(_FORBIDDEN_DIRS)


def test_file_line_counts_within_elmas_limit() -> None:
    """Every Python file in src/ and main.py must be <= 350 lines."""
    for py_file in _iter_source_files():
        line_count = len(py_file.read_text(encoding="utf-8").splitlines())
        assert line_count <= 350, f"{py_file} has {line_count} lines (> 350)."


def test_function_parameter_count_and_indentation_limits() -> None:
    """Every function in src/ and main.py must have <= 3 params and <= 3 indent depth."""
    for py_file in _iter_source_files():
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            params = [
                arg.arg
                for arg in node.args.posonlyargs + node.args.args
                if arg.arg not in {"self", "cls"}
            ]
            assert len(params) <= 3, (
                f"{py_file.name}:{node.name} has {len(params)} params: {params}"
            )
            depth = _max_control_depth(node)
            assert depth <= 3, (
                f"{py_file.name}:{node.name} has control depth {depth} (> 3)."
            )


def test_exported_functions_have_docstrings() -> None:
    """All public functions in src/ must document contract and behavior."""
    for py_file in SRC_ROOT.rglob("*.py"):
        if py_file.name == "__init__.py":
            continue
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and not node.name.startswith("_"):
                doc = ast.get_docstring(node)
                assert doc is not None and len(doc.strip()) > 10, (
                    f"Public function {py_file.name}:{node.name} lacks docstring."
                )


def test_unidirectional_domain_import_dag() -> None:
    """Verify strict acyclic domain dependencies without reverse-layer imports."""
    for py_file in SRC_ROOT.rglob("*.py"):
        rel_parts = py_file.relative_to(SRC_ROOT).parts
        if len(rel_parts) == 1:
            continue
        layer = rel_parts[0]
        allowed = _ALLOWED_LAYER_IMPORTS[layer] | {layer}
        tree = ast.parse(py_file.read_text(encoding="utf-8"), filename=str(py_file))
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module:
                if node.module.startswith("src."):
                    target_layer = node.module.split(".")[1]
                    assert target_layer in allowed, (
                        f"Forbidden import {node.module} inside layer '{layer}' "
                        f"({py_file.name})."
                    )
