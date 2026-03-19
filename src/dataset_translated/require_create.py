#!/usr/bin/env python3
"""
Generate requirements.txt by:
1) scanning all .py files for imports (AST)
2) removing stdlib modules
3) mapping imported top-level modules -> installed PyPI distributions
4) writing pinned requirements (dist==version) by default

Run from project root inside a CLEAN venv where the app actually runs.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path
from typing import Iterable, Set, Dict

try:
    from importlib import metadata as importlib_metadata  # py3.8+
except ImportError:  # pragma: no cover
    import importlib_metadata  # type: ignore


EXCLUDE_DIRS = {
    ".git", ".hg", ".svn",
    "__pycache__", ".mypy_cache", ".pytest_cache",
    "venv", ".venv", "env", ".env",
    "build", "dist",
    "node_modules",
}

# Common “import name” != “pip install name” mismatches.
# Add to this when validation reveals a missing/incorrect name.
IMPORT_TO_DIST_OVERRIDES: Dict[str, str] = {
    "cv2": "opencv-python",
    "PIL": "Pillow",
    "skimage": "scikit-image",
    "yaml": "PyYAML",
    "Crypto": "pycryptodome",
    "OpenSSL": "pyOpenSSL",
    "bs4": "beautifulsoup4",
}


def iter_py_files(root: Path) -> Iterable[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        # prune excluded directories
        dirnames[:] = [d for d in dirnames if d not in EXCLUDE_DIRS and not d.startswith(".")]
        for fn in filenames:
            if fn.endswith(".py"):
                yield Path(dirpath) / fn


def extract_imports(py_file: Path) -> Set[str]:
    """Return top-level import module names from a .py file."""
    try:
        src = py_file.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        src = py_file.read_text(encoding="latin-1")

    try:
        tree = ast.parse(src, filename=str(py_file))
    except SyntaxError:
        # Skip files that are not valid for the running Python version
        return set()

    mods: Set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mods.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module and node.level == 0:
                mods.add(node.module.split(".")[0])
    return mods


def stdlib_modules() -> Set[str]:
    # Python 3.10+ provides this list
    names = set(getattr(sys, "stdlib_module_names", set()))
    # Add builtins and some common ones that may not appear depending on version
    names.update({"__future__", "typing", "importlib", "pathlib", "dataclasses"})
    return names


def modules_to_distributions() -> Dict[str, Set[str]]:
    """
    Build mapping from importable top-level package -> installed distribution(s).
    """
    mapping: Dict[str, Set[str]] = {}
    pkg_to_dists = importlib_metadata.packages_distributions()  # type: ignore[attr-defined]
    for pkg, dists in pkg_to_dists.items():
        mapping[pkg] = set(dists)
    return mapping


def main() -> int:
    root = Path.cwd()
    all_imports: Set[str] = set()
    for py in iter_py_files(root):
        all_imports |= extract_imports(py)

    # Remove stdlib and local package names heuristically
    std = stdlib_modules()
    all_imports = {m for m in all_imports if m not in std}

    # Optionally drop your own project package if it’s importable by name
    # (This is imperfect; keep it simple.)
    project_dir_names = {p.name for p in root.iterdir() if p.is_dir()}
    all_imports = {m for m in all_imports if m not in project_dir_names}

    mod2dist = modules_to_distributions()

    dists: Set[str] = set()
    unresolved: Set[str] = set()

    for mod in sorted(all_imports):
        if mod in IMPORT_TO_DIST_OVERRIDES:
            dists.add(IMPORT_TO_DIST_OVERRIDES[mod])
            continue

        candidates = mod2dist.get(mod)
        if candidates:
            # If multiple dists provide the same top-level module, take all.
            dists |= candidates
        else:
            unresolved.add(mod)

    # Build pinned requirements
    requirements = []
    for dist in sorted(dists, key=str.lower):
        try:
            ver = importlib_metadata.version(dist)
            requirements.append(f"{dist}=={ver}")
        except importlib_metadata.PackageNotFoundError:
            # Could be a local/editable or missing metadata
            requirements.append(dist)

    out = root / "requirements.txt"
    out.write_text("\n".join(requirements) + "\n", encoding="utf-8")

    if unresolved:
        warn = root / "requirements.unresolved.txt"
        warn.write_text("\n".join(sorted(unresolved)) + "\n", encoding="utf-8")
        print(f"[WARN] Unresolved imports written to: {warn}")

    print(f"[OK] Wrote {len(requirements)} packages to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())