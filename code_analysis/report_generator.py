"""
Enhanced CRCT report_generator
==============================

Augments the original static-only `report_generator.py` with runtime metadata
produced by `runtime_inspector.py` (and, when available, the merged
`project_symbol_map.json` from `symbol_map_merger.py`).

Design summary
--------------
The original generator detects incomplete / placeholder code through:
  * regex patterns (TODO, FIXME, "for now", "placeholder", ...)
  * tree-sitter AST checks for empty / stub functions, classes, NotImplementedError
  * pyright's "is not accessed" diagnostics for unused items

These signals are *local* — they say "this line/this body looks incomplete"
but cannot say:
  * who calls this symbol
  * what it claims to return (annotated return type vs. empty body)
  * whether it is exported (`__all__`) or imported elsewhere
  * what its inheritance / decorator stack looks like
  * what attribute accesses or globals it depends on

Runtime Inspector already collects all of that, per file, in
`runtime_symbols.json`.  This generator builds an in-memory index of that
data keyed by (file, line_range) so every static issue can be attributed
to its owning symbol and enriched with:

  - owning symbol, qualname, signature, type annotations
  - decorators, inheritance / MRO, async / property flags
  - closure dependencies and globals references
  - cross-file links: callers, importers, attribute-access references
  - export status (in `__all__`?)
  - severity score derived from the combined signals

It also introduces *runtime-only* checks that pure static analysis misses:

  1. Annotated stub                — body is empty / pass / `...` / "return None"
                                     but the function declares a non-trivial
                                     return type.
  2. Exported placeholder          — symbol is in `__all__` (or imported by
                                     another file) yet matches a stub /
                                     placeholder pattern.
  3. Confirmed NotImplementedError — tree-sitter saw `raise NotImplementedError`
                                     and runtime confirms the symbol is
                                     importable (i.e. it really is reachable
                                     code, not a dead branch).
  4. Async without await           — `is_async` / `async` decorator chain but
                                     `attribute_accesses` shows no `await`-ish
                                     usage (heuristic).
  5. Orphan symbol                 — defined and exported but no other file
                                     references the name in `calls`,
                                     `attribute_accesses`, `scope_references`,
                                     or `imports`.  Complements pyright.
  6. Stub method on non-abstract   — method body trivial, class MRO contains
     class                           no `ABC` / `abstractmethod` ancestor.

The merge respects runtime as authoritative for type / inheritance / decorator
fields and uses AST data to widen the call-graph view (which is what makes
"linked areas" possible).

Output
------
`code_analysis/issues_report.md` (same path as before) plus a sibling
machine-readable `code_analysis/issues_report.json` containing the full
context block for every issue, suitable for downstream LLM consumption.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union, cast

# ---------------------------------------------------------------------------
# Project-root bootstrap
# ---------------------------------------------------------------------------
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from cline_utils.dependency_system.utils import path_utils  # noqa: E402
from cline_utils.dependency_system.utils.config_manager import (
    ConfigManager,
)  # noqa: E402

# Optional: use the merger if available so we get runtime + AST in one shot.
# (Note: Current implementation of load_runtime_data checks for the merged file directly)

# Tree-sitter configuration
_has_tree_sitter = False
_Language: Any = None
_Parser: Any = None
_ts_js: Any = None
_ts_py: Any = None
_ts_ts: Any = None

try:
    import tree_sitter_javascript as _ts_js
    import tree_sitter_python as _ts_py
    import tree_sitter_typescript as _ts_ts
    from tree_sitter import Language as _Language, Parser as _Parser

    _has_tree_sitter = True
except ImportError:
    print("Warning: tree-sitter not available. Falling back to regex-based analysis.")

config = ConfigManager()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
EXTENSIONS = {".py", ".js", ".ts", ".jsx", ".tsx", ".md", ".txt"}

PATTERNS = {
    "TODO": re.compile(r"TODO", re.IGNORECASE),
    "FIXME": re.compile(r"FIXME", re.IGNORECASE),
    "pass": re.compile(r"^\s*pass\s*$", re.MULTILINE),
    "NotImplementedError": re.compile(r"NotImplementedError"),
    "in a real": re.compile(r"in a real", re.IGNORECASE),
    "for now": re.compile(r"for now", re.IGNORECASE),
    "simplified": re.compile(r"simplified", re.IGNORECASE),
    "placeholder": re.compile(r"placeholder", re.IGNORECASE),
}

EXCLUSION_PATTERNS = {
    "placeholder": [
        re.compile(r"_placeholder", re.IGNORECASE),
        re.compile(r"placeholder_", re.IGNORECASE),
        re.compile(r"sql\.Placeholder", re.IGNORECASE),
        re.compile(r"placeholders\s*=", re.IGNORECASE),
        re.compile(r"placeholder\s*=", re.IGNORECASE),
        re.compile(r"placeholder\s*:", re.IGNORECASE),
        re.compile(r"Placeholder\(\)", re.IGNORECASE),
        re.compile(r"\.placeholder\.", re.IGNORECASE),
        re.compile(r'"placeholder"', re.IGNORECASE),
    ],
}

OUTPUT_FILE = "code_analysis/issues_report.md"
OUTPUT_JSON_FILE = "code_analysis/issues_report.json"
PYRIGHT_OUTPUT = "pyright_output.json"

# Standard locations where runtime data may live.
RUNTIME_SYMBOLS_PATH = os.path.join(
    "cline_utils", "dependency_system", "core", "runtime_symbols.json"
)
PROJECT_SYMBOL_MAP_PATH = os.path.join(
    "cline_utils", "dependency_system", "core", "project_symbol_map.json"
)

# Return-type annotations that are "trivial" — declaring them does NOT make
# an empty body a real promise.
_TRIVIAL_RETURNS = {
    "",
    "None",
    "<class 'NoneType'>",
    "<class 'inspect._empty'>",
    "typing.Any",
    "Any",
}

# Body texts that count as "no real implementation".
_TRIVIAL_BODY_RE = re.compile(r"^\s*(pass|\.\.\.|return\s+None|return)\s*$")


# ===========================================================================
# Tree-sitter helpers
# ===========================================================================
def get_parser(lang_name: str) -> Any:
    """Initialize and return a tree-sitter parser for the given language."""
    if not _has_tree_sitter or _Parser is None:
        return None
    try:
        parser = _Parser()
        if lang_name == "python":
            parser.language = _Language(_ts_py.language())
        elif lang_name == "javascript":
            parser.language = _Language(_ts_js.language())
        elif lang_name == "typescript":
            parser.language = _Language(_ts_ts.language_typescript())
        elif lang_name == "tsx":
            parser.language = _Language(_ts_ts.language_tsx())
        else:
            return None
        return parser
    except Exception as e:
        print(f"Error initializing parser for {lang_name}: {e}")
        return None


def analyze_node(
    node: Any,
    issues: List[Dict[str, Any]],
    filepath: Union[str, Path],
    source_code: bytes,
) -> None:
    """Recursively walk tree-sitter nodes; emit raw issue dicts.

    Each emitted issue carries `line` and (for definition-bearing nodes)
    `end_line` so the runtime indexer can attribute the issue to its
    enclosing symbol with high precision.
    """
    if node.type in ("function_definition", "async_function_definition"):
        body_node = node.child_by_field_name("body")
        if body_node:
            has_raise_not_implemented = False
            non_trivial_children: List[Any] = []
            for child in body_node.children:
                if child.type == "comment":
                    continue
                if child.type == "pass_statement":
                    continue
                if child.type == "expression_statement":
                    if child.child_count == 1 and child.children[0].type == "string":
                        continue
                if child.type == "raise_statement":
                    if "NotImplementedError" in child.text.decode("utf8"):
                        has_raise_not_implemented = True
                        continue
                non_trivial_children.append(child)

            if not non_trivial_children:
                kind = (
                    "NotImplementedError"
                    if has_raise_not_implemented
                    else "Empty/Stub Function"
                )
                issues.append(
                    {
                        "type": (
                            "Incomplete Implementation"
                            if has_raise_not_implemented
                            else "Improper Implementation"
                        ),
                        "subtype": kind,
                        "file": str(filepath),
                        "line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "content": node.text.decode("utf8").split("\n")[0] + "...",
                    }
                )

    elif node.type == "class_definition":
        body_node = node.child_by_field_name("body")
        if body_node:
            non_trivial_children: List[Any] = []
            for child in body_node.children:
                if child.type == "comment":
                    continue
                if child.type == "pass_statement":
                    continue
                if child.type == "expression_statement":
                    if child.child_count == 1 and child.children[0].type == "string":
                        continue
                non_trivial_children.append(child)
            if not non_trivial_children:
                issues.append(
                    {
                        "type": "Improper Implementation",
                        "subtype": "Empty/Stub Class",
                        "file": str(filepath),
                        "line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "content": node.text.decode("utf8").split("\n")[0] + "...",
                    }
                )

    elif node.type in (
        "function_declaration",
        "method_definition",
        "arrow_function",
        "class_declaration",
    ):
        body_node = node.child_by_field_name("body")
        if body_node and body_node.type == "statement_block":
            non_comment_children = [
                c for c in body_node.children if c.type not in ("comment", "{", "}")
            ]
            if not non_comment_children:
                issues.append(
                    {
                        "type": "Improper Implementation",
                        "subtype": "Empty/Stub Function/Class",
                        "file": str(filepath),
                        "line": node.start_point[0] + 1,
                        "end_line": node.end_point[0] + 1,
                        "content": node.text.decode("utf8").split("\n")[0] + "...",
                    }
                )

    for child in node.children:
        analyze_node(child, issues, filepath, source_code)


def scan_file(filepath: str) -> List[Dict[str, Any]]:
    """Per-file static scan (regex + tree-sitter). Same shape as before
    but each issue may now carry an `end_line` attribute."""
    issues: List[Dict[str, Any]] = []
    try:
        with open(filepath, "rb") as f:
            content = f.read()

        try:
            text_content = content.decode("utf-8", errors="ignore")
            lines = text_content.splitlines()
            for i, line in enumerate(lines):
                for label, pattern in PATTERNS.items():
                    ext = Path(filepath).suffix
                    is_parsed = _has_tree_sitter and ext in (
                        ".py",
                        ".js",
                        ".ts",
                        ".jsx",
                        ".tsx",
                    )
                    if is_parsed and label in ("pass", "NotImplementedError"):
                        continue
                    if pattern.search(line):
                        excluded = False
                        if label in EXCLUSION_PATTERNS:
                            for excl_pattern in EXCLUSION_PATTERNS[label]:
                                if excl_pattern.search(line):
                                    excluded = True
                                    break
                        if excluded:
                            continue
                        issues.append(
                            {
                                "type": "Incomplete/Improper",
                                "subtype": label,
                                "file": str(filepath),
                                "line": i + 1,
                                "content": line.strip(),
                            }
                        )

                if not _has_tree_sitter and "def " in line and "pass" in line:
                    issues.append(
                        {
                            "type": "Improper Implementation",
                            "subtype": "One-line stub",
                            "file": str(filepath),
                            "line": i + 1,
                            "content": line.strip(),
                        }
                    )
        except Exception as e:
            print(f"Error doing regex scan on {filepath}: {e}")

        if _has_tree_sitter:
            ext = Path(filepath).suffix
            lang = None
            if ext == ".py":
                lang = "python"
            elif ext == ".js":
                lang = "javascript"
            elif ext == ".ts":
                lang = "typescript"
            elif ext in (".jsx", ".tsx"):
                lang = "tsx"

            if lang:
                parser = get_parser(lang)
                if parser:
                    tree = parser.parse(content)
                    analyze_node(tree.root_node, issues, filepath, content)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
    return issues


def get_unused_items() -> List[Dict[str, Any]]:
    unused: List[Dict[str, Any]] = []
    if os.path.exists(PYRIGHT_OUTPUT):
        try:
            with open(PYRIGHT_OUTPUT, "r", encoding="utf-8") as f:
                data = json.load(f)
                if "generalDiagnostics" in data:
                    for diag in data["generalDiagnostics"]:
                        if "is not accessed" in diag.get("message", ""):
                            unused.append(
                                {
                                    "type": "Unused Item",
                                    "subtype": "Pyright Diagnostic",
                                    "file": diag.get("file", "unknown"),
                                    "line": diag.get("range", {})
                                    .get("start", {})
                                    .get("line", 0)
                                    + 1,
                                    "content": diag.get("message", ""),
                                }
                            )
        except Exception as e:
            print(f"Error parsing pyright output: {e}")
    else:
        print(f"Warning: {PYRIGHT_OUTPUT} not found. Skipping unused item analysis.")
    return unused


# ===========================================================================
# Runtime-data integration
# ===========================================================================
class RuntimeIndex:
    """In-memory index over runtime_symbols.json (or merged symbol map).

    Provides:
      * symbol_at(file, line)            -> dict | None
      * callers_of(qualname)             -> list[dict]   (cross-file references)
      * exports_of(file)                 -> list[str]
      * is_exported(file, name)          -> bool
      * is_in_abstract_mro(symbol_dict)  -> bool
    """

    def __init__(self, data: Dict[str, Dict[str, Any]]):
        super().__init__()
        self.raw = data or {}
        # (norm_file) -> list[(start, end, symbol_dict, kind)]
        self._by_file: Dict[str, List[Tuple[int, int, Dict[str, Any], str]]] = (
            defaultdict(list)
        )
        # qualname -> list of (file, symbol_dict, kind)
        self._by_qualname: Dict[str, List[Tuple[str, Dict[str, Any], str]]] = (
            defaultdict(list)
        )
        # name -> list of files that *reference* the name (callers / accessors)
        self._refs: Dict[str, Set[str]] = defaultdict(set)
        # file -> list of __all__ entries
        self._exports: Dict[str, List[str]] = {}

        self._build()

    # -- helpers -----------------------------------------------------------
    @staticmethod
    def norm(p: str) -> str:
        try:
            return os.path.normpath(os.path.abspath(p)).replace("\\", "/")
        except Exception:
            return p

    @staticmethod
    def _line_range(sym: Dict[str, Any]) -> Tuple[int, int]:
        ctx: Dict[str, Any] = sym.get("source_context") or {}
        rng = ctx.get("line_range")
        if isinstance(rng, (list, tuple)) and len(cast(List[Any], rng)) == 2:
            try:
                # Use cast to help Pyright with Unknown types in list access
                return int(cast(Any, rng[0])), int(cast(Any, rng[1]))
            except Exception:
                return (0, 0)
        # fallback: AST 'line' field added by symbol_map_merger
        if "line" in sym:
            try:
                start = int(sym["line"])
                return (start, start + 1)
            except Exception:
                return (0, 0)
        return (0, 0)

    def _record_symbol(
        self,
        file_path: str,
        sym: Dict[str, Any],
        kind: str,
        qualname: Optional[str] = None,
    ):
        start, end = self._line_range(sym)
        if start or end:
            self._by_file[file_path].append((start, end, sym, kind))
        qn = qualname or sym.get("name")
        if qn:
            self._by_qualname[qn].append((file_path, sym, kind))

    def _build(self):
        for raw_path, finfo in self.raw.items():
            file_path = self.norm(raw_path)

            # Exports
            exports_val = finfo.get("exports")
            if isinstance(exports_val, dict):
                self._exports[file_path] = list(
                    cast(Dict[Any, Any], exports_val).keys()
                )
            elif isinstance(exports_val, list):
                self._exports[file_path] = list(cast(List[Any], exports_val))
            else:
                self._exports[file_path] = []

            # Functions
            fns = cast(List[Any], finfo.get("functions") or [])
            for fn_any in fns:
                if isinstance(fn_any, dict):
                    fn = cast(Dict[str, Any], fn_any)
                    self._record_symbol(file_path, fn, "function")
                    # collect cross-file references
                    attrs = cast(List[Any], fn.get("attribute_accesses") or [])
                    for ref in attrs:
                        if isinstance(ref, str):
                            self._refs[ref].add(file_path)
                    scope = cast(Dict[str, Any], fn.get("scope_references") or {})
                    g_list = cast(List[Any], scope.get("globals") or [])
                    for g in g_list:
                        if isinstance(g, str):
                            self._refs[g].add(file_path)

            # Classes (and their methods)
            classes_val = cast(List[Any], finfo.get("classes") or [])
            for cls_any in classes_val:
                if isinstance(cls_any, dict):
                    cls = cast(Dict[str, Any], cls_any)
                    cls_qual: Optional[str] = cast(Optional[str], cls.get("name"))
                    self._record_symbol(file_path, cls, "class", qualname=cls_qual)
                    meths = cast(List[Any], cls.get("methods") or [])
                    for meth_any in meths:
                        if isinstance(meth_any, dict):
                            meth = cast(Dict[str, Any], meth_any)
                            meth_name: str = cast(str, meth.get("name") or "?")
                            qual: str = (
                                f"{cls_qual}.{meth_name}" if cls_qual else meth_name
                            )
                            self._record_symbol(
                                file_path, meth, "method", qualname=qual
                            )
                            m_attrs = cast(
                                List[Any], meth.get("attribute_accesses") or []
                            )
                            for ref in m_attrs:
                                if isinstance(ref, str):
                                    self._refs[ref].add(file_path)
                            it_scope = cast(
                                Dict[str, Any], meth.get("scope_references") or {}
                            )
                            m_gs = cast(List[Any], it_scope.get("globals") or [])
                            for g in m_gs:
                                if isinstance(g, str):
                                    self._refs[g].add(file_path)

            # Calls (only present when symbol_map_merger.py has been run)
            calls_list = cast(List[Any], finfo.get("calls") or [])
            for call in calls_list:
                n: Optional[str] = None
                if isinstance(call, dict):
                    n = cast(Optional[str], cast(Dict[Any, Any], call).get("name"))
                elif isinstance(call, str):
                    n = call
                if n:
                    self._refs[n].add(file_path)

            # Imports also count as references for "linked areas"
            imports_list = cast(List[Any], finfo.get("imports") or [])
            for imp in imports_list:
                inm: Optional[str] = None
                if isinstance(imp, dict):
                    imp_dict = cast(Dict[Any, Any], imp)
                    inm = cast(
                        Optional[str], imp_dict.get("name") or imp_dict.get("module")
                    )
                elif isinstance(imp, str):
                    inm = imp
                if inm:
                    self._refs[inm].add(file_path)

    # -- query API --------------------------------------------------------
    def symbol_at(self, file_path: str, line: int) -> Optional[Dict[str, Any]]:
        """Return the smallest symbol whose line range encloses *line*."""
        nf = self.norm(file_path)
        candidates = self._by_file.get(nf, [])
        best: Optional[Tuple[int, int, Dict[str, Any], str]] = None
        for start, end, sym, kind in candidates:
            if start <= line <= end:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (start, end, sym, kind)
        if best is None:
            return None
        start, end, sym, kind = best
        out = dict(sym)
        out["_kind"] = kind
        out["_line_range"] = (start, end)
        return out

    def callers_of(self, name: str, exclude_file: Optional[str] = None) -> List[str]:
        files = self._refs.get(name, set())
        if exclude_file:
            ex = self.norm(exclude_file)
            files = {f for f in files if f != ex}
        return sorted(files)

    def enclosing_class(self, file_path: str, line: int) -> Optional[Dict[str, Any]]:
        nf = self.norm(file_path)
        candidates = self._by_file.get(nf, [])
        best: Optional[Tuple[int, int, Dict[str, Any], str]] = None
        for start, end, sym, kind in candidates:
            if kind != "class":
                continue
            if start <= line <= end:
                if best is None or (end - start) < (best[1] - best[0]):
                    best = (start, end, sym, kind)
        if best is None:
            return None
        start, end, sym, kind = best
        out = dict(sym)
        out["_kind"] = kind
        out["_line_range"] = (start, end)
        return out

    def is_exported(self, file_path: str, name: str) -> bool:
        return name in (self._exports.get(self.norm(file_path)) or [])

    @staticmethod
    def is_in_abstract_mro(sym: Dict[str, Any]) -> bool:
        inh = cast(Dict[str, Any], sym.get("inheritance") or {})
        bases = cast(List[str], inh.get("bases") or [])
        mro = cast(List[str], inh.get("mro") or [])
        joined = " ".join(bases) + " " + " ".join(mro)
        return "ABC" in joined or "abc." in joined.lower()


def load_runtime_data(project_root_path: str) -> Dict[str, Dict[str, Any]]:
    """Locate the richest runtime data available.

    Preference order:
      1. project_symbol_map.json (runtime + AST merged) — best signal
      2. runtime_symbols.json    (runtime only)
      3. on-the-fly merge via symbol_map_merger if both halves exist
      4. {} (graceful degradation: enhanced report falls back to static-only)
    """
    merged_path = os.path.join(project_root_path, PROJECT_SYMBOL_MAP_PATH)
    runtime_path = os.path.join(project_root_path, RUNTIME_SYMBOLS_PATH)

    if os.path.exists(merged_path):
        try:
            with open(merged_path, "r", encoding="utf-8") as f:
                print(f"[runtime] Using merged symbol map: {merged_path}")
                return json.load(f)
        except Exception as e:
            print(f"[runtime] Failed to load merged map ({e}); falling back.")

    if os.path.exists(runtime_path):
        try:
            with open(runtime_path, "r", encoding="utf-8") as f:
                print(f"[runtime] Using runtime symbols: {runtime_path}")
                return json.load(f)
        except Exception as e:
            print(f"[runtime] Failed to load runtime symbols ({e}).")

    print("[runtime] No runtime data found; running in static-only mode.")
    return {}


# ===========================================================================
# Runtime-derived enhancement passes
# ===========================================================================
def _has_trivial_body(sym: Dict[str, Any]) -> bool:
    """Heuristic: function's source body is empty / pass / `...` / `return None`."""
    ctx: Dict[str, Any] = sym.get("source_context") or {}
    lines: List[str] = ctx.get("source_lines") or []
    if not lines:
        return False
    # Drop the def-header and any docstring lines.
    body: List[str] = []
    in_doc = False
    doc_quote: Optional[str] = None
    for ln in lines[1:]:  # skip the 'def ...:' header
        s = ln.strip()
        if not s:
            continue
        if in_doc:
            if doc_quote and doc_quote in s:
                in_doc = False
            continue
        if s.startswith(('"""', "'''")):
            q = s[:3]
            # single-line docstring like """foo"""
            if s.count(q) >= 2 and len(s) > 3:
                continue
            doc_quote = q
            in_doc = True
            continue
        if s.startswith("#"):
            continue
        body.append(s)
    if not body:
        return True
    return all(_TRIVIAL_BODY_RE.match(b) for b in body)


def _annotated_non_trivial_return(sym: Dict[str, Any]) -> Optional[str]:
    ann: Dict[str, Any] = sym.get("type_annotations") or {}
    rt = ann.get("return_type") or ann.get("parameters", {}).get("return") or ""
    rt_s = str(rt).strip()
    if rt_s and rt_s not in _TRIVIAL_RETURNS and rt_s != "None":
        return rt_s
    return None


def _inherits_from(sym: Dict[str, Any], *needles: str) -> bool:
    inheritance = cast(Dict[str, Any], sym.get("inheritance") or {})
    bases = cast(List[str], inheritance.get("bases") or [])
    mro = cast(List[str], inheritance.get("mro") or [])
    haystack = " ".join(str(part) for part in [*bases, *mro]).lower()
    return any(needle.lower() in haystack for needle in needles)


def _source_mentions(sym: Dict[str, Any], *needles: str) -> bool:
    source_context = cast(Dict[str, Any], sym.get("source_context") or {})
    source_lines = cast(List[str], source_context.get("source_lines") or [])
    haystack = "\n".join(source_lines).lower()
    return any(needle.lower() in haystack for needle in needles)


def _is_protocol_class(sym: Dict[str, Any]) -> bool:
    return _inherits_from(sym, "Protocol", "typing.Protocol") or _source_mentions(
        sym, "(Protocol", ", Protocol", "typing.Protocol"
    )


def _is_abstract_class(sym: Dict[str, Any]) -> bool:
    return RuntimeIndex.is_in_abstract_mro(sym) or _source_mentions(
        sym, "(ABC", ", ABC", "abc.ABC"
    )


def _is_exception_class(sym: Dict[str, Any]) -> bool:
    name = str(sym.get("name") or "")
    inheritance = cast(Dict[str, Any], sym.get("inheritance") or {})
    bases = cast(List[str], inheritance.get("bases") or [])
    mro = cast(List[str], inheritance.get("mro") or [])
    lineage = [name, *[str(item).split(".")[-1] for item in [*bases, *mro]]]
    return any(
        item.endswith(("Error", "Exception")) or item in {"Exception", "BaseException"}
        for item in lineage
    )


def _is_marker_exception_class(sym: Dict[str, Any]) -> bool:
    methods = cast(List[Dict[str, Any]], sym.get("methods") or [])
    return _is_exception_class(sym) and not methods


def _is_abstract_method(sym: Dict[str, Any]) -> bool:
    decorators = cast(List[str], sym.get("decorators") or [])
    return any("abstract" in str(decorator).lower() for decorator in decorators) or (
        _source_mentions(sym, "@abstractmethod", "@abc.abstractmethod")
    )


def _is_data_container_class(sym: Dict[str, Any]) -> bool:
    return _inherits_from(
        sym,
        "BaseModel",
        "Enum",
        "TypedDict",
    ) or _source_mentions(
        sym,
        "@dataclass",
        "(BaseModel",
        "(Enum",
        "(TypedDict",
        "str, Enum",
        "int, Enum",
    )


def _should_suppress_issue(issue: Dict[str, Any], ctx: Dict[str, Any]) -> bool:
    owning = cast(Dict[str, Any], ctx.get("owning_symbol") or {})
    inheritance = cast(Dict[str, Any], ctx.get("inheritance") or {})
    qualname = str(owning.get("qualname") or "")
    subtype = str(issue.get("subtype") or "")
    kind = str(owning.get("kind") or "")

    sym_for_checks: Dict[str, Any] = {"inheritance": inheritance}
    if "." not in qualname and kind == "class":
        sym_for_checks["methods"] = []

    if ctx.get("abstract_method") or ctx.get("abstract_class"):
        if subtype in {
            "NotImplementedError",
            "Empty/Stub Function",
            "Annotated Stub",
            "Concrete Stub Method",
            "Async Stub",
        }:
            return True

    if subtype == "Empty/Stub Class" and _is_marker_exception_class(sym_for_checks):
        return True

    if subtype == "Bare Class" and (
        ctx.get("data_container_class") or _is_data_container_class(sym_for_checks)
    ):
        return True

    if (
        subtype
        in {
            "Annotated Stub",
            "Concrete Stub Method",
            "Async Stub",
            "Empty/Stub Function",
        }
        and "." in qualname
        and _is_protocol_class(sym_for_checks)
    ):
        return True

    return False


def runtime_only_findings(idx: RuntimeIndex) -> List[Dict[str, Any]]:
    """Emit issues that the original static pipeline cannot detect."""
    findings: List[Dict[str, Any]] = []

    for file_path, finfo in idx.raw.items():
        nf = RuntimeIndex.norm(file_path)
        # ---- functions ----
        fns = cast(List[Dict[str, Any]], finfo.get("functions") or [])
        for fn in fns:
            _emit_runtime_issues_for_symbol(
                findings,
                idx,
                nf,
                fn,
                kind="function",
                qualname=cast(str, fn.get("name", "?")),
            )
        # ---- classes & methods ----
        classes = cast(List[Dict[str, Any]], finfo.get("classes") or [])
        for cls in classes:
            cls_name: str = cast(str, cls.get("name") or "?")
            if _is_marker_exception_class(cls) or _is_data_container_class(cls):
                continue
            # empty class? runtime can confirm "no methods, no bases other than object"
            cls_ctx: Dict[str, Any] = cast(
                Dict[str, Any], cls.get("source_context") or {}
            )
            inheritance = cast(Dict[str, Any], cls.get("inheritance") or {})
            if not (cls.get("methods") or []) and not (inheritance.get("bases") or []):
                findings.append(
                    {
                        "type": "Improper Implementation (runtime)",
                        "subtype": "Bare Class",
                        "file": nf,
                        "line": cast(List[int], cls_ctx.get("line_range", [0]))[0],
                        "content": f"class {cls_name}: (no methods, no bases)",
                    }
                )
            methods = cast(List[Dict[str, Any]], cls.get("methods") or [])
            for meth in methods:
                meth_name = cast(str, meth.get("name") or "?")
                qn = f"{cls_name}.{meth_name}"
                _emit_runtime_issues_for_symbol(
                    findings,
                    idx,
                    nf,
                    meth,
                    kind="method",
                    qualname=qn,
                    owning_class=cls,
                )
    return findings


def _emit_runtime_issues_for_symbol(
    sink: List[Dict[str, Any]],
    idx: RuntimeIndex,
    file_path: str,
    sym: Dict[str, Any],
    kind: str,
    qualname: str,
    owning_class: Optional[Dict[str, Any]] = None,
) -> None:
    name: str = cast(str, sym.get("name") or qualname)
    ctx: Dict[str, Any] = cast(Dict[str, Any], sym.get("source_context") or {})
    line: int = cast(List[int], ctx.get("line_range", [0]))[0]

    is_trivial = _has_trivial_body(sym)
    annotated_return = _annotated_non_trivial_return(sym)
    decorators = cast(List[str], sym.get("decorators") or [])
    is_async = bool(sym.get("is_async")) or any(
        "async" in str(d).lower() for d in decorators
    )
    is_abstract = any("abstract" in str(d).lower() for d in decorators)
    belongs_to_protocol = owning_class is not None and _is_protocol_class(owning_class)

    # 1. Annotated stub
    if is_trivial and annotated_return and not is_abstract and not belongs_to_protocol:
        sink.append(
            {
                "type": "Incomplete Implementation (runtime)",
                "subtype": "Annotated Stub",
                "file": file_path,
                "line": line,
                "content": f"{qualname}{sym.get('signature','')} -> {annotated_return}",
                "_runtime_only": True,
                "_qualname": qualname,
            }
        )

    # 2. Exported placeholder
    if is_trivial and idx.is_exported(file_path, name):
        sink.append(
            {
                "type": "Incomplete Implementation (runtime)",
                "subtype": "Exported Placeholder",
                "file": file_path,
                "line": line,
                "content": f"{qualname} is in __all__ but body is trivial",
                "_runtime_only": True,
                "_qualname": qualname,
            }
        )

    # 3. Async without await (heuristic — no 'await' or awaitable attr access)
    if is_async and is_trivial and not belongs_to_protocol:
        sink.append(
            {
                "type": "Improper Implementation (runtime)",
                "subtype": "Async Stub",
                "file": file_path,
                "line": line,
                "content": f"async {qualname} has trivial body",
                "_runtime_only": True,
                "_qualname": qualname,
            }
        )

    # 4. Stub method on non-abstract class
    if kind == "method" and is_trivial and owning_class is not None:
        if (
            not belongs_to_protocol
            and not RuntimeIndex.is_in_abstract_mro(owning_class)
            and not is_abstract
        ):
            sink.append(
                {
                    "type": "Improper Implementation (runtime)",
                    "subtype": "Concrete Stub Method",
                    "file": file_path,
                    "line": line,
                    "content": f"{qualname} is a stub on a non-abstract class",
                    "_runtime_only": True,
                    "_qualname": qualname,
                }
            )

    # 5. Orphan — exported / public name with zero cross-file references
    if name and not name.startswith("_"):
        callers = idx.callers_of(name, exclude_file=file_path)
        if not callers and idx.is_exported(file_path, name):
            sink.append(
                {
                    "type": "Unused Item (runtime)",
                    "subtype": "Orphan Export",
                    "file": file_path,
                    "line": line,
                    "content": f"{qualname} exported but no other file references it",
                    "_runtime_only": True,
                    "_qualname": qualname,
                }
            )


# ===========================================================================
# Enrichment: attach runtime context to every issue
# ===========================================================================
def enrich_issue(issue: Dict[str, Any], idx: RuntimeIndex) -> Dict[str, Any]:
    """Attach runtime-derived context fields to a single issue dict."""
    ctx: Dict[str, Any] = {}
    file_path: str = issue.get("file", "")
    line: int = int(issue.get("line", 0) or 0)

    sym: Optional[Dict[str, Any]] = idx.symbol_at(file_path, line) if line else None
    enclosing_class: Optional[Dict[str, Any]] = (
        idx.enclosing_class(file_path, line) if line else None
    )
    if sym:
        ctx["owning_symbol"] = {
            "name": sym.get("name"),
            "qualname": issue.get("_qualname") or sym.get("name"),
            "kind": sym.get("_kind"),
            "signature": sym.get("signature"),
            "docstring": (sym.get("docstring") or "")[:240] or None,
            "line_range": list(sym.get("_line_range", [])),
        }
        if sym.get("type_annotations"):
            ctx["type_annotations"] = sym["type_annotations"]
        if sym.get("decorators"):
            ctx["decorators"] = sym["decorators"]
        if sym.get("inheritance"):
            ctx["inheritance"] = sym["inheritance"]
        scope: Dict[str, Any] = sym.get("scope_references") or {}
        if scope:
            ctx["scope_references"] = scope
        if sym.get("closure_dependencies"):
            ctx["closure_dependencies"] = sym["closure_dependencies"]
        if sym.get("attribute_accesses"):
            ctx["attribute_accesses"] = list(sym["attribute_accesses"])[:25]
        ctx["abstract_method"] = _is_abstract_method(sym)
        if sym.get("_kind") == "class":
            ctx["data_container_class"] = _is_data_container_class(sym)

        # Cross-file links
        name = sym.get("name")
        if name:
            callers = idx.callers_of(name, exclude_file=file_path)
            if callers:
                ctx["linked_areas"] = {
                    "callers": callers[:20],
                    "caller_count": len(callers),
                }
            ctx["exported"] = idx.is_exported(file_path, name)

    if enclosing_class:
        ctx["abstract_class"] = _is_abstract_class(enclosing_class)
        if ctx.get("data_container_class") is None:
            ctx["data_container_class"] = _is_data_container_class(enclosing_class)
        if not ctx.get("inheritance") and enclosing_class.get("inheritance"):
            ctx["inheritance"] = enclosing_class["inheritance"]

    # Severity heuristic
    ctx["severity"] = _score_severity(issue, ctx)
    issue_copy = dict(issue)
    issue_copy["context"] = ctx
    return issue_copy


def _score_severity(issue: Dict[str, Any], ctx: Dict[str, Any]) -> str:
    score = 0
    sub = issue.get("subtype", "")
    if "NotImplementedError" in sub:
        score += 2
    if sub in {"Annotated Stub", "Exported Placeholder", "Concrete Stub Method"}:
        score += 2
    if sub in {"Empty/Stub Function", "Empty/Stub Class", "Async Stub"}:
        score += 1
    if ctx.get("exported"):
        score += 1
    if ctx.get("linked_areas", {}).get("caller_count", 0) > 0:
        score += 1
    if sub in {"TODO", "FIXME", "for now", "simplified", "placeholder", "in a real"}:
        score += 0  # informational by default
    if score >= 4:
        return "critical"
    if score >= 2:
        return "high"
    if score >= 1:
        return "medium"
    return "low"


# ===========================================================================
# Reporting
# ===========================================================================
_SEV_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def generate_report(issues: List[Dict[str, Any]], unused: List[Dict[str, Any]]) -> None:
    # Persist machine-readable JSON first (full context preserved)
    try:
        os.makedirs(os.path.dirname(OUTPUT_JSON_FILE), exist_ok=True)
        with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
            json.dump(
                {"issues": issues, "unused": unused}, f, indent=2, ensure_ascii=False
            )
    except Exception as e:
        print(f"[report] Failed writing JSON report: {e}")

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("# Code Analysis Issues Report\n\n")
        f.write("_Static analysis enhanced with runtime inspector data._\n\n")

        # Summary
        by_sev: Dict[str, int] = defaultdict(int)
        for it in issues:
            it_ctx: Dict[str, Any] = it.get("context") or {}
            by_sev[it_ctx.get("severity", "low")] += 1
        f.write("## Summary\n")
        f.write(f"- Total issues: **{len(issues)}**\n")
        for s in ("critical", "high", "medium", "low"):
            if by_sev.get(s):
                f.write(f"- {s.title()}: {by_sev[s]}\n")
        f.write(f"- Unused items (pyright): {len(unused)}\n\n")

        f.write("## Incomplete & Improper Items\n")
        if not issues:
            f.write("No incomplete items found.\n")
        else:

            def _sort_key(x: Dict[str, Any]) -> Tuple[int, str, int]:
                x_ctx = cast(Dict[str, Any], x.get("context") or {})
                x_sev = cast(str, x_ctx.get("severity", "low"))
                return (
                    _SEV_ORDER.get(x_sev, 9),
                    cast(str, x.get("file", "")),
                    cast(int, x.get("line", 0)),
                )

            issues_sorted = sorted(issues, key=_sort_key)
            for it in issues_sorted:
                ctx: Dict[str, Any] = it.get("context") or {}
                sev = ctx.get("severity", "low")
                f.write(
                    f"- **[{sev.upper()}] {it.get('subtype','?')}** "
                    f"in `{it.get('file','?')}:{it.get('line','?')}`\n"
                )
                content = it.get("content", "")
                if content:
                    f.write(f"  ```\n  {content}\n  ```\n")

                owning: Optional[Dict[str, Any]] = ctx.get("owning_symbol")
                if owning:
                    sig: str = owning.get("signature") or ""
                    f.write(
                        f"  - **Owning symbol**: `{owning.get('qualname')}`{sig} "
                        f"({owning.get('kind')})\n"
                    )
                if ctx.get("type_annotations"):
                    f.write(f"  - **Types**: `{ctx['type_annotations']}`\n")
                if ctx.get("decorators"):
                    f.write(f"  - **Decorators**: {ctx['decorators']}\n")
                if ctx.get("inheritance"):
                    inh = ctx["inheritance"]
                    if inh.get("bases") or inh.get("mro"):
                        f.write(
                            f"  - **Inheritance**: bases={inh.get('bases', [])} "
                            f"mro={inh.get('mro', [])}\n"
                        )
                if ctx.get("scope_references"):
                    sr: Dict[str, Any] = ctx["scope_references"]
                    g_list: List[str] = sr.get("globals") or []
                    nl_list: List[str] = sr.get("nonlocals") or []
                    g = g_list[:10]
                    nl = nl_list[:10]
                    if g or nl:
                        f.write(f"  - **Scope refs**: globals={g} nonlocals={nl}\n")
                if ctx.get("closure_dependencies"):
                    f.write(f"  - **Closure deps**: {ctx['closure_dependencies']}\n")
                la: Dict[str, Any] = cast(Dict[str, Any], ctx.get("linked_areas") or {})
                callers_attr = la.get("callers")
                if callers_attr:
                    callers_list: List[str] = cast(List[str], callers_attr)
                    f.write(
                        f"  - **Linked areas** ({la.get('caller_count')} files): "
                        + ", ".join(f"`{c}`" for c in callers_list)
                        + "\n"
                    )
                if ctx.get("exported") is True:
                    f.write("  - **Exported**: yes (in `__all__`)\n")

        f.write("\n## Unused Items\n")
        if unused:
            for item in unused:
                f.write(f"- **{item['subtype']}** in `{item['file']}:{item['line']}`\n")
                f.write(f"  > {item['content']}\n")
        else:
            f.write("No unused items found (or pyright output missing).\n")


# ===========================================================================
# Driver
# ===========================================================================
def maybe_run_runtime_inspector(project_root_path: str) -> None:
    """Best-effort: run the inspector if its output is missing.

    Controlled by env var CRCT_AUTO_RUNTIME=1 to avoid surprising users.
    """
    if os.environ.get("CRCT_AUTO_RUNTIME") != "1":
        return
    target = os.path.join(project_root_path, RUNTIME_SYMBOLS_PATH)
    if os.path.exists(target):
        return
    print("[runtime] CRCT_AUTO_RUNTIME=1; invoking runtime_inspector...")
    try:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "cline_utils.dependency_system.analysis.runtime_inspector",
                project_root_path,
            ],
            cwd=project_root_path,
            check=False,
        )
    except Exception as e:
        print(f"[runtime] Auto-run failed: {e}")


def main():
    all_issues: List[Dict[str, Any]] = []

    code_roots = config.get_code_root_directories()
    excluded_paths = config.get_excluded_paths()

    # ---- pyright ----
    try:
        print("Running pyright for unused item analysis...")
        with open(PYRIGHT_OUTPUT, "w") as f:
            result = subprocess.run(
                ["pyright", "--outputjson"],
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=project_root,
            )
        if result.returncode == 0:
            print("Pyright analysis completed successfully.")
        else:
            print(
                f"Pyright completed with warnings/errors (exit code {result.returncode}). "
                f"Output file generated."
            )
    except Exception as e:
        print(f"Warning: Unexpected error running pyright: {e}")

    # ---- runtime data ----
    maybe_run_runtime_inspector(project_root)
    runtime_data = load_runtime_data(project_root)
    idx = RuntimeIndex(runtime_data)

    # ---- static walk ----
    print(f"Scanning code roots: {code_roots}")
    for root_dir in code_roots:
        start_dir = root_dir
        if not os.path.exists(start_dir):
            print(f"Warning: Code root {start_dir} does not exist. Skipping.")
            continue
        for root, dirs, files in os.walk(start_dir):
            dirs[:] = [
                d
                for d in dirs
                if not path_utils.is_path_excluded(
                    os.path.join(root, d), excluded_paths
                )
            ]
            for file in files:
                filepath = os.path.join(root, file)
                if path_utils.is_path_excluded(filepath, excluded_paths):
                    continue
                if Path(file).suffix not in EXTENSIONS:
                    continue
                all_issues.extend(scan_file(filepath))

    # ---- runtime-only findings ----
    if runtime_data:
        rt_findings = runtime_only_findings(idx)
        print(f"[runtime] Added {len(rt_findings)} runtime-only finding(s).")
        all_issues.extend(rt_findings)

    # ---- enrich every issue with runtime context ----
    if runtime_data:
        all_issues = [enrich_issue(it, idx) for it in all_issues]
        all_issues = [
            it
            for it in all_issues
            if not _should_suppress_issue(
                it, cast(Dict[str, Any], it.get("context") or {})
            )
        ]
    else:
        # Still attach an empty context block so downstream consumers
        # have a stable schema.
        for it in all_issues:
            it.setdefault("context", {"severity": _score_severity(it, {})})

    # ---- de-dup: same (file, line, subtype) collapses, keep richest ctx ----
    seen: Dict[Tuple[str, int, str], Dict[str, Any]] = {}
    for it in all_issues:
        key = (it.get("file", ""), it.get("line", 0), it.get("subtype", ""))
        cur = seen.get(key)
        if cur is None or len((it.get("context") or {})) > len(
            (cur.get("context") or {})
        ):
            seen[key] = it
    all_issues = list(seen.values())

    unused_items = get_unused_items()
    generate_report(all_issues, unused_items)
    print(f"Report generated at {OUTPUT_FILE}")
    print(f"JSON report at {OUTPUT_JSON_FILE}")


if __name__ == "__main__":
    main()
