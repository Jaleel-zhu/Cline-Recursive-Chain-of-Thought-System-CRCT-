# runtime_inspector.py
from __future__ import annotations

import inspect
import sys
import os
import json
import importlib.util
import logging
import typing
import ast
import textwrap
from collections.abc import Iterator
from types import CodeType, FunctionType, ModuleType
from typing import Any, Optional, cast

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

JSONDict = dict[str, Any]
StringList = list[str]


def get_type_annotations(obj: object) -> JSONDict:
    """Extract parameter and return type annotations."""
    try:
        return {
            "parameters": {
                k: str(v)
                for k, v in typing.get_type_hints(
                    cast(Any, obj), include_extras=True
                ).items()
            },
            "return_type": str(inspect.signature(cast(Any, obj)).return_annotation),
        }
    except Exception:
        return {}


def get_source_context(obj: object, code_roots: list[str]) -> JSONDict:
    """
    Get source file location and import context.
    Returns empty dict if source is outside code roots.
    """
    try:
        source_file = inspect.getsourcefile(cast(Any, obj))
        if not source_file:
            return {}

        # Normalize and validate against code roots
        from cline_utils.dependency_system.utils.path_utils import (
            normalize_path,
            is_subpath,
        )

        norm_source = normalize_path(source_file)

        # Check if file is within any code root
        in_code_roots = False
        for code_root in code_roots:
            norm_root = normalize_path(code_root)
            if norm_source == norm_root or is_subpath(norm_source, norm_root):
                in_code_roots = True
                break

        if not in_code_roots:
            logger.debug(f"Skipping source outside code roots: {norm_source}")
            return {}

        source_lines, start_line = inspect.getsourcelines(cast(Any, obj))
        # Strip line endings to prevent escape artifacts in JSON (improves embedding quality)
        clean_source_lines = [line.rstrip("\n").rstrip("\r") for line in source_lines]
        return {
            "file": norm_source,
            "line_range": (start_line, start_line + len(source_lines)),
            "source_lines": clean_source_lines,
        }
    except Exception:
        return {}


def get_module_exports(module: ModuleType) -> dict[str, str]:
    """Identify all exported symbols and their origins."""
    exports: dict[str, str] = {}
    exported_names = getattr(module, "__all__", None)
    if isinstance(exported_names, (list, tuple, set)):
        normalized_export_names: list[str] = []
        for raw_name in cast(list[object], list(cast(Any, exported_names))):
            if isinstance(raw_name, str):
                normalized_export_names.append(raw_name)
        for name in normalized_export_names:
            obj = getattr(module, name, None)
            if obj is None:
                continue
            module_obj = inspect.getmodule(obj)
            if module_obj is not None:
                exports[name] = module_obj.__name__
    return exports


def get_inheritance_info(cls: type[Any], code_roots: list[str]) -> JSONDict:
    """
    Extract inheritance hierarchy and method resolution order.
    Only includes bases/mro that are within code roots.
    """
    from cline_utils.dependency_system.utils.path_utils import (
        normalize_path,
        is_subpath,
    )

    try:
        bases: StringList = []
        for base in cls.__bases__:
            try:
                base_file = inspect.getsourcefile(base)
                if base_file:
                    norm_base_file = normalize_path(base_file)
                    # Check if base is in code roots
                    in_roots = any(
                        norm_base_file == normalize_path(root)
                        or is_subpath(norm_base_file, normalize_path(root))
                        for root in code_roots
                    )
                    if in_roots:
                        bases.append(base.__module__ + "." + base.__qualname__)
            except (TypeError, AttributeError):
                pass

        mro: StringList = []
        for c in inspect.getmro(cls)[1:]:  # Skip self
            try:
                c_file = inspect.getsourcefile(c)
                if c_file:
                    norm_c_file = normalize_path(c_file)
                    in_roots = any(
                        norm_c_file == normalize_path(root)
                        or is_subpath(norm_c_file, normalize_path(root))
                        for root in code_roots
                    )
                    if in_roots:
                        mro.append(c.__module__ + "." + c.__qualname__)
            except (TypeError, AttributeError):
                pass

        return {"bases": bases, "mro": mro}
    except Exception:
        return {}


def get_closure_dependencies(func: FunctionType, code_roots: list[str]) -> list[str]:
    """
    Identify variables captured in function closures.
    Only includes modules within code roots.
    """
    from cline_utils.dependency_system.utils.path_utils import (
        normalize_path,
        is_subpath,
    )

    deps: set[str] = set()
    if inspect.isfunction(func) and func.__closure__:
        for cell in func.__closure__:
            try:
                obj = cell.cell_contents
                module = inspect.getmodule(obj)
                if module:
                    try:
                        module_file = inspect.getsourcefile(module)
                        if module_file:
                            norm_module_file = normalize_path(module_file)
                            in_roots = any(
                                norm_module_file == normalize_path(root)
                                or is_subpath(norm_module_file, normalize_path(root))
                                for root in code_roots
                            )
                            if in_roots:
                                deps.add(module.__name__)
                    except (TypeError, AttributeError):
                        pass
            except Exception:
                pass
    return sorted(deps)


def _get_declaration_node(
    source_code: str,
) -> Optional[ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return the primary class/function node for a source snippet."""
    try:
        dedented_source = textwrap.dedent(source_code)
        tree = ast.parse(dedented_source)
    except Exception:
        return None

    for node in tree.body:
        if isinstance(
            node,
            (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef),
        ):
            return node
    return None


def get_decorator_info(obj: object) -> list[str]:
    """
    Extract declared decorator expressions from source.

    This is intentionally semantic rather than dependency-scoped:
    SES needs the decorators the file actually declares, even when the
    decorator originates from the stdlib or a third-party package.
    """
    try:
        source = inspect.getsource(cast(Any, obj))
    except Exception:
        return []

    node = _get_declaration_node(source)
    if node is None:
        return []

    decorators: list[str] = []
    for decorator in node.decorator_list:
        try:
            decorator_text = ast.unparse(decorator).strip()
        except Exception:
            continue
        if decorator_text:
            decorators.append(decorator_text)
    return decorators


def get_scope_references(func: FunctionType) -> dict[str, list[str]]:
    """Extract global and nonlocal variable references."""
    try:
        code: CodeType = func.__code__
        return {"globals": list(code.co_names), "nonlocals": list(code.co_freevars)}
    except Exception:
        return {}


def get_attribute_accesses(source_code: str) -> list[str]:
    """Parse source to identify attribute access patterns."""
    accesses: set[str] = set()
    try:
        dedented_source = textwrap.dedent(source_code)
        tree = ast.parse(dedented_source)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                accesses.add(node.attr)
    except Exception:
        pass
    return sorted(accesses)


def iter_declared_methods(cls: type[Any]) -> Iterator[tuple[str, FunctionType]]:
    """
    Yield methods declared directly on a class, preserving descriptor-backed
    APIs such as @property, @classmethod, and @staticmethod.
    """
    seen: set[str] = set()
    members = cast(dict[str, object], cls.__dict__)
    for method_name, member in members.items():
        func: Optional[FunctionType] = None
        if isinstance(member, property):
            getter = member.fget
            if getter is not None:
                func = getter
        elif isinstance(member, (staticmethod, classmethod)):
            func_candidate = cast(Any, member).__func__
            if inspect.isfunction(func_candidate):
                func = func_candidate
        elif inspect.isfunction(member):
            func = member

        if func is None or method_name in seen:
            continue
        seen.add(method_name)
        yield method_name, func


def get_module_info(
    file_path: str, module_name: str, code_roots: list[str]
) -> JSONDict:
    """
    Safely imports a module and extracts symbol information using inspect.
    All collected paths are validated against code_roots.
    """
    file_dir = os.path.dirname(file_path)
    try:
        # Add file directory to sys.path to handle relative imports if needed
        if file_dir not in sys.path:
            sys.path.insert(0, file_dir)

        spec = importlib.util.spec_from_file_location(module_name, file_path)
        if not spec or not spec.loader:
            return {}

        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)

        runtime_classes: list[JSONDict] = []
        runtime_functions: list[JSONDict] = []
        symbols: JSONDict = {
            "classes": runtime_classes,
            "functions": runtime_functions,
            "exports": get_module_exports(module),
        }

        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and obj.__module__ == module_name:
                source_context = get_source_context(obj, code_roots)

                # Skip if source is outside code roots
                if not source_context:
                    logger.debug(f"Skipping class {name} - source outside code roots")
                    continue

                class_info: JSONDict = {
                    "name": name,
                    "docstring": inspect.getdoc(obj),
                    "inheritance": get_inheritance_info(obj, code_roots),
                    "decorators": get_decorator_info(obj),
                    "source_context": source_context,
                    "methods": [],
                }

                for method_name, method in iter_declared_methods(obj):
                    # Validate method source
                    method_source_context = get_source_context(method, code_roots)
                    if not method_source_context:
                        logger.debug(
                            f"Skipping method {method_name} - source outside code roots"
                        )
                        continue

                    try:
                        sig = str(inspect.signature(method))
                    except ValueError:
                        sig = "(...)"

                    # Get source code for attribute access analysis
                    try:
                        source = inspect.getsource(method)
                        attr_accesses = get_attribute_accesses(source)
                    except Exception:
                        attr_accesses = []

                    method_entries = cast(list[JSONDict], class_info["methods"])
                    method_entries.append(
                        {
                            "name": method_name,
                            "signature": sig,
                            "docstring": inspect.getdoc(method),
                            "type_annotations": get_type_annotations(method),
                            "closure_dependencies": get_closure_dependencies(
                                method, code_roots
                            ),
                            "scope_references": get_scope_references(method),
                            "decorators": get_decorator_info(method),
                            "source_context": method_source_context,
                            "attribute_accesses": attr_accesses,
                        }
                    )

                runtime_classes.append(class_info)

            elif inspect.isfunction(obj) and obj.__module__ == module_name:
                source_context = get_source_context(obj, code_roots)

                # Skip if source is outside code roots
                if not source_context:
                    logger.debug(
                        f"Skipping function {name} - source outside code roots"
                    )
                    continue

                try:
                    sig = str(inspect.signature(obj))
                except ValueError:
                    sig = "(...)"

                # Get source code for attribute access analysis
                try:
                    source = inspect.getsource(obj)
                    attr_accesses = get_attribute_accesses(source)
                except Exception:
                    attr_accesses = []

                runtime_functions.append(
                    {
                        "name": name,
                        "signature": sig,
                        "docstring": inspect.getdoc(obj),
                        "type_annotations": get_type_annotations(obj),
                        "closure_dependencies": get_closure_dependencies(
                            obj, code_roots
                        ),
                        "scope_references": get_scope_references(obj),
                        "decorators": get_decorator_info(obj),
                        "source_context": source_context,
                        "attribute_accesses": attr_accesses,
                    }
                )

        return symbols

    except Exception as e:
        logger.warning(f"Failed to inspect {file_path}: {e}")
        return {}
    finally:
        # Cleanup sys.path
        if file_dir in sys.path:
            sys.path.remove(file_dir)


def main():
    if len(sys.argv) < 2:
        print("Usage: python runtime_inspector.py <project_root>")
        sys.exit(1)

    project_root = os.path.abspath(sys.argv[1])

    # Add project root to sys.path to allow importing cline_utils
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

    try:
        from cline_utils.dependency_system.utils.config_manager import ConfigManager
        from cline_utils.dependency_system.utils.path_utils import normalize_path
    except ImportError as e:
        logger.error(
            f"Could not import ConfigManager: {e}. Ensure cline_utils is in python path."
        )
        sys.exit(1)

    # Initialize ConfigManager
    original_cwd = os.getcwd()
    os.chdir(project_root)

    try:
        config_manager = ConfigManager()

        # Get configuration - code_roots are already normalized by config_manager
        code_roots = config_manager.get_code_root_directories()
        excluded_dirs = set(config_manager.get_excluded_dirs())
        excluded_extensions = set(config_manager.get_excluded_extensions())
        excluded_paths = set(config_manager.get_excluded_paths())

        logger.info(f"Loaded configuration. Code roots: {code_roots}")

        # Convert relative code roots to absolute paths
        absolute_code_roots: list[str] = []
        for root_dir_rel in code_roots:
            if os.path.isabs(root_dir_rel):
                absolute_code_roots.append(normalize_path(root_dir_rel))
            else:
                absolute_code_roots.append(
                    normalize_path(os.path.join(project_root, root_dir_rel))
                )

        logger.info(f"Absolute code roots for validation: {absolute_code_roots}")

        # Save to cline_utils/dependency_system/core/runtime_symbols.json
        core_dir = os.path.join(
            project_root, "cline_utils", "dependency_system", "core"
        )
        os.makedirs(core_dir, exist_ok=True)
        output_file = os.path.join(core_dir, "runtime_symbols.json")

        all_symbols: dict[str, JSONDict] = {}

        if not code_roots:
            logger.warning(
                "No code roots defined in configuration. Skipping runtime inspection."
            )
            sys.exit(0)

        # Process each root
        for root_dir_rel in code_roots:
            # Resolve to absolute path
            if os.path.isabs(root_dir_rel):
                root_dir = root_dir_rel
            else:
                root_dir = os.path.join(project_root, root_dir_rel)

            root_dir = normalize_path(root_dir)

            if not os.path.exists(root_dir):
                logger.warning(f"Code root not found: {root_dir}")
                continue

            logger.info(f"Scanning root: {root_dir}")

            for root, dirs, files in os.walk(root_dir):
                root = normalize_path(root)

                # Modify dirs in-place to skip excluded directories
                dirs[:] = [d for d in dirs if d not in excluded_dirs]

                # Filter by path (excluded_paths)
                valid_dirs: list[str] = []
                for d in dirs:
                    dir_path = normalize_path(os.path.join(root, d))
                    if dir_path not in excluded_paths:
                        valid_dirs.append(d)
                dirs[:] = valid_dirs

                for file in files:
                    if not file.endswith(".py") or file.startswith("__"):
                        continue

                    _, ext = os.path.splitext(file)
                    if ext in excluded_extensions:
                        continue

                    file_path = normalize_path(os.path.join(root, file))
                    if file_path in excluded_paths:
                        continue

                    # Construct a module name (approximate)
                    rel_path = os.path.relpath(file_path, project_root)
                    module_name = rel_path.replace(os.sep, ".").replace(".py", "")

                    logger.info(f"Inspecting {module_name}...")

                    # Pass absolute_code_roots to get_module_info for validation
                    info = get_module_info(file_path, module_name, absolute_code_roots)
                    if info:
                        all_symbols[file_path] = info

        # Use ensure_ascii=False to prevent escape character pollution in embeddings
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(all_symbols, f, indent=2, ensure_ascii=False)

        logger.info(f"Runtime inspection complete. Saved to {output_file}")

    finally:
        os.chdir(original_cwd)


if __name__ == "__main__":
    main()
