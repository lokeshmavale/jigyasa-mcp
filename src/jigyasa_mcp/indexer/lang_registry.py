"""Multi-language tree-sitter registry with auto-detection and lazy grammar loading.

Maps file extensions to tree-sitter grammars. Grammars are loaded lazily on
first use — if the pip package isn't installed, falls back gracefully to
text-based chunking.

Supported languages (when grammar is installed):
  Python, TypeScript, JavaScript, Go, C#, Rust, C, C++, Ruby, Kotlin, Scala,
  Swift, PHP, Bash, Lua, Haskell, R, Java (always available)
"""

import importlib
import logging
import subprocess
import sys
from dataclasses import dataclass, field

from tree_sitter import Language, Parser

logger = logging.getLogger(__name__)


@dataclass
class LanguageProfile:
    """Defines how to extract symbols from a language's AST.

    Maps tree-sitter node types to jigyasa symbol kinds (class, method, field, etc.).
    Each language has different AST node names for the same concepts.
    """
    name: str
    extensions: list[str]
    pip_package: str
    module_name: str  # Python module to import
    # Function name to call on the module to get the language.
    # Default is "language". Some packages (e.g., tree-sitter-typescript)
    # expose separate functions like "language_typescript" / "language_tsx".
    language_func: str = "language"

    # AST node types → symbol kind mappings
    class_nodes: list[str] = field(default_factory=list)
    function_nodes: list[str] = field(default_factory=list)
    method_nodes: list[str] = field(default_factory=list)
    field_nodes: list[str] = field(default_factory=list)
    constructor_nodes: list[str] = field(default_factory=list)
    interface_nodes: list[str] = field(default_factory=list)
    enum_nodes: list[str] = field(default_factory=list)

    # How to find the name of a declaration
    name_node_type: str = "identifier"
    # Node types that indicate visibility/decorators
    decorator_node_types: list[str] = field(default_factory=list)
    # Comment node types (for docstrings)
    comment_node_types: list[str] = field(default_factory=list)

    # Import extraction regex pattern (applied to source text)
    import_pattern: str = ""
    # Package/namespace extraction regex
    package_pattern: str = ""

    def node_to_kind(self, node_type: str) -> str | None:
        """Map a tree-sitter node type to a symbol kind."""
        if node_type in self.class_nodes:
            return "class"
        if node_type in self.function_nodes:
            return "function"
        if node_type in self.method_nodes:
            return "method"
        if node_type in self.field_nodes:
            return "field"
        if node_type in self.constructor_nodes:
            return "constructor"
        if node_type in self.interface_nodes:
            return "interface"
        if node_type in self.enum_nodes:
            return "enum"
        return None

    @property
    def all_declaration_nodes(self) -> set[str]:
        """All node types that represent declarations."""
        return set(
            self.class_nodes + self.function_nodes + self.method_nodes
            + self.field_nodes + self.constructor_nodes
            + self.interface_nodes + self.enum_nodes
        )


# ---------------------------------------------------------------------------
# Language profile definitions
# ---------------------------------------------------------------------------

PYTHON_PROFILE = LanguageProfile(
    name="python",
    extensions=[".py"],
    pip_package="tree-sitter-python",
    module_name="tree_sitter_python",
    class_nodes=["class_definition"],
    function_nodes=["function_definition"],
    method_nodes=[],  # methods are function_definition inside class_definition
    field_nodes=["assignment", "augmented_assignment"],
    decorator_node_types=["decorator"],
    comment_node_types=["comment", "string"],
    name_node_type="identifier",
    import_pattern=r"^\s*(?:from\s+([\w.]+)\s+)?import\s+([\w., ]+)",
    package_pattern="",
)

TYPESCRIPT_PROFILE = LanguageProfile(
    name="typescript",
    extensions=[".ts"],
    pip_package="tree-sitter-typescript",
    module_name="tree_sitter_typescript",
    language_func="language_typescript",
    class_nodes=["class_declaration"],
    function_nodes=["function_declaration", "arrow_function"],
    method_nodes=["method_definition"],
    field_nodes=["public_field_definition", "property_signature"],
    constructor_nodes=[],
    interface_nodes=["interface_declaration"],
    enum_nodes=["enum_declaration"],
    decorator_node_types=["decorator"],
    comment_node_types=["comment"],
    name_node_type="identifier",
    import_pattern=r"""import\s+.*?from\s+['"]([^'"]+)['"]""",
    package_pattern="",
)

TSX_PROFILE = LanguageProfile(
    name="tsx",
    extensions=[".tsx"],
    pip_package="tree-sitter-typescript",
    module_name="tree_sitter_typescript",
    language_func="language_tsx",
    class_nodes=["class_declaration"],
    function_nodes=["function_declaration", "arrow_function"],
    method_nodes=["method_definition"],
    field_nodes=["public_field_definition", "property_signature"],
    constructor_nodes=[],
    interface_nodes=["interface_declaration"],
    enum_nodes=["enum_declaration"],
    decorator_node_types=["decorator"],
    comment_node_types=["comment"],
    name_node_type="identifier",
    import_pattern=r"""import\s+.*?from\s+['"]([^'"]+)['"]""",
    package_pattern="",
)

JAVASCRIPT_PROFILE = LanguageProfile(
    name="javascript",
    extensions=[".js", ".jsx", ".mjs", ".cjs"],
    pip_package="tree-sitter-javascript",
    module_name="tree_sitter_javascript",
    class_nodes=["class_declaration"],
    function_nodes=["function_declaration", "arrow_function"],
    method_nodes=["method_definition"],
    field_nodes=["field_definition"],
    constructor_nodes=[],
    interface_nodes=[],
    enum_nodes=[],
    decorator_node_types=["decorator"],
    comment_node_types=["comment"],
    name_node_type="identifier",
    import_pattern=r"""(?:import|require)\s*\(?['"]([^'"]+)['"]\)?""",
    package_pattern="",
)

GO_PROFILE = LanguageProfile(
    name="go",
    extensions=[".go"],
    pip_package="tree-sitter-go",
    module_name="tree_sitter_go",
    class_nodes=["type_declaration"],
    function_nodes=["function_declaration"],
    method_nodes=["method_declaration"],
    field_nodes=["field_declaration"],
    constructor_nodes=[],
    interface_nodes=["type_declaration"],  # filtered by body type
    enum_nodes=[],
    comment_node_types=["comment"],
    name_node_type="identifier",
    import_pattern=r"""import\s+(?:\(\s*)?"([^"]+)""",
    package_pattern=r"^package\s+(\w+)",
)

CSHARP_PROFILE = LanguageProfile(
    name="c_sharp",
    extensions=[".cs"],
    pip_package="tree-sitter-c-sharp",
    module_name="tree_sitter_c_sharp",
    class_nodes=["class_declaration", "record_declaration"],
    function_nodes=[],
    method_nodes=["method_declaration"],
    field_nodes=["field_declaration", "property_declaration"],
    constructor_nodes=["constructor_declaration"],
    interface_nodes=["interface_declaration"],
    enum_nodes=["enum_declaration"],
    decorator_node_types=["attribute_list"],
    comment_node_types=["comment"],
    name_node_type="identifier",
    import_pattern=r"^\s*using\s+([\w.]+);",
    package_pattern=r"^\s*namespace\s+([\w.]+)",
)

RUST_PROFILE = LanguageProfile(
    name="rust",
    extensions=[".rs"],
    pip_package="tree-sitter-rust",
    module_name="tree_sitter_rust",
    class_nodes=["struct_item"],
    function_nodes=["function_item"],
    method_nodes=[],
    field_nodes=["field_declaration"],
    constructor_nodes=[],
    interface_nodes=["trait_item"],
    enum_nodes=["enum_item"],
    decorator_node_types=["attribute_item"],
    comment_node_types=["line_comment", "block_comment"],
    name_node_type="identifier",
    import_pattern=r"^\s*use\s+([\w:]+)",
    package_pattern="",
)

C_PROFILE = LanguageProfile(
    name="c",
    extensions=[".c", ".h"],
    pip_package="tree-sitter-c",
    module_name="tree_sitter_c",
    class_nodes=["struct_specifier"],
    function_nodes=["function_definition"],
    method_nodes=[],
    field_nodes=["field_declaration"],
    constructor_nodes=[],
    interface_nodes=[],
    enum_nodes=["enum_specifier"],
    comment_node_types=["comment"],
    name_node_type="identifier",
    import_pattern=r"""#include\s+[<"]([^>"]+)[>"]""",
    package_pattern="",
)

CPP_PROFILE = LanguageProfile(
    name="cpp",
    extensions=[".cpp", ".cc", ".cxx", ".hpp", ".hh"],
    pip_package="tree-sitter-cpp",
    module_name="tree_sitter_cpp",
    class_nodes=["class_specifier", "struct_specifier"],
    function_nodes=["function_definition"],
    method_nodes=[],
    field_nodes=["field_declaration"],
    constructor_nodes=[],
    interface_nodes=[],
    enum_nodes=["enum_specifier"],
    comment_node_types=["comment"],
    name_node_type="identifier",
    import_pattern=r"""#include\s+[<"]([^>"]+)[>"]""",
    package_pattern=r"^\s*namespace\s+(\w+)",
)

RUBY_PROFILE = LanguageProfile(
    name="ruby",
    extensions=[".rb"],
    pip_package="tree-sitter-ruby",
    module_name="tree_sitter_ruby",
    class_nodes=["class", "module"],
    function_nodes=[],
    method_nodes=["method", "singleton_method"],
    field_nodes=[],
    constructor_nodes=[],
    interface_nodes=[],
    enum_nodes=[],
    comment_node_types=["comment"],
    name_node_type="identifier",
    import_pattern=r"""require\s+['"]([^'"]+)['"]""",
    package_pattern="",
)

KOTLIN_PROFILE = LanguageProfile(
    name="kotlin",
    extensions=[".kt", ".kts"],
    pip_package="tree-sitter-kotlin",
    module_name="tree_sitter_kotlin",
    class_nodes=["class_declaration", "object_declaration"],
    function_nodes=["function_declaration"],
    method_nodes=[],
    field_nodes=["property_declaration"],
    constructor_nodes=["primary_constructor"],
    interface_nodes=["class_declaration"],  # with 'interface' modifier
    enum_nodes=["class_declaration"],  # with 'enum' modifier
    decorator_node_types=["annotation"],
    comment_node_types=["line_comment", "multiline_comment"],
    name_node_type="simple_identifier",
    import_pattern=r"^\s*import\s+([\w.]+)",
    package_pattern=r"^\s*package\s+([\w.]+)",
)

SCALA_PROFILE = LanguageProfile(
    name="scala",
    extensions=[".scala", ".sc"],
    pip_package="tree-sitter-scala",
    module_name="tree_sitter_scala",
    class_nodes=["class_definition", "object_definition"],
    function_nodes=["function_definition"],
    method_nodes=[],
    field_nodes=["val_definition", "var_definition"],
    constructor_nodes=[],
    interface_nodes=["trait_definition"],
    enum_nodes=[],
    comment_node_types=["comment", "block_comment"],
    name_node_type="identifier",
    import_pattern=r"^\s*import\s+([\w.]+)",
    package_pattern=r"^\s*package\s+([\w.]+)",
)

# Java is always available (hard dependency)
JAVA_PROFILE = LanguageProfile(
    name="java",
    extensions=[".java"],
    pip_package="tree-sitter-java",
    module_name="tree_sitter_java",
    class_nodes=["class_declaration"],
    function_nodes=[],
    method_nodes=["method_declaration"],
    field_nodes=["field_declaration"],
    constructor_nodes=["constructor_declaration"],
    interface_nodes=["interface_declaration"],
    enum_nodes=["enum_declaration"],
    decorator_node_types=["marker_annotation", "annotation"],
    comment_node_types=["line_comment", "block_comment"],
    name_node_type="identifier",
    import_pattern=r"^\s*import\s+(?:static\s+)?([\w.]+);",
    package_pattern=r"^\s*package\s+([\w.]+)\s*;",
)

# All profiles in load order
ALL_PROFILES = [
    JAVA_PROFILE, PYTHON_PROFILE, TYPESCRIPT_PROFILE, TSX_PROFILE,
    JAVASCRIPT_PROFILE, GO_PROFILE, CSHARP_PROFILE, RUST_PROFILE,
    C_PROFILE, CPP_PROFILE, RUBY_PROFILE, KOTLIN_PROFILE, SCALA_PROFILE,
]


class LanguageRegistry:
    """Auto-detects languages and lazy-loads tree-sitter grammars.

    Usage:
        registry = LanguageRegistry()
        parser, profile = registry.get_parser(".py")
        if parser:
            tree = parser.parse(source.encode("utf-8"))
    """

    def __init__(self, auto_install: bool = False):
        self.auto_install = auto_install
        self._ext_map: dict[str, LanguageProfile] = {}
        self._parsers: dict[str, Parser | None] = {}
        self._load_failed: set[str] = set()

        # Build extension → profile map
        for profile in ALL_PROFILES:
            for ext in profile.extensions:
                self._ext_map[ext] = profile

    def get_profile(self, file_path: str) -> LanguageProfile | None:
        """Get the language profile for a file, or None if unsupported."""
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        return self._ext_map.get(ext)

    def get_parser(
        self, file_path: str,
    ) -> tuple[Parser | None, LanguageProfile | None]:
        """Get a parser + profile for a file. Returns (None, None) if
        the language is not supported or grammar can't be loaded."""
        profile = self.get_profile(file_path)
        if profile is None:
            return None, None

        lang_name = profile.name
        if lang_name in self._load_failed:
            return None, profile

        if lang_name not in self._parsers:
            self._parsers[lang_name] = self._load_parser(profile)

        parser = self._parsers.get(lang_name)
        return parser, profile

    def _load_parser(self, profile: LanguageProfile) -> Parser | None:
        """Load a tree-sitter parser for a language profile."""
        try:
            mod = importlib.import_module(profile.module_name)
            lang_func = getattr(mod, profile.language_func)
            lang = Language(lang_func())
            parser = Parser(lang)
            logger.info(f"Loaded tree-sitter grammar: {profile.name}")
            return parser
        except ImportError:
            if self.auto_install:
                return self._install_and_load(profile)
            logger.info(
                f"tree-sitter grammar for {profile.name} not installed. "
                f"Install with: pip install {profile.pip_package}"
            )
            self._load_failed.add(profile.name)
            return None
        except Exception as e:
            logger.warning(
                f"Failed to load tree-sitter grammar for "
                f"{profile.name}: {e}"
            )
            self._load_failed.add(profile.name)
            return None

    def _install_and_load(
        self, profile: LanguageProfile,
    ) -> Parser | None:
        """Auto-install a tree-sitter grammar package and load it."""
        logger.info(
            f"Auto-installing tree-sitter grammar: {profile.pip_package}"
        )
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install",
                 profile.pip_package, "--quiet"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            mod = importlib.import_module(profile.module_name)
            lang_func = getattr(mod, profile.language_func)
            lang = Language(lang_func())
            parser = Parser(lang)
            logger.info(
                f"Installed and loaded tree-sitter grammar: {profile.name}"
            )
            return parser
        except Exception as e:
            logger.warning(
                f"Failed to install {profile.pip_package}: {e}"
            )
            self._load_failed.add(profile.name)
            return None

    def supported_extensions(self) -> list[str]:
        """Return all file extensions that have a language profile."""
        return sorted(self._ext_map.keys())

    def available_languages(self) -> list[str]:
        """Return names of languages whose grammars are installed."""
        available = []
        for profile in ALL_PROFILES:
            try:
                importlib.import_module(profile.module_name)
                available.append(profile.name)
            except ImportError:
                pass
        return available

    def status(self) -> dict[str, str]:
        """Return install status for all languages."""
        result = {}
        for profile in ALL_PROFILES:
            try:
                importlib.import_module(profile.module_name)
                result[profile.name] = "installed"
            except ImportError:
                result[profile.name] = "not installed"
        return result


# Module-level singleton for shared use
_registry: LanguageRegistry | None = None


def get_registry(auto_install: bool = False) -> LanguageRegistry:
    """Get or create the global language registry."""
    global _registry
    if _registry is None:
        _registry = LanguageRegistry(auto_install=auto_install)
    return _registry
