"""JDT Language Server integration for semantic code intelligence.

Provides call hierarchy, find references, and type resolution by managing
a JDT LS process and communicating via LSP (Language Server Protocol).

The JDT LS JAR is auto-downloaded on first use (~50MB).
"""

import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import urllib.request
import zipfile
from typing import Optional

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".jigyasa-mcp")
JDT_DIR = os.path.join(CONFIG_DIR, "jdt-ls")
JDT_WORKSPACE = os.path.join(CONFIG_DIR, "jdt-workspace")
JDT_DOWNLOAD_URL = (
    "https://download.eclipse.org/jdtls/milestones/1.40.0/"
    "jdt-language-server-1.40.0-202409261450.tar.gz"
)

_lsp_proc: Optional[subprocess.Popen] = None
_lsp_lock = threading.Lock()
_request_id = 0


def _next_id() -> int:
    global _request_id
    _request_id += 1
    return _request_id


def _ensure_jdt_installed() -> str:
    """Download and extract JDT LS if not present. Returns path to launcher JAR."""
    launcher_pattern = "org.eclipse.equinox.launcher_"
    plugins_dir = os.path.join(JDT_DIR, "plugins")

    if os.path.isdir(plugins_dir):
        for f in os.listdir(plugins_dir):
            if f.startswith(launcher_pattern) and f.endswith(".jar"):
                return os.path.join(plugins_dir, f)

    logger.info("JDT Language Server not found. Downloading (~50MB)...")
    os.makedirs(JDT_DIR, exist_ok=True)

    archive_path = os.path.join(JDT_DIR, "jdt-ls.tar.gz")
    urllib.request.urlretrieve(JDT_DOWNLOAD_URL, archive_path)

    # Extract
    import tarfile
    with tarfile.open(archive_path, "r:gz") as tar:
        tar.extractall(JDT_DIR)
    os.remove(archive_path)

    # Find launcher
    for f in os.listdir(plugins_dir):
        if f.startswith(launcher_pattern) and f.endswith(".jar"):
            logger.info(f"JDT LS installed at {JDT_DIR}")
            return os.path.join(plugins_dir, f)

    raise RuntimeError("JDT LS extraction failed — launcher JAR not found")


def _get_config_dir() -> str:
    """Get platform-specific JDT config directory."""
    if sys.platform == "win32":
        return os.path.join(JDT_DIR, "config_win")
    elif sys.platform == "darwin":
        return os.path.join(JDT_DIR, "config_mac")
    else:
        return os.path.join(JDT_DIR, "config_linux")


class LspClient:
    """Minimal LSP client for JDT Language Server.

    Communicates via stdin/stdout using JSON-RPC 2.0 over the LSP base protocol.
    """

    def __init__(self, proc: subprocess.Popen):
        self.proc = proc
        self._lock = threading.Lock()

    def _send(self, method: str, params: dict) -> dict:
        """Send an LSP request and wait for response."""
        req_id = _next_id()
        body = json.dumps({
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params,
        })
        header = f"Content-Length: {len(body)}\r\n\r\n"
        message = header + body

        with self._lock:
            self.proc.stdin.write(message.encode("utf-8"))
            self.proc.stdin.flush()
            return self._read_response(req_id)

    def _read_response(self, expected_id: int, timeout: float = 30.0) -> dict:
        """Read LSP response matching expected_id."""
        deadline = time.time() + timeout
        buffer = b""

        while time.time() < deadline:
            # Read header
            line = self.proc.stdout.readline()
            if not line:
                break
            buffer += line

            if b"\r\n\r\n" in buffer:
                header_part, rest = buffer.split(b"\r\n\r\n", 1)
                headers = header_part.decode("utf-8")
                content_length = 0
                for h in headers.split("\r\n"):
                    if h.lower().startswith("content-length:"):
                        content_length = int(h.split(":")[1].strip())

                # Read body
                body = rest
                while len(body) < content_length:
                    chunk = self.proc.stdout.read(content_length - len(body))
                    if not chunk:
                        break
                    body += chunk

                try:
                    msg = json.loads(body.decode("utf-8"))
                except json.JSONDecodeError:
                    buffer = b""
                    continue

                if msg.get("id") == expected_id:
                    return msg.get("result", {})

                # Not our response — notification or other response, skip
                buffer = b""

        return {}

    def _notify(self, method: str, params: dict):
        """Send an LSP notification (no response expected)."""
        body = json.dumps({
            "jsonrpc": "2.0",
            "method": method,
            "params": params,
        })
        header = f"Content-Length: {len(body)}\r\n\r\n"
        self.proc.stdin.write((header + body).encode("utf-8"))
        self.proc.stdin.flush()

    def initialize(self, root_uri: str) -> dict:
        return self._send("initialize", {
            "processId": os.getpid(),
            "rootUri": root_uri,
            "capabilities": {
                "textDocument": {
                    "callHierarchy": {"dynamicRegistration": False},
                    "references": {"dynamicRegistration": False},
                }
            },
            "workspaceFolders": [{"uri": root_uri, "name": "workspace"}],
        })

    def initialized(self):
        self._notify("initialized", {})

    def shutdown(self):
        try:
            self._send("shutdown", {})
            self._notify("exit", {})
        except Exception:
            pass


class JdtLanguageServer:
    """Manages JDT LS lifecycle and provides semantic queries."""

    def __init__(self, repo_root: str):
        self.repo_root = os.path.abspath(repo_root)
        self.root_uri = "file:///" + self.repo_root.replace("\\", "/").lstrip("/")
        self._client: Optional[LspClient] = None
        self._proc: Optional[subprocess.Popen] = None

    def start(self) -> bool:
        """Start JDT LS process."""
        try:
            launcher = _ensure_jdt_installed()
        except Exception as e:
            logger.error(f"Failed to install JDT LS: {e}")
            return False

        java = shutil.which("java")
        if not java:
            logger.error("Java not found — needed for JDT LS")
            return False

        os.makedirs(JDT_WORKSPACE, exist_ok=True)
        config_dir = _get_config_dir()

        cmd = [
            java,
            "-Declipse.application=org.eclipse.jdt.ls.core.id1",
            "-Dosgi.bundles.defaultStartLevel=4",
            "-Declipse.product=org.eclipse.jdt.ls.core.product",
            "-Xms256m", "-Xmx512m",
            "-jar", launcher,
            "-configuration", config_dir,
            "-data", JDT_WORKSPACE,
        ]

        logger.info("Starting JDT Language Server...")
        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
        self._client = LspClient(self._proc)

        # Initialize
        result = self._client.initialize(self.root_uri)
        self._client.initialized()
        logger.info("JDT LS initialized")
        return True

    def stop(self):
        if self._client:
            self._client.shutdown()
        if self._proc:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:
                self._proc.kill()
        self._client = None
        self._proc = None

    def is_running(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def find_references(self, file_path: str, line: int, character: int) -> list[dict]:
        """Find all references to the symbol at the given position."""
        if not self._client:
            return []

        uri = "file:///" + os.path.join(self.repo_root, file_path).replace("\\", "/").lstrip("/")
        result = self._client._send("textDocument/references", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character},
            "context": {"includeDeclaration": True},
        })

        if not isinstance(result, list):
            return []

        refs = []
        for loc in result:
            ref_uri = loc.get("uri", "")
            ref_range = loc.get("range", {})
            ref_start = ref_range.get("start", {})
            # Convert URI back to relative path
            ref_path = ref_uri.replace("file:///", "").replace(
                self.repo_root.replace("\\", "/").lstrip("/") + "/", ""
            )
            refs.append({
                "file_path": ref_path,
                "line": ref_start.get("line", 0) + 1,
                "character": ref_start.get("character", 0),
            })
        return refs

    def call_hierarchy_incoming(self, file_path: str, line: int, character: int) -> list[dict]:
        """Find what calls the symbol at the given position."""
        if not self._client:
            return []

        uri = "file:///" + os.path.join(self.repo_root, file_path).replace("\\", "/").lstrip("/")

        # Step 1: Prepare call hierarchy item
        items = self._client._send("textDocument/prepareCallHierarchy", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character},
        })

        if not items or not isinstance(items, list):
            return []

        # Step 2: Get incoming calls
        incoming = self._client._send("callHierarchy/incomingCalls", {
            "item": items[0],
        })

        if not isinstance(incoming, list):
            return []

        calls = []
        for call in incoming:
            from_item = call.get("from", {})
            from_uri = from_item.get("uri", "")
            from_range = from_item.get("range", {}).get("start", {})
            from_path = from_uri.replace("file:///", "").replace(
                self.repo_root.replace("\\", "/").lstrip("/") + "/", ""
            )
            calls.append({
                "name": from_item.get("name", "?"),
                "kind": from_item.get("kind", 0),
                "file_path": from_path,
                "line": from_range.get("line", 0) + 1,
            })
        return calls

    def call_hierarchy_outgoing(self, file_path: str, line: int, character: int) -> list[dict]:
        """Find what the symbol at the given position calls."""
        if not self._client:
            return []

        uri = "file:///" + os.path.join(self.repo_root, file_path).replace("\\", "/").lstrip("/")

        items = self._client._send("textDocument/prepareCallHierarchy", {
            "textDocument": {"uri": uri},
            "position": {"line": line - 1, "character": character},
        })

        if not items or not isinstance(items, list):
            return []

        outgoing = self._client._send("callHierarchy/outgoingCalls", {
            "item": items[0],
        })

        if not isinstance(outgoing, list):
            return []

        calls = []
        for call in outgoing:
            to_item = call.get("to", {})
            to_uri = to_item.get("uri", "")
            to_range = to_item.get("range", {}).get("start", {})
            to_path = to_uri.replace("file:///", "").replace(
                self.repo_root.replace("\\", "/").lstrip("/") + "/", ""
            )
            calls.append({
                "name": to_item.get("name", "?"),
                "kind": to_item.get("kind", 0),
                "file_path": to_path,
                "line": to_range.get("line", 0) + 1,
            })
        return calls
