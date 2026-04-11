"""Jigyasa server launcher — starts Jigyasa with correct parameters.

Ensures index data goes to a fixed location (~/.jigyasa-mcp/data/)
regardless of working directory.
"""

import logging
import os
import shutil
import subprocess
import sys
import time

logger = logging.getLogger(__name__)

CONFIG_DIR = os.path.join(os.path.expanduser("~"), ".jigyasa-mcp")
DATA_DIR = os.path.join(CONFIG_DIR, "data")
INDEX_DIR = os.path.join(DATA_DIR, "IndexData")
TRANSLOG_DIR = os.path.join(DATA_DIR, "TransLog")
PID_FILE = os.path.join(CONFIG_DIR, "jigyasa.pid")
JIGYASA_REPO_URL = "https://github.com/lokeshmavale/jigyasa.git"
JIGYASA_CLONE_DIR = os.path.join(CONFIG_DIR, "jigyasa")

# Known JAR locations to search
JAR_SEARCH_PATHS = [
    os.path.join(CONFIG_DIR, "jigyasa", "build", "libs"),
    os.path.join(os.path.expanduser("~"), "repos", "jigyasa", "build", "libs"),
    r"C:\azs\repos\jigyasa\build\libs",
    r"C:\engram\jigyasa\build\libs",
]


def find_jigyasa_jar() -> str:
    """Find the Jigyasa fat JAR. Auto-clones and builds if not found."""
    # Check env var first
    jar_path = os.environ.get("JIGYASA_JAR", "")
    if jar_path and os.path.exists(jar_path):
        return jar_path

    # Search known locations
    for search_dir in JAR_SEARCH_PATHS:
        if not os.path.isdir(search_dir):
            continue
        for f in os.listdir(search_dir):
            if f.endswith("-all.jar") and "Jigyasa" in f:
                return os.path.join(search_dir, f)

    # Not found — try auto-clone + build
    logger.info("Jigyasa JAR not found. Auto-cloning and building...")
    return _auto_build_jigyasa()


def is_running(port: int = 50051) -> bool:
    """Check if Jigyasa is running on the given port."""
    try:
        from jigyasa_mcp.grpc_client import JigyasaClient
        client = JigyasaClient(endpoint=f"localhost:{port}", timeout=2.0)
        health = client.health()
        client.close()
        return health["status"] == "SERVING"
    except Exception:
        return False


def _auto_build_jigyasa() -> str:
    """Clone Jigyasa from GitHub and build the fat JAR."""
    if not os.path.isdir(JIGYASA_CLONE_DIR):
        logger.info(f"Cloning {JIGYASA_REPO_URL} → {JIGYASA_CLONE_DIR}")
        result = subprocess.run(
            ["git", "clone", JIGYASA_REPO_URL, JIGYASA_CLONE_DIR],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error(f"Git clone failed: {result.stderr}")
            return ""

    # Build
    gradlew_name = "gradlew.bat" if sys.platform == "win32" else "gradlew"
    gradlew = os.path.join(JIGYASA_CLONE_DIR, gradlew_name)
    if not os.path.exists(gradlew):
        logger.error(f"gradlew not found at {gradlew}")
        return ""

    logger.info("Building Jigyasa JAR (this may take a minute)...")
    result = subprocess.run(
        [gradlew, "shadowJar", "--quiet"],
        cwd=JIGYASA_CLONE_DIR, capture_output=True, text=True,
    )
    if result.returncode != 0:
        logger.error(f"Build failed: {result.stderr[-500:]}")
        return ""

    # Find built JAR
    libs_dir = os.path.join(JIGYASA_CLONE_DIR, "build", "libs")
    if os.path.isdir(libs_dir):
        for f in os.listdir(libs_dir):
            if f.endswith("-all.jar") and "Jigyasa" in f:
                jar = os.path.join(libs_dir, f)
                logger.info(f"Built JAR: {jar}")
                return jar

    logger.error("JAR not found after build")
    return ""


def start(
    port: int = 50051,
    heap_min: str = "512m",
    heap_max: str = "1g",
    jar_path: str = "",
) -> bool:
    """Start Jigyasa server with fixed data directories.

    Returns True if server started successfully.
    """
    if is_running(port):
        logger.info(f"Jigyasa already running on port {port}")
        return True

    if not jar_path:
        jar_path = find_jigyasa_jar()
    if not jar_path:
        logger.error(
            "Cannot find Jigyasa JAR. Set JIGYASA_JAR env var or build with: "
            "cd <jigyasa-repo> && ./gradlew shadowJar"
        )
        return False

    # Ensure data directories exist
    os.makedirs(INDEX_DIR, exist_ok=True)
    os.makedirs(TRANSLOG_DIR, exist_ok=True)

    java = shutil.which("java")
    if not java:
        logger.error("Java not found in PATH. Install Java 21+.")
        return False

    cmd = [
        java,
        "--add-modules", "jdk.incubator.vector",
        f"-Xms{heap_min}", f"-Xmx{heap_max}",
        "-XX:+AlwaysPreTouch",
        "-Dlucene.useScalarFMA=true",
        "-Dlucene.useVectorFMA=true",
        "-jar", jar_path,
    ]

    env = os.environ.copy()
    env["INDEX_CACHE_DIR"] = INDEX_DIR
    env["TRANSLOG_DIRECTORY"] = TRANSLOG_DIR
    env["GRPC_SERVER_PORT"] = str(port)

    logger.info(f"Starting Jigyasa: port={port}, index={INDEX_DIR}, translog={TRANSLOG_DIR}")
    logger.info(f"JAR: {jar_path}")

    # Start detached
    if sys.platform == "win32":
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            creationflags=subprocess.DETACHED_PROCESS | subprocess.CREATE_NO_WINDOW,
        )
    else:
        proc = subprocess.Popen(
            cmd, env=env,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            start_new_session=True,
        )

    # Save PID
    with open(PID_FILE, "w") as f:
        f.write(str(proc.pid))

    # Wait for ready
    for _attempt in range(20):
        time.sleep(1)
        if is_running(port):
            logger.info(f"Jigyasa ready on port {port} (PID {proc.pid})")
            return True

    logger.error("Jigyasa failed to start within 20 seconds")
    return False


def stop():
    """Stop a running Jigyasa server using saved PID."""
    if not os.path.exists(PID_FILE):
        logger.info("No PID file found — Jigyasa may not be running")
        return

    with open(PID_FILE) as f:
        pid = int(f.read().strip())

    try:
        if sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/PID", str(pid)],
                           capture_output=True)
        else:
            import signal
            os.kill(pid, signal.SIGTERM)
        logger.info(f"Stopped Jigyasa (PID {pid})")
    except (ProcessLookupError, OSError) as e:
        logger.info(f"Jigyasa process {pid} not found: {e}")

    try:
        os.remove(PID_FILE)
    except OSError:
        pass


def status(port: int = 50051) -> dict:
    """Get Jigyasa server status."""
    running = is_running(port)
    pid = None
    if os.path.exists(PID_FILE):
        with open(PID_FILE) as f:
            pid = int(f.read().strip())

    return {
        "running": running,
        "port": port,
        "pid": pid,
        "index_dir": INDEX_DIR,
        "translog_dir": TRANSLOG_DIR,
        "index_dir_exists": os.path.isdir(INDEX_DIR),
        "index_size_mb": _dir_size_mb(INDEX_DIR),
    }


def _dir_size_mb(path: str) -> float:
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            total += os.path.getsize(os.path.join(dirpath, f))
    return round(total / (1024 * 1024), 1)
