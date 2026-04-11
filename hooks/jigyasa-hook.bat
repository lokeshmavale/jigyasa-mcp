@echo off
REM Git hook: post-commit / post-merge / post-checkout (Windows)
REM Triggers incremental reindexing after git operations.
REM
REM Install:
REM   copy hooks\jigyasa-hook.bat .git\hooks\post-commit
REM   copy hooks\jigyasa-hook.bat .git\hooks\post-merge
REM   copy hooks\jigyasa-hook.bat .git\hooks\post-checkout

REM Check if jigyasa-index is available
where jigyasa-index >nul 2>&1
if errorlevel 1 exit /b 0

for /f "delims=" %%i in ('git rev-parse --show-toplevel') do set REPO_ROOT=%%i

REM Skip if another indexer is already running
if exist "%REPO_ROOT%\.jigyasa\index.lock" exit /b 0

REM Set up log directory
if not exist "%USERPROFILE%\.jigyasa-mcp" mkdir "%USERPROFILE%\.jigyasa-mcp"

REM Run indexer in background, log errors
start /b cmd /c "jigyasa-index --repo "%REPO_ROOT%" --incremental >> "%USERPROFILE%\.jigyasa-mcp\hook.log" 2>&1"
exit /b 0
