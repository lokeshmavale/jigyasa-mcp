@echo off
REM Git hook: post-commit / post-merge / post-checkout (Windows)
REM Triggers incremental reindexing after git operations.
REM
REM Install:
REM   copy hooks\jigyasa-hook.bat .git\hooks\post-commit
REM   copy hooks\jigyasa-hook.bat .git\hooks\post-merge
REM   copy hooks\jigyasa-hook.bat .git\hooks\post-checkout

for /f "delims=" %%i in ('git rev-parse --show-toplevel') do set REPO_ROOT=%%i

REM Run indexer in background
start /b jigyasa-index --repo "%REPO_ROOT%" --incremental >nul 2>&1
