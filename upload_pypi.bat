@echo off
REM upload_pypi.bat - Build and upload CopyTalker to PyPI (Windows)
REM
REM Usage:
REM   upload_pypi.bat          - Upload to PyPI (production)
REM   upload_pypi.bat test     - Upload to TestPyPI first
REM   upload_pypi.bat build    - Build only, no upload
REM

setlocal enabledelayedexpansion
cd /d "%~dp0"

set "MODE=%~1"

REM ── Step 1: Check prerequisites ──────────────────────────────────────────
echo [INFO] Checking prerequisites...

where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] python not found. Install Python 3.9+ from https://www.python.org/
    exit /b 1
)

python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] pip not found.
    exit /b 1
)

REM Install build tools if missing
python -m build --help >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing build...
    python -m pip install --upgrade build
)

python -m twine --help >nul 2>&1
if %errorlevel% neq 0 (
    echo [INFO] Installing twine...
    python -m pip install --upgrade twine
)

REM ── Step 2: Read version from __init__.py (single source of truth) ──────────
echo [INFO] Reading version...

for /f "delims=" %%v in ('python -c "import re; f=open('src/copytalker/__init__.py').read(); m=re.search(r'__version__\s*=\s*\"(.+?)\"',f); print(m.group(1) if m else 'UNKNOWN')"') do set "VERSION=%%v"

if "%VERSION%"=="UNKNOWN" (
    echo [ERROR] Could not read version from src\copytalker\__init__.py
    exit /b 1
)

echo [INFO] Building version: %VERSION%

REM ── Step 3: Clean previous builds ──────────────────────────────────────────
echo [INFO] Cleaning previous builds...
if exist build rmdir /s /q build
if exist dist rmdir /s /q dist
for /d %%d in (src\*.egg-info) do rmdir /s /q "%%d"
for /d %%d in (*.egg-info) do rmdir /s /q "%%d"

REM ── Step 4: Build ──────────────────────────────────────────────────────────
echo [INFO] Building sdist and wheel...
python -m build
if %errorlevel% neq 0 (
    echo [ERROR] Build failed!
    exit /b 1
)

echo [INFO] Build artifacts:
dir /b dist\

REM ── Step 5: Check package ──────────────────────────────────────────────────
echo [INFO] Running twine check...
python -m twine check dist\*
if %errorlevel% neq 0 (
    echo [ERROR] twine check failed!
    exit /b 1
)

REM ── Step 6: Upload ─────────────────────────────────────────────────────────
if "%MODE%"=="build" (
    echo [INFO] Build-only mode. Skipping upload.
    echo [INFO] Artifacts are in dist\
    exit /b 0
)

if "%MODE%"=="test" (
    echo [INFO] Uploading to TestPyPI...
    echo [WARN] Make sure you have a TestPyPI account: https://test.pypi.org/account/register/
    echo.
    python -m twine upload --repository testpypi dist\*
    if %errorlevel% neq 0 (
        echo [ERROR] TestPyPI upload failed!
        exit /b 1
    )
    echo.
    echo [INFO] Uploaded to TestPyPI!
    echo [INFO] Install with: pip install -i https://test.pypi.org/simple/ copytalker==%VERSION%
    echo.
    set /p "CONFIRM=Upload to production PyPI as well? [y/N] "
    if /i not "!CONFIRM!"=="y" (
        echo [INFO] Skipped production upload.
        exit /b 0
    )
)

echo [INFO] Uploading to PyPI...
echo [WARN] Make sure you have a PyPI account: https://pypi.org/account/register/
echo [WARN] Use API token: https://pypi.org/manage/account/token/
echo.
python -m twine upload dist\*
if %errorlevel% neq 0 (
    echo [ERROR] PyPI upload failed!
    exit /b 1
)

echo.
echo [INFO] Successfully uploaded copytalker %VERSION% to PyPI!
echo [INFO] Install with: pip install copytalker==%VERSION%
