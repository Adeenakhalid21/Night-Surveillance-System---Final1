@echo off
setlocal

cd /d "%~dp0"

echo ==============================================
echo Night Surveillance System - Python Setup
echo ==============================================

if not exist "requirements.txt" (
    echo [ERROR] requirements.txt not found in:
    echo %CD%
    exit /b 1
)

set "PYTHON_EXE="
set "PYTHON_ARGS="

if defined VIRTUAL_ENV if exist "%VIRTUAL_ENV%\Scripts\python.exe" (
    set "PYTHON_EXE=%VIRTUAL_ENV%\Scripts\python.exe"
)

if "%PYTHON_EXE%"=="" if exist ".venv\Scripts\python.exe" (
    set "PYTHON_EXE=.venv\Scripts\python.exe"
)

if "%PYTHON_EXE%"=="" if exist "..\.venv\Scripts\python.exe" (
    set "PYTHON_EXE=..\.venv\Scripts\python.exe"
)

if "%PYTHON_EXE%"=="" (
    where python >nul 2>nul
    if not errorlevel 1 (
        set "PYTHON_EXE=python"
    ) else (
        where py >nul 2>nul
        if not errorlevel 1 (
            set "PYTHON_EXE=py"
            set "PYTHON_ARGS=-3"
        )
    )
)

if "%PYTHON_EXE%"=="" (
    echo [ERROR] Python was not found.
    echo Install Python 3.10+ or create .venv first, then rerun this file.
    exit /b 1
)

echo [1/5] Upgrading pip tools...
"%PYTHON_EXE%" %PYTHON_ARGS% -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :install_failed

echo [2/5] Installing required libraries...
"%PYTHON_EXE%" %PYTHON_ARGS% -m pip install -r requirements.txt
if errorlevel 1 goto :install_failed

echo [3/5] Verifying super-resolution backend and downloading models...
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import enhancement; ready = enhancement.ensure_superres_models(); print('superres-backend:', enhancement.superres_backend_name()); print('superres-models-ready:', ready)"
if errorlevel 1 goto :install_failed

echo [4/5] Checking scipy health...
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import scipy; print('scipy-ok')"
if errorlevel 1 (
    echo [INFO] scipy seems broken. Reinstalling scientific stack...
    "%PYTHON_EXE%" %PYTHON_ARGS% -m pip install --upgrade --force-reinstall --no-cache-dir numpy scipy scikit-image
    if errorlevel 1 goto :install_failed
)

echo [5/5] Verifying key imports...
"%PYTHON_EXE%" %PYTHON_ARGS% -c "import flask, ultralytics, cv2, numpy, PIL, skimage, scipy, dotenv, psycopg2, matplotlib; print('All required libraries imported successfully.')"
if errorlevel 1 goto :install_failed

echo.
echo [SUCCESS] All Python libraries are installed.
echo You can now run the project.
exit /b 0

:install_failed
echo.
echo [ERROR] Library installation failed. Review the output above.
exit /b 1
