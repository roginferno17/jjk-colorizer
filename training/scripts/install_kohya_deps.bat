@echo off
REM GEGEAI2.0 — Install kohya_ss dependencies into Python 3.10 venv
REM Run AFTER setup_kohya.bat and AFTER PyTorch install completes.
REM Prerequisites:
REM   - training\kohya_venv310\ exists (Python 3.10 venv)
REM   - PyTorch 2.3.1+cu121 installed in venv

set PROJECT_ROOT=C:\Users\Vishu\Desktop\lock-in-jjk
set VENV=%PROJECT_ROOT%\training\kohya_venv310
set KOHYA_DIR=%PROJECT_ROOT%\training\kohya_ss
set PIP=%VENV%\Scripts\pip.exe
set PYTHON=%VENV%\Scripts\python.exe

echo.
echo GEGEAI2.0 -- kohya_ss Dependency Install
echo ==========================================

REM Verify Python 3.10
& %PYTHON% --version
& %PYTHON% -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"

echo.
echo Step 1: Install xformers...
%PIP% install xformers==0.0.27 --index-url https://download.pytorch.org/whl/cu121 -q

echo Step 2: Install kohya_ss package (installs sd-scripts and all deps)...
cd /d "%KOHYA_DIR%"
%PIP% install -e . -q 2>nul

REM If -e . fails (Python 3.12 check), install core deps manually:
if errorlevel 1 (
    echo kohya_ss direct install failed - installing core deps manually...
    %PIP% install accelerate>=1.7.0 bitsandbytes>=0.45.0 diffusers==0.32.2 -q
    %PIP% install lycoris-lora==3.2.0.post2 safetensors==0.4.4 prodigyopt==1.1.2 -q
    %PIP% install transformers sentencepiece omegaconf -q
    %PIP% install tensorboard -q
)

echo Step 3: Configure accelerate for bf16...
%VENV%\Scripts\accelerate.exe config

echo.
echo Step 4: Verify installation...
%PYTHON% -c "import torch; print('torch:', torch.__version__)"
%PYTHON% -c "import xformers; print('xformers:', xformers.__version__)"
%PYTHON% -c "import accelerate; print('accelerate:', accelerate.__version__)"
%PYTHON% -c "import bitsandbytes; print('bitsandbytes:', bitsandbytes.__version__)"
%PYTHON% -c "import diffusers; print('diffusers:', diffusers.__version__)"

echo.
echo Installation complete!
echo.
echo To test training launch:
echo   %VENV%\Scripts\accelerate.exe launch --help
echo.
echo To find train_network.py:
echo   dir /s /b "%KOHYA_DIR%\*train_network*"
pause
