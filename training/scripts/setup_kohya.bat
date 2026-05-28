@echo off
echo GEGEAI2.0 — kohya_ss Setup
echo ===========================

set PROJECT_ROOT=C:\Users\Vishu\Desktop\lock-in-jjk
set KOHYA_DIR=%PROJECT_ROOT%\training\kohya_ss
set VENV=%PROJECT_ROOT%\training\kohya_venv310
set PY310=C:\Users\Vishu\AI\StabilityMatrix\Data\Assets\Python310\python.exe

echo.
echo IMPORTANT: kohya_ss v25.x requires Python 3.10 (not 3.11 or 3.12)
echo Using StabilityMatrix Python 3.10 at: %PY310%
echo.

echo Step 1: Cloning kohya_ss...
if exist "%KOHYA_DIR%" (
    echo kohya_ss already exists at %KOHYA_DIR%
    echo Pulling latest...
    cd /d "%KOHYA_DIR%"
    git pull
) else (
    git clone https://github.com/bmaltais/kohya_ss.git "%KOHYA_DIR%"
    cd /d "%KOHYA_DIR%"
)

echo.
echo Step 2: Creating virtual environment (inheriting system torch/xformers)...
python -m venv venv --system-site-packages
if errorlevel 1 (
    echo ERROR: venv creation failed
    pause
    exit /b 1
)

echo.
echo Step 3: Installing kohya_ss requirements...
call venv\Scripts\activate

pip install --upgrade pip

REM Install kohya requirements but skip torch (already installed in system)
pip install --no-deps -r requirements.txt 2>nul
pip install -r requirements.txt 2>nul

REM Windows-specific requirements
if exist requirements_windows.txt (
    pip install -r requirements_windows.txt 2>nul
)

REM Ensure critical packages
pip install tensorboard
pip install prodigyopt
pip install schedulefree

echo.
echo Step 4: Verifying installation...
python -c "import torch; print('torch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
python -c "import xformers; print('xformers:', xformers.__version__)"
python -c "import accelerate; print('accelerate:', accelerate.__version__)"

echo.
echo Step 5: Configure accelerate (answer: NO machine cluster, 1 GPU, bf16)
accelerate config

echo.
echo Setup complete!
echo.
echo To train BW LoRA:
echo   cd %KOHYA_DIR%
echo   venv\Scripts\activate
echo   accelerate launch --num_cpu_threads_per_process=4 train_network.py --config_file "%PROJECT_ROOT%\training\configs\bw_lora_config.toml"
echo.
pause
