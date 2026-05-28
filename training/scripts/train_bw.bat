@echo off
REM JJK Colorizer — Launch BW LoRA Training (SDXL)
REM Prerequisites:
REM   1. kohya_venv310 created with Python 3.10.11
REM   2. PyTorch 2.3.1+cu121, xformers installed in venv
REM   3. sd-scripts library installed (pip install -e training\kohya_ss\sd-scripts\)
REM   4. Training deps installed (accelerate, bitsandbytes, diffusers, etc.)
REM   5. accelerate configured for bf16 in kohya_venv310
REM   6. BW dataset prepared (300+ images in bw_gege folder)
REM   7. Illustrious XL at correct Forge path

set PROJECT_ROOT=C:\Users\Vishu\Desktop\lock-in-jjk
set SD_SCRIPTS=%PROJECT_ROOT%\training\kohya_ss\sd-scripts
set VENV=%PROJECT_ROOT%\training\kohya_venv310
set CONFIG=%PROJECT_ROOT%\training\configs\bw_lora_config.toml
set ACCELERATE=%VENV%\Scripts\accelerate.exe
set DATASET_DIR=%PROJECT_ROOT%\dataset\bw_gege\10_gegeakutami, monochrome manga

echo.
echo JJK Colorizer -- BW LoRA Training (SDXL)
echo ======================================
echo Script: sdxl_train_network.py
echo Config: %CONFIG%
echo Output: %PROJECT_ROOT%\training\output\bw_lora\
echo.

REM Verify venv accelerate exists
if not exist "%ACCELERATE%" (
    echo ERROR: accelerate not found at %ACCELERATE%
    echo Run install_kohya_deps.bat first.
    pause
    exit /b 1
)

REM Verify dataset folder exists
if not exist "%DATASET_DIR%" (
    echo ERROR: BW dataset folder not found
    echo Expected: %DATASET_DIR%
    pause
    exit /b 1
)

REM Count images using PowerShell (handles comma in path correctly)
for /f %%c in ('powershell -NoProfile -Command "(Get-ChildItem -Path '%PROJECT_ROOT%\dataset\bw_gege\10_gegeakutami, monochrome manga' -Filter '*.png').Count"') do set count=%%c
echo Images found: %count%
if %count% LSS 100 (
    echo WARNING: Less than 100 images. Training quality will be low.
    echo Recommend: 300+ high-quality manga panel crops from tankobon scans.
    echo Press any key to continue anyway, or Ctrl+C to abort.
    pause
)

echo.
echo Starting training...
echo Monitor with: %VENV%\Scripts\tensorboard.exe --logdir "%PROJECT_ROOT%\training\logs\bw_lora"
echo.

cd /d "%SD_SCRIPTS%"

"%ACCELERATE%" launch --num_cpu_threads_per_process=4 sdxl_train_network.py ^
  --config_file "%CONFIG%"

echo.
echo Training complete. Check output at: %PROJECT_ROOT%\training\output\bw_lora\
echo Run evaluate_checkpoint.py to compare checkpoints.
pause
