@echo off
REM GEGEAI2.0 — Launch Color LoRA Training (SDXL)
REM Run AFTER BW LoRA training is complete and validated.
REM Same prerequisites as train_bw.bat.

set PROJECT_ROOT=C:\Users\Vishu\Desktop\lock-in-jjk
set SD_SCRIPTS=%PROJECT_ROOT%\training\kohya_ss\sd-scripts
set VENV=%PROJECT_ROOT%\training\kohya_venv310
set CONFIG=%PROJECT_ROOT%\training\configs\color_lora_config.toml
set ACCELERATE=%VENV%\Scripts\accelerate.exe

echo.
echo GEGEAI2.0 -- Color LoRA Training (SDXL)
echo =========================================
echo Script: sdxl_train_network.py
echo Config: %CONFIG%
echo Output: %PROJECT_ROOT%\training\output\color_lora\
echo.

if not exist "%ACCELERATE%" (
    echo ERROR: accelerate not found at %ACCELERATE%
    echo Run install_kohya_deps.bat first.
    pause
    exit /b 1
)

if not exist "%PROJECT_ROOT%\dataset\color_gege\15_gegecover, painterly watercolor cover" (
    echo ERROR: Color dataset folder not found
    pause
    exit /b 1
)

cd /d "%SD_SCRIPTS%"

echo Starting training...
echo Monitor: %VENV%\Scripts\tensorboard.exe --logdir "%PROJECT_ROOT%\training\logs\color_lora"
echo.

"%ACCELERATE%" launch --num_cpu_threads_per_process=4 sdxl_train_network.py ^
  --config_file "%CONFIG%"

echo Training complete. Check: %PROJECT_ROOT%\training\output\color_lora\
pause
