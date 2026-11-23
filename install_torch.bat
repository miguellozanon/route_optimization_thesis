@echo off
rem Double-click to set up BRLO (Python venv + PyTorch + tests). Log: install_log.txt
cd /d "%~dp0"
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0install_torch.ps1"
echo.
echo Proceso terminado. Revisa install_log.txt. Puedes cerrar esta ventana.
pause
