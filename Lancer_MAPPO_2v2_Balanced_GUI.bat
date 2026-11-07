@echo off
setlocal
cd /d "%~dp0"

echo [LOCAL_GUI] Construction incrementale Release...
cmake --build build/release --config Release --target PPO
if errorlevel 1 goto :error

py -3.12 --version >nul 2>nul
if errorlevel 1 goto :python_error
py -3.12 tools\local_mappo_gui_curriculum.py --profile balanced_v1 --expected-source-updates 1790

:result
if errorlevel 1 goto :error
echo [LOCAL_GUI] Curriculum termine ou interrompu proprement.
pause
exit /b 0

:python_error
echo [LOCAL_GUI] Python 3.12 est requis via le lanceur Windows ^(py -3.12^).
goto :error

:error
echo [LOCAL_GUI] Echec. Consulte le message ci-dessus.
pause
exit /b 1
