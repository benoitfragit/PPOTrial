@echo off
setlocal
cd /d "%~dp0"
where py >nul 2>nul
if errorlevel 1 (
  echo Le lanceur Python Windows "py" est introuvable.
  echo Installez Python 3 avec Tkinter puis relancez ce fichier.
  pause
  exit /b 1
)
py -3 app.py
if errorlevel 1 (
  echo.
  echo Le diagnostic s'est termine avec une erreur.
  pause
)
endlocal
