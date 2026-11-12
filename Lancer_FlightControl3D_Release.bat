@echo off
setlocal
cd /d "%~dp0"
set "FLIGHT_EXECUTABLE=%~dp0build\release\Main\Release\FlightControl3D.exe"
if not exist "%FLIGHT_EXECUTABLE%" (
    echo FlightControl3D Release n'est pas encore compile.
    echo Configurez build/release puis compilez la cible FlightControl3D en Release.
    pause
    exit /b 1
)
"%FLIGHT_EXECUTABLE%" %*
set "FLIGHT_EXIT_CODE=%errorlevel%"
if not "%FLIGHT_EXIT_CODE%"=="0" pause
exit /b %FLIGHT_EXIT_CODE%
