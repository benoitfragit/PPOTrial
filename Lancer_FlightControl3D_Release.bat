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
if not exist "%~dp0logs" mkdir "%~dp0logs"
set "FLIGHT_LOG=%~dp0logs\FlightControl3D_Release.log"
echo [%date% %time%] FlightControl3D Release >>"%FLIGHT_LOG%"
"%FLIGHT_EXECUTABLE%" %* >>"%FLIGHT_LOG%" 2>&1
set "FLIGHT_EXIT_CODE=%errorlevel%"
if not "%FLIGHT_EXIT_CODE%"=="0" (
    type "%FLIGHT_LOG%"
    echo Exit code %FLIGHT_EXIT_CODE%. Log: "%FLIGHT_LOG%"
    pause
)
exit /b %FLIGHT_EXIT_CODE%
