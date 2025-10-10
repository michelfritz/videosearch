@echo off
chcp 65001 >NUL
set PYTHONIOENCODING=utf-8
set NO_EMOJI=0
set PYEXE=C:\Transcript\venv_streamlit\Scripts\python.exe
set SCRIPT=C:\Transcript\fusionner_et_vectoriser.py
"%PYEXE%" "%SCRIPT%"
echo.
pause
