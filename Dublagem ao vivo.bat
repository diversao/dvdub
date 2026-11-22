@echo off
chcp 65001 >nul
title Dublagem ao vivo (Chiaki) - feche esta janela para desligar
cd /d "%~dp0"
set PYTHONIOENCODING=utf-8
echo Dublagem ao vivo: ouve o Chiaki, traduz o ingles e fala em portugues.
echo   F8 = pausa / continua     F9 = abaixar o som do jogo enquanto fala (liga/desliga)
echo   Para desligar: feche esta janela.
echo.
".venv\Scripts\python.exe" dublagem.py 2>nul
pause
