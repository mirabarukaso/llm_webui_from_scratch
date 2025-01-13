@echo off
call .\venv\Scripts\activate.bat

@set GRADIO_SERVER_PORT=57861
cd Scrpit
py -m aya-expanse "aya-expanse-8b"
pause
