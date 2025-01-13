@echo off
call .\venv\Scripts\activate.bat

@set GRADIO_SERVER_PORT=57861
cd Scrpit
py -m aya-expanse -v -g -n 36 -c 16384 "GGUF_aya-expanse-32b"
pause
