@echo off
call .\venv\Scripts\activate.bat

@set GRADIO_SERVER_PORT=57861
cd Scrpit
py -m Meta-Llama -v -g -n 34 -c 12800 -t 16 -tb 16 -nmm True "GGUF_Llama-3.3-70B-Inst-Q5_K_M"
pause
