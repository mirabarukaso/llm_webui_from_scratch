# local-llm-webui
This is a (not) detailed guide for those who want to create their own local LLM service from scratch.    
------
   
# Before you start - Hardware, Software and LLM
1. A computer with at least 64GB RAM and a not too old graphics card with large VRAM. `(e.g. My 6yo build, i9-9960x,128G,Titan RTX,905P)`   
2. Keyboards, mouse or trackball and at least one monitor, bla bla bla......   
3. Windows 10/11 with Visual Studio 2077-55, python 3.11+, Nvidia CUDA toolkit `(e.g. Win10 with Start Menu on left, VS2022 C++ package with Buildtools and Win10SDK, Python 3.11, CUDA 12.7)`   
4. Recommend [HuggingFace](https://huggingface.co/) `(I know what you're looking for, but I won't talk about it.)`    

# Create VENV and build llama-cpp-python with CUDA support
1. Make sure you have installed `VS`, `NV Toolkit` and anything pip complains about missing. Google it yourself.
2. Copy following files to VS
```
(There are 4 files to copy)
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\extras\visual_studio_integration\MSBuildExtensions

(Copy them to both the x86 and x64 VS paths, you may need them in future)
C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\MSBuild\Microsoft\VC\v170\BuildCustomizations
C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations
```
3. Use `x64 Native Tools Command Prompt for VS 2022` in your Start menu
4. Double Check you have installed `CUDA Toolkits` and `NVCC`, try following commands:   
```
nvcc -V
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2024 NVIDIA Corporation
Built on Thu_Sep_12_02:55:00_Pacific_Daylight_Time_2024
Cuda compilation tools, release 12.6, V12.6.77
Build cuda_12.6.r12.6/compiler.34841621_0

nvidia-smi
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 566.36                 Driver Version: 566.36         CUDA Version: 12.7     |
|-----------------------------------------+------------------------+----------------------+
| GPU  Name                  Driver-Model | Bus-Id          Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |           Memory-Usage | GPU-Util  Compute M. |
|                                         |                        |               MIG M. |
|=========================================+========================+======================|
|   0  NVIDIA TITAN RTX             WDDM  |   00000000:65:00.0  On |                  N/A |
|  0%   38C    P0             61W /  320W |   23971MiB /  24576MiB |      0%      Default |
|                                         |                        |                  N/A |
+-----------------------------------------+------------------------+----------------------+
```
5. You may need to upgrade pip and install anything pip complains not found. Google it yourself.    
6. Clone [this repository](https://github.com/mirabarukaso/local-llm-webui/tree/main), then enter the directory.
```
git clone https://github.com/mirabarukaso/local-llm-webui.git
cd local-llm-webui
```
8. Create VENV, install requirements   
```
git clone https://github.com/mirabarukaso/local-llm-webui.git
cd local-llm-webui
python -m venv .\venv
.\venv\Scripts\activate
py -m pip install --upgrade pip
py -m pip install -r .\Scripts\requirements.txt
```
9. Compile llama-cpp-python with CUDA Support from GitHub source
```
set CMAKE_ARGS="-DGGML_CUDA=ON"
(First time?)
py -m pip install llama-cpp-python -v
(In case you messed up, use the following command)
py -m pip install "git+https://github.com/abetlen/llama-cpp-python.git" --force-reinstall --upgrade -v
(Take a nap...)

(Optional: Re-install numpy 1.26.4, you hve numpy-2.2.1 now. Solve:numpy.dtype size changed, may indicate binary incompatibility. Expected 96 from C header, got 88 from PyObject)
py -m pip install numpy==1.26.4

```
10. Quick test, both "Failed" and "Success" are valid results.
```
cd ..
cd Scripts
py -m Meta-Llama -v -g -n 0 -c 4096 "GGUF_Meta-Llama-3.1-13B-Instruct"

(Failed to load model)
INFO: This is NOT Vision model
Loading GGUF: ..\Meta-Llama\GGUF_Meta-Llama-3.1-13B\Meta-Llama-3.1-13B.gguf
| n_threads: 16 | n_threads_batch: 16 | n_gpu_layers: 33 | n_ctx: 20480 | verbose: True |
Traceback (most recent call last):
.....

(Success)
.....
* Running on local URL:  http://127.0.0.1:7860

To create a public link, set `share=True` in `launch()`.
```
11. Make sure you are using CUDA now, open "Task Manager" check your GPU tab and debug output from console
```
(llama.cpp debug info)
llm_load_tensors: offloading 33 repeating layers to GPU
llm_load_tensors: offloaded 33/81 layers to GPU
llm_load_tensors:        CUDA0 model buffer size = 16240.69 MiB
llm_load_tensors:   CPU_Mapped model buffer size = 24302.42 MiB
```
