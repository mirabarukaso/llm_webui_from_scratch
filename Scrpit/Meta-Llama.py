from lib.gguf_model import llama_gguf
from llama_cpp.llama_chat_format import Llama3VisionAlpha
from threading import Thread
import time
import spaces
import torch
import gradio as gr
from transformers import GenerationConfig, TextIteratorStreamer, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from typing import Generator
import asyncio
import gc
import os
from lib.basic import parse_arguments, MyStopCriteria, process_files
from lib.basic import LATEX_DELIMITERS_SET, SYSTEM_ROLE, MODEL_PATH_TEMPLATE
from colorama import Fore, Style
from pathlib import Path

MODEL_USE = ""
PATH_PREFIX = ""

using_gguf_model = False

model = None
processor = None
tokenizer = None

cancel_event = None
start_time = 0

gguf = llama_gguf()		  

system_prompt_prefix = ''

def load_model(prefix, model_path):	 	
	full_path = MODEL_PATH_TEMPLATE.format(prefix, model_path)
	print("{}Loading: {}{}".format(Fore.LIGHTGREEN_EX, full_path, Style.RESET_ALL))	

	model = AutoModelForCausalLM.from_pretrained(full_path, torch_dtype=torch.bfloat16, device_map="auto",)   		  	
	processor = AutoProcessor.from_pretrained(full_path)
	tokenizer = AutoTokenizer.from_pretrained(full_path)
	return model, processor, tokenizer

def create_convo(system_role, history, prompt, files, input_use_history, input_image_size): 
	global system_prompt_prefix	
	convo = [] 
 
	if not using_gguf_model:
		_, file_prompt = process_files(files, vision_model=False, input_image_size=0)
		convo = create_convo_local(system_role, history, prompt, file_prompt, input_use_history)
	else:
		files_list=gguf.process_files(files, input_image_size)
		convo = gguf.create_convo(system_role, history, prompt, use_history=input_use_history, file_list=files_list, system_prompt_prefix=system_prompt_prefix)
		
	return convo

def create_convo_local(system_role, history, prompt, file_prompt, use_history=True):
	convo = [{'role': 'system', 'content': f'{system_role}{system_prompt_prefix}'}]
  
	if len(history) > 0 and use_history:
		convo.extend(history)
	convo.append({'role': 'user', "content": f'{file_prompt}\n{prompt}'})
     
	return convo

def do_chat(messages, input_debug_log=False): 
	global start_time
	
	# Tokenize the prompt
	inputs = None
	generation_kwargs = None  
	streamer = None

	# Preparation for inference
	text = processor.apply_chat_template(
		messages, tokenize=False, add_generation_prompt=True
	)
	inputs = tokenizer(text, return_tensors="pt").to(model.device)
 
	if input_debug_log:
		print("{}convo = {}{}".format(Fore.LIGHTGREEN_EX, text, Style.RESET_ALL))    

	streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)
	generation_kwargs = dict(
		**inputs,
		streamer=streamer,
		pad_token_id=tokenizer.eos_token_id,
	)

	# Add stopping criteria
	generation_kwargs['stopping_criteria'] = [MyStopCriteria(cancel_event)]
			
	# Time counter
	start_time = time.time()

	thread = Thread(target=model.generate, kwargs=generation_kwargs)
	thread.start()

	outputs = ''
	for new_text in streamer:
		outputs +=new_text
		yield outputs

	print("{}output_text = {}{}".format(Fore.LIGHTGREEN_EX, outputs, Style.RESET_ALL))
	# Wait for the thread to finish
	thread.join()
	print("{}Time used: {} seconds{}".format(Fore.LIGHTGREEN_EX, time.time() - start_time, Style.RESET_ALL))
		 
@spaces.GPU()
@torch.no_grad()
def generate_description(message: dict, history, btn_cancel, btn_system_prefix_u, btn_system_prefix_d, system_role, input_use_history, input_temperature, input_top_k, input_top_p, input_repetition_penalty, input_max_new_tokens, input_debug_log, input_image_size) -> Generator[str, None, None]:
	if not using_gguf_model:
		configure_model(input_temperature, input_top_p, input_repetition_penalty, input_max_new_tokens)

	prompt = message['text'].strip()
	if input_debug_log:
		print("{}prompt = {}{}".format(Fore.LIGHTCYAN_EX, prompt, Style.RESET_ALL))

	convo = create_convo(system_role, history, prompt, message["files"], input_use_history, input_image_size)
	cancel_event.clear()	
	if not using_gguf_model:
		yield from do_chat(convo, input_debug_log=input_debug_log)
	else:
		yield from gguf.do_chat(
			convo=convo,
			input_temperature=input_temperature,
			input_top_k=input_top_k,
			input_top_p=input_top_p,
			input_repetition_penalty=input_repetition_penalty,
			input_max_new_tokens=input_max_new_tokens,
			input_debug_log=input_debug_log,
			stop='<|eot_id|>'
		)
		
	gc.collect()
	torch.cuda.empty_cache()

def configure_model(input_temperature, input_top_p, input_repetition_penalty, input_max_new_tokens):
	full_path = MODEL_PATH_TEMPLATE.format(PATH_PREFIX, MODEL_USE)
	config = GenerationConfig.from_pretrained(full_path, temperature=input_temperature, top_p=input_top_p, repetition_penalty=input_repetition_penalty, max_new_tokens=input_max_new_tokens)
	model.generation_config = config

def cancel_btn():	
	if cancel_event:		
		print("{}Cancel button clicked!{}".format(Fore.LIGHTRED_EX, Style.RESET_ALL))	
		cancel_event.set()			  	
		print("{}Time used: {} seconds{}".format(Fore.LIGHTGREEN_EX, time.time() - start_time, Style.RESET_ALL))

def upload_file(filepath):
    global system_prompt_prefix
    name = Path(filepath).name
    _, prefix = process_files([filepath], vision_model=False, input_image_size=0)
    system_prompt_prefix = f'\n{prefix}'
    return [gr.UploadButton(visible=False), gr.Button(f"Clear prefix {name}", value=filepath, visible=True)]

def clear_file():
    global system_prompt_prefix 
    system_prompt_prefix = ''
    return [gr.UploadButton(visible=True), gr.Button(visible=False)]

if __name__ == "__main__":	
	PATH_TEMPLATE = os.path.splitext(os.path.basename(__file__))[0]
	PATH_PREFIX, MODEL_USE, N_THREADS, N_THREADS_BATCH, N_GPU_LAYERS, N_CTX, VERBOSE, using_gguf_model, FINETUNE_PATH, LORA_PATH, LORA_SCALE, MMAP, MLOCK, CHAT_HANDLER, TITLE = parse_arguments(PATH_TEMPLATE)
	
	JSON_FILE_TYPE = ".json"
	TEXT_FILE_TYPE = "text"
	SUPPORT_FILE_TYPE = [TEXT_FILE_TYPE, JSON_FILE_TYPE]
 
	if using_gguf_model:
		if CHAT_HANDLER:
			chat_handler = Llama3VisionAlpha(clip_model_path=CHAT_HANDLER, verbose=VERBOSE)
			SUPPORT_FILE_TYPE = [TEXT_FILE_TYPE, JSON_FILE_TYPE, "image"]
		else:
			gguf.load_model(prefix=PATH_PREFIX, model_name=MODEL_USE, n_threads=N_THREADS, n_threads_batch=N_THREADS_BATCH, n_gpu_layers=N_GPU_LAYERS, n_ctx=N_CTX, verbose=VERBOSE, lora_path=LORA_PATH, lora_scale=LORA_SCALE, use_mmap=MMAP, use_mlock=MLOCK)
		cancel_event = gguf.cancel_event		
	else:
		model, processor, tokenizer = load_model(PATH_PREFIX, MODEL_USE)
		cancel_event = asyncio.Event()
	
	
	avatars_list = ['.\\Images\\avatar_user.png', '.\\Images\\avatar_system.png']
	chatbot=gr.Chatbot(
			height=1152, 
			placeholder="", 
			label=TITLE, 
			type="messages", 
			latex_delimiters=LATEX_DELIMITERS_SET,
			avatar_images=avatars_list,
   			resizeable=True,
			layout="bubble",
		)
 
	textbox = gr.MultimodalTextbox(file_types=SUPPORT_FILE_TYPE, file_count="single", max_lines=200)
		
	with gr.Blocks() as demo:		  
		btn_cancel = gr.Button(value="Cancel", render=False)
		btn_cancel.click(fn=cancel_btn, inputs=[], outputs=[])
	
		btn_system_prefix_u = gr.UploadButton("System role prefix. Upload a file", file_count="single", file_types=SUPPORT_FILE_TYPE, render=False)		
		btn_system_prefix_d = gr.Button("Clear the file", visible=False, render=False)
		btn_system_prefix_u.upload(upload_file, btn_system_prefix_u, [btn_system_prefix_u, btn_system_prefix_d])
		btn_system_prefix_d.click(clear_file, None, [btn_system_prefix_u, btn_system_prefix_d])
  
		chat_interface = gr.ChatInterface(
			fn=generate_description,
			chatbot=chatbot,
			type="messages",
			fill_height=True,
			multimodal=True,
			textbox=textbox,
			additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=True, render=False),
			additional_inputs=[
				btn_cancel,
				btn_system_prefix_u,
				btn_system_prefix_d,
				gr.Dropdown(label="System role", choices=SYSTEM_ROLE, value=SYSTEM_ROLE[0], allow_custom_value=True, render=False),
				gr.Checkbox(label="Enable Conversations History", value=True, render=False),	
				gr.Slider(minimum=0.1,
							maximum=1, 
							step=0.1,
							value=0.9, 
							label="Temperature", 
							render=False),
				gr.Slider(minimum=10,
							maximum=100,
							step=5,
							value=100,
							label="Top K",
							render=False),
				gr.Slider(minimum=0,
							maximum=1,
							step=0.01,
							value=0.98,
							label="Top p",
							render=False),
				gr.Slider(minimum=0,
							maximum=2,
							step=0.01,
							value=1.03,
							label="Repetition penalty",
							render=False),
				gr.Slider(minimum=256, 
							maximum=16384,
							step=256,
							value=4096, 
							label="Max new tokens", 
							render=False ),			
				gr.Checkbox(label="Show prompt in log window", value=True, render=False),	
				gr.Slider(minimum=128, 
							maximum=1280,
							step=64,
							value=512, 
							label="Resize Image (only for multiple images)", 
							render=False),
			],
		)
  
		gr.HTML(TITLE)

  
	demo.launch()
