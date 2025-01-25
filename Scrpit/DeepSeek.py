from lib.gguf_model import llama_gguf
import re
import spaces
import torch
import gradio as gr
from typing import Generator
import gc
import os
from lib.basic import check_vision_support, parse_arguments, resize_image
from lib.basic import LATEX_DELIMITERS_SET, SYSTEM_ROLE, MODEL_PATH_TEMPLATE, MAX_IMAGE_SIZE
from colorama import Fore, Style

MODEL_USE = ""
PATH_PREFIX = ""

gguf = llama_gguf()

def remove_think_content(history):
    #pattern = r'\"think\".*?\"/think\"'
	pattern = r'\"think\"\s*.*?\s*\"/think\"\s*'
	for entry in history:
		if entry.get('role') == 'assistant' and 'content' in entry:
			entry['content'] = re.sub(pattern, '', entry['content'], flags=re.DOTALL)
	return history

def create_convo(system_role, history, prompt, input_file, input_use_history, input_debug_log): 
	convo = [] 
	
	file_date, file_type=gguf.process_file(input_file)	
	#print("{}history = {}{}".format(Fore.LIGHTGREEN_EX, history, Style.RESET_ALL))	 
	cleaned_history = remove_think_content(history)
	#print("{}cleaned_history = {}{}".format(Fore.LIGHTGREEN_EX, cleaned_history, Style.RESET_ALL))
	convo = gguf.create_convo(system_role, cleaned_history, prompt, input_use_history, file_date, file_type)
	
	if input_debug_log:
		print("{}convo = {}{}".format(Fore.LIGHTGREEN_EX, convo, Style.RESET_ALL))
	
	return convo
		 
@spaces.GPU()
@torch.no_grad()
def generate_description(message: dict, history, system_role, input_use_history, input_temperature, input_top_k, input_top_p, input_repetition_penalty, input_max_new_tokens, input_debug_log) -> Generator[str, None, None]:	
	prompt = message['text'].strip()
	if input_debug_log:
		print("{}prompt = {}{}".format(Fore.LIGHTGREEN_EX, prompt, Style.RESET_ALL))
 
	# Create conversation
	input_file = None
	if len(message["files"]) > 0:
		input_file = message["files"][0]
	convo = create_convo(system_role, history, prompt, input_file, input_use_history, input_debug_log)
 
	streamer = gguf.do_chat(
			convo=convo,
			input_temperature=input_temperature, 
			input_top_k=input_top_k, 
			input_top_p=input_top_p, 
			input_repetition_penalty=input_repetition_penalty, 
			input_max_new_tokens=input_max_new_tokens,
			)	
	
	outputs = ""
	for msg in streamer:
		message = msg['choices'][0]['delta']
		if 'content' in message:
			outputs += message['content'].replace("<", "\"").replace(">", "\"")
			yield outputs				
		
	print("{}output_text = {}{}".format(Fore.LIGHTGREEN_EX, outputs, Style.RESET_ALL))
  
	gc.collect()
	torch.cuda.empty_cache()

if __name__ == "__main__":	
	PATH_TEMPLATE = os.path.splitext(os.path.basename(__file__))[0]
	PATH_PREFIX, MODEL_USE, N_THREADS, N_THREADS_BATCH, N_GPU_LAYERS, N_CTX, VERBOSE, _, FINETUNE_PATH, LORA_PATH, LORA_SCALE, TITLE = parse_arguments(PATH_TEMPLATE)
	vision_model = check_vision_support(MODEL_USE)
 
	gguf.load_model(prefix=PATH_PREFIX, model_name=MODEL_USE, n_threads=N_THREADS, n_threads_batch=N_THREADS_BATCH, n_gpu_layers=N_GPU_LAYERS, n_ctx=N_CTX, verbose=VERBOSE, lora_path=LORA_PATH, lora_scale=LORA_SCALE)
		
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
 
	textbox = None
	if not vision_model:
		textbox = gr.MultimodalTextbox(file_types=["text", ".json"], file_count="single", max_lines=200)
	else:
		textbox = gr.MultimodalTextbox(file_types=["image", "text", ".json"], file_count="single", max_lines=200)
		
	with gr.Blocks() as demo:		 	
		chat_interface = gr.ChatInterface(
			fn=generate_description,
			chatbot=chatbot,
			type="messages",
			fill_height=True,
			multimodal=True,
			textbox=textbox,
			additional_inputs_accordion=gr.Accordion(label="⚙️ Parameters", open=True, render=False),
			additional_inputs=[
				gr.Dropdown(label="system role", choices=SYSTEM_ROLE, value=SYSTEM_ROLE[0], allow_custom_value=True, render=False),
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
							step=0.05,
							value=1.3,
							label="Repetition penalty",
							render=False),
				gr.Slider(minimum=256, 
							maximum=16384,
							step=256,
							value=2048, 
							label="Max new tokens", 
							render=False ),
				gr.Checkbox(label="Show prompt in log window", value=True, render=False),	
			],
		)
  
		gr.HTML(TITLE)

  
	demo.launch()
