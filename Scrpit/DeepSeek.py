from lib.gguf_model import llama_gguf
import spaces
import torch
import gradio as gr
from typing import Generator
import gc
import os
import base64
from lib.basic import check_vision_support, parse_arguments
from lib.basic import LATEX_DELIMITERS_SET, SYSTEM_ROLE
from lib.comfyui import comfyui, extract_comfyui_content
from colorama import Fore, Style

MODEL_USE = ""
PATH_PREFIX = ""

gguf = llama_gguf()

cancel_event = None

def remove_think_content(history):
	history = fix_think_content(history)
    
	new_history = []
	for entry in history:
		if entry.get('role') in ['assistant', 'system'] and 'content' in entry:
			entry['content'] = entry['content'].split("<p>")[0]
			entry['content'] = entry['content'].split("</think>")[-1]
		new_history.append(entry)
	return new_history

def fix_think_content(history):
	for entry in history:
		if entry.get('role') in ['assistant', 'system'] and 'content' in entry:
			entry['content'] = entry['content'].split("<p>")[0]
			entry['content'] = entry['content'].replace("\"think\"","<think>").replace("\"/think\"","</think>")			
	return history

def create_convo(system_role, history, prompt, files, input_use_history, input_debug_log, input_remove_think, input_no_system_prompt): 
	
	if input_remove_think:
		history = remove_think_content(history)
	else:
 		#Fix <think> & </think>
		history = fix_think_content(history)
	
	files_list=gguf.process_files(files, 0)
	convo = gguf.create_convo(system_role, history, prompt, input_use_history, files_list, no_system_prompt=input_no_system_prompt)
 	
	if input_debug_log:
		print("{}convo = {}{}".format(Fore.LIGHTGREEN_EX, convo, Style.RESET_ALL))
	
	return convo
	 
@spaces.GPU()
@torch.no_grad()
def generate_description(message: dict, history, btn_cancel, system_role, input_use_history, input_temperature, input_top_k, input_top_p, input_repetition_penalty, input_max_new_tokens, input_debug_log, input_remove_think, input_no_system_prompt) -> Generator[str, None, None]:	
	prompt = message['text'].strip()
	if input_debug_log:
		print("{}prompt = {}{}".format(Fore.LIGHTRED_EX, prompt, Style.RESET_ALL))
 
	convo = create_convo(system_role, history, prompt, message["files"], input_use_history, input_debug_log, input_remove_think, input_no_system_prompt)	
	cancel_event.clear()
	streamer = gguf.do_chat_ex(
			convo=convo,
			input_temperature=input_temperature, 
			input_top_k=input_top_k, 
			input_top_p=input_top_p, 
			input_repetition_penalty=input_repetition_penalty, 
			input_max_new_tokens=input_max_new_tokens,
			)	
	
	outputs = ""
	for msg in streamer:
		message = msg['choices'][0]
		if 'text' in message:
				new_token = message['text']
				if new_token != "<":
					outputs += new_token.replace("<", "\"").replace(">", "\"")
					yield outputs
		
	print("{}output_text = {}{}".format(Fore.LIGHTGREEN_EX, outputs, Style.RESET_ALL))
 
	pose_tags = extract_comfyui_content(outputs.split("\"/think\"")[-1])
	if pose_tags:
		yield outputs+"\n\nWaiting for images..."
		image_path = comfyui(pose_tags)		
  
		with open(image_path, 'rb') as f:
			binary = f.read()
		base64_encoded = base64.b64encode(binary).decode('utf-8')
		print("{}image_path = {}{}".format(Fore.LIGHTGREEN_EX, image_path, Style.RESET_ALL))
		yield outputs + "\n\n" + '<p><img src="data:image/png;base64,'+base64_encoded+'"></p>'
  
	gc.collect()
	torch.cuda.empty_cache()

def cancel_btn():	
	print("{}Cancel button clicked!{}".format(Fore.LIGHTRED_EX, Style.RESET_ALL))	
	cancel_event.set()
 
if __name__ == "__main__":	
	PATH_TEMPLATE = os.path.splitext(os.path.basename(__file__))[0]
	PATH_PREFIX, MODEL_USE, N_THREADS, N_THREADS_BATCH, N_GPU_LAYERS, N_CTX, VERBOSE, using_gguf_model, FINETUNE_PATH, LORA_PATH, LORA_SCALE, MMAP, MLOCK, _, TITLE = parse_arguments(PATH_TEMPLATE)
 
	gguf.load_model(prefix=PATH_PREFIX, model_name=MODEL_USE, n_threads=N_THREADS, n_threads_batch=N_THREADS_BATCH, n_gpu_layers=N_GPU_LAYERS, n_ctx=N_CTX, verbose=VERBOSE, lora_path=LORA_PATH, lora_scale=LORA_SCALE, use_mmap=MMAP, use_mlock=MLOCK)
	cancel_event = gguf.cancel_event
		
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
 
	textbox = gr.MultimodalTextbox(file_types=["text", ".json"], file_count="single", max_lines=200)
		 
	with gr.Blocks() as demo:		 	
		btn_cancel = gr.Button(value="Cancel", render=False)
		btn_cancel.click(fn=cancel_btn, inputs=[], outputs=[])
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
				gr.Dropdown(label="System Role", choices=SYSTEM_ROLE, value=SYSTEM_ROLE[0], allow_custom_value=True, render=False),
				gr.Checkbox(label="Enable Conversations History", value=True, render=False),	
				gr.Slider(minimum=0.1,
							maximum=1, 
							step=0.1,
							value=0.6, 
							label="Temperature", 
							render=False),
				gr.Slider(minimum=10,
							maximum=100,
							step=5,
							value=50,
							label="Top K",
	   						render=False),
				gr.Slider(minimum=0,
							maximum=1,
							step=0.01,
							value=0.9,
							label="Top p",
							render=False),
				gr.Slider(minimum=0,
							maximum=2,
							step=0.01,
							value=1,
							label="Repetition penalty",
							render=False),
				gr.Slider(minimum=256, 
							maximum=16384,
							step=256,
							value=4096, 
							label="Max new tokens", 
							render=False),
				gr.Checkbox(label="Show prompt in log window", value=True, render=False),	
    			gr.Checkbox(label="Remove <think></think> from history", value=True, render=False),	
       			gr.Checkbox(label="No System Role", value=True, render=False),	
			],
		)
  
		gr.HTML(TITLE)

	current_file_path = os.path.abspath(__file__)
	current_folder = os.path.dirname(current_file_path) + '\\Outputs'
	print("{}current_folder = {}{}".format(Fore.LIGHTGREEN_EX, current_folder, Style.RESET_ALL))
	demo.launch(
		
	)
