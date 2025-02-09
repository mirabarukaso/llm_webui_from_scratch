from lib.gguf_model import llama_gguf
from typing import Generator
import gradio as gr
import os
import gc
import torch
from lib.basic import check_vision_support, parse_arguments
from lib.basic import LATEX_DELIMITERS_SET, SYSTEM_ROLE
from colorama import Fore, Style

cancel_event = None
gguf = llama_gguf()

def generate_description(message: dict, history, btn_cancel, system_role, input_use_history, input_temperature, input_top_k, input_top_p, input_repetition_penalty, input_max_new_tokens, input_debug_log) -> Generator[str, None, None]:
	gc.collect()
	torch.cuda.empty_cache()
 
	# Create conversation
	prompt = message['text'].strip()	
 
	files_list=gguf.process_files(message["files"], 0)
	convo = gguf.create_convo(system_role, history, prompt, use_history=input_use_history, file_list=files_list)

	yield from gguf.do_chat(
		convo=convo,
		input_temperature=input_temperature,
		input_top_k=input_top_k,
		input_top_p=input_top_p,
		input_repetition_penalty=input_repetition_penalty,
		input_max_new_tokens=input_max_new_tokens,
		input_debug_log=input_debug_log,
	)

def cancel_btn():	
	print("{}Cancel button clicked!{}".format(Fore.LIGHTRED_EX, Style.RESET_ALL))	
	cancel_event.set()
 
if __name__ == "__main__":
	PATH_TEMPLATE = os.path.splitext(os.path.basename(__file__))[0]
	PATH_PREFIX, MODEL_USE, N_THREADS, N_THREADS_BATCH, N_GPU_LAYERS, N_CTX, VERBOSE, using_gguf_model, FINETUNE_PATH, LORA_PATH, LORA_SCALE, MMAP, MLOCK, _, TITLE = parse_arguments(PATH_TEMPLATE)

	if using_gguf_model:
		gguf.load_model(prefix=PATH_PREFIX, model_name=MODEL_USE, n_threads=N_THREADS, n_threads_batch=N_THREADS_BATCH, n_gpu_layers=N_GPU_LAYERS, n_ctx=N_CTX, verbose=VERBOSE, lora_path=LORA_PATH, lora_scale=LORA_SCALE, use_mmap=MMAP, use_mlock=MLOCK)
		cancel_event = gguf.cancel_event
	else:
		error_message = "Error: Unsupported model type."
		raise RuntimeError("{}{}{}".format(Fore.LIGHTRED_EX, error_message, Style.RESET_ALL))
 
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
				gr.Dropdown(label="system role", choices=SYSTEM_ROLE, value=SYSTEM_ROLE[0], allow_custom_value=True, render=False),
				gr.Checkbox(label="Enable Conversations History", value=True, render=False),	
				gr.Slider(minimum=0.1,
							maximum=1, 
							step=0.1,
							value=0.8, 
							label="Temperature", 
							render=False),
				gr.Slider(minimum=10,
							maximum=100,
							step=5,
							value=40,
							label="Top K",
       						render=False),
				gr.Slider(minimum=0,
							maximum=1,
							step=0.05,
							value=0.95,
							label="Top p",
							render=False),
				gr.Slider(minimum=0,
							maximum=2,
							step=0.05,
							value=1.1,
							label="Repetition penalty",
							render=False),
				gr.Slider(minimum=64, 
							maximum=8192,
							step=64,
							value=2048, 
							label="Max new tokens", 
							render=False ),
				gr.Checkbox(label="Show prompt in log window", value=True, render=False),	
			],
		)	
  
	demo.launch()
