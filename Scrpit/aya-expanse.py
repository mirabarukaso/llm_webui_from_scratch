from lib.gguf_model import llama_gguf
from threading import Thread
import time
import spaces
import torch
import gradio as gr
from transformers import GenerationConfig, StoppingCriteria, TextIteratorStreamer, AutoModelForCausalLM, AutoTokenizer
from typing import Generator
import asyncio
import gc
import os
from lib.basic import parse_arguments, check_vision_support, LATEX_DELIMITERS_SET, SYSTEM_ROLE, MODEL_PATH_TEMPLATE
from colorama import Fore, Style

MAX_IMAGE_SIZE = 640
MODEL_USE = ""
PATH_PREFIX = ""

using_gguf_model = False

model = None
tokenizer = None

cancel_event = None
start_time = 0

gguf = llama_gguf()

# Stopping criteria for cancel the generation process
class StopCriteria(StoppingCriteria):
	def __init__(self, event):
		self.event = event

	def __call__(self, *args, **kwargs):
		return self.event.is_set() 

def load_model(prefix, model_path):
	global cancel_event
 
	full_path = MODEL_PATH_TEMPLATE.format(prefix, model_path)
	print("{}Loading: {}{}".format(Fore.LIGHTGREEN_EX, full_path, Style.RESET_ALL))

	model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_TEMPLATE.format(PATH_PREFIX, model_path), torch_dtype=torch.bfloat16, device_map="auto",)   
	model.to('cuda')
	tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_TEMPLATE.format(PATH_PREFIX, model_path))
  
	cancel_event = asyncio.Event()
	return model, tokenizer

def create_convo(system_role, history, prompt, input_file, input_use_history, input_debug_log): 
	convo = [] 
 
	if not using_gguf_model:
		convo = create_convo_local(system_role, history, prompt, input_use_history)
	else:
		file_date, file_type=gguf.process_file(input_file)
		convo = gguf.create_convo(system_role, history, prompt, input_use_history, file_date, file_type, combine_mode=True)
	
	if input_debug_log:
		print("{}convo = {}{}".format(Fore.LIGHTGREEN_EX, convo, Style.RESET_ALL))
	
	return convo

def create_convo_local(system_role, history, prompt, input_use_history):
	convo = [{'role': 'system', 'content': system_role}]
	if len(history) > 0 and input_use_history:
		convo.extend(history)
	convo.append({'role': 'user', 'content': prompt,})
	return convo

def do_chat(messages): 
	global cancel_event
	global start_time
 
	# Preparation for inference
	input_ids = tokenizer.apply_chat_template(
	 	messages, tokenize=False, add_generation_prompt=True
	 )
	inputs = tokenizer(input_ids, return_tensors="pt")
	inputs=inputs.to(model.device)

	streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

	generation_kwargs = dict(
		**inputs,
		streamer=streamer,
	)
 
	# Clear the event
	cancel_event.clear() 
	# Add stopping criteria
	generation_kwargs['stopping_criteria'] = [StopCriteria(cancel_event)]
			
	# Time counter
	start_time = time.time()

	thread = Thread(target=model.generate, kwargs=generation_kwargs)
	thread.start()
	for new_text in streamer:
		yield new_text

	# Wait for the thread to finish
	thread.join()
	print("{}Time used: {} seconds{}".format(Fore.LIGHTGREEN_EX, time.time() - start_time, Style.RESET_ALL))

def append_file_content_to_prompt(file_path, prompt):
    with open(file_path, 'r', encoding='utf-8') as file:
        file_content = file.read()
    prompt = file_content + prompt
    return prompt

@spaces.GPU()
@torch.no_grad()	
def generate_description(message: dict, history, btn_cancel, system_role, input_use_history, input_temperature, input_top_k, input_top_p, input_repetition_penalty, input_max_new_tokens, input_dosample:bool, input_debug_log) -> Generator[str, None, None]:
	if not using_gguf_model:
		# Create config
		full_path = MODEL_PATH_TEMPLATE.format(PATH_PREFIX, MODEL_USE)
		config = GenerationConfig.from_pretrained(full_path, temperature = input_temperature, top_p = input_top_p, repetition_penalty = input_repetition_penalty, do_sample = input_dosample, max_new_tokens = input_max_new_tokens)
		model.generation_config = config

	prompt = message['text'].strip()	
 
	# Create conversation
	input_file = None
	if len(message["files"]) > 0:
		input_file = message["files"][0]
	convo = create_convo(system_role, history, prompt, input_file, input_use_history, input_debug_log) 
 
	if not using_gguf_model:
		# Generate response
		response = do_chat(convo)

		outputs = []
		for new_text in response:
			outputs.append(new_text)
			yield "".join(outputs)
	else:
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
				outputs += message['content']
				yield outputs		

	print("{}output_text = {}{}".format(Fore.LIGHTGREEN_EX, outputs, Style.RESET_ALL))
  
	gc.collect()
	torch.cuda.empty_cache()

def cancel_btn():
	if cancel_event:
		print("{}Cancel button clicked!{}".format(Fore.LIGHTRED_EX, Style.RESET_ALL))	
		cancel_event.set()	
		print("{}Time used: {} seconds{}".format(Fore.LIGHTGREEN_EX, time.time() - start_time, Style.RESET_ALL))

if __name__ == "__main__":	
	PATH_TEMPLATE = os.path.splitext(os.path.basename(__file__))[0]
	PATH_PREFIX, MODEL_USE, N_THREADS, N_THREADS_BATCH, N_GPU_LAYERS, N_CTX, VERBOSE, using_gguf_model, FINETUNE_PATH, TITLE = parse_arguments(PATH_TEMPLATE)
	vision_model = check_vision_support(MODEL_USE)
  
	if using_gguf_model:
		gguf.load_model(prefix=PATH_PREFIX, model_name=MODEL_USE, n_threads=N_THREADS, n_threads_batch=N_THREADS_BATCH, n_gpu_layers=N_GPU_LAYERS, n_ctx=N_CTX, verbose=VERBOSE)
	else:
		model, tokenizer = load_model(PATH_PREFIX, MODEL_USE)
 
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
							value=0.9,
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
				gr.Checkbox(label="Do Sample", value=True, render=False),					
				gr.Checkbox(label="Show prompt in log window", value=True, render=False),	
			],
		)	

	demo.launch()
