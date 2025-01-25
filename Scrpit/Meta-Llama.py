from lib.gguf_model import llama_gguf
from threading import Thread
import time
import spaces
import torch
import gradio as gr
from transformers import GenerationConfig, MllamaForConditionalGeneration, StoppingCriteria, TextIteratorStreamer, AutoProcessor, AutoModelForCausalLM, AutoTokenizer
from PIL import Image
from typing import Generator
import asyncio
import gc
import os
from lib.basic import check_vision_support, parse_arguments, resize_image
from lib.basic import LATEX_DELIMITERS_SET, SYSTEM_ROLE, MODEL_PATH_TEMPLATE, MAX_IMAGE_SIZE
from colorama import Fore, Style

MODEL_USE = ""
PATH_PREFIX = ""

using_gguf_model = False

model = None
processor = None
tokenizer = None

cancel_event = None
start_time = 0

vision_model = True
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

	if not vision_model:
		model = AutoModelForCausalLM.from_pretrained(full_path, torch_dtype=torch.bfloat16, device_map="auto",)   		  
	else:
		model = MllamaForConditionalGeneration.from_pretrained(full_path, torch_dtype=torch.bfloat16, device_map="auto",)   		  
  
	processor = AutoProcessor.from_pretrained(full_path)
	tokenizer = AutoTokenizer.from_pretrained(full_path)
	cancel_event = asyncio.Event()
	return model, processor, tokenizer

def create_convo(system_role, history, prompt, input_file, input_use_history, input_debug_log, input_image_size): 
	convo = [] 
 
	if not using_gguf_model:
		image = Image.open(input_file)
		resized_image = resize_image(image, input_image_size)
		convo = create_convo_local(system_role, history, prompt, resized_image, input_use_history)
	else:
		file_date, file_type=gguf.process_file(input_file)
		convo = gguf.create_convo(system_role, history, prompt, input_use_history, file_date, file_type)
	
	if input_debug_log:
		print("{}convo = {}{}".format(Fore.LIGHTGREEN_EX, convo, Style.RESET_ALL))
	
	return convo

def create_convo_local(system_role, history, prompt, images, input_use_history):
	convo = [{'role': 'system', 'content': system_role}]
	if len(history) > 0 and input_use_history:
		convo.extend(history)
	if not images:
		convo.append({'role': 'user', 'content': [{"type": "text", "text": prompt}]})
	else:
		convo.append({'role': 'user', 'content': [{"type": "image"}, {"type": "text", "text": prompt}]})
	return convo

def do_chat(messages, images): 
	global cancel_event
	global start_time
	
	# Tokenize the prompt
	inputs = None
	generation_kwargs = None  
	streamer = None
	if not vision_model or not images:
	 	# Preparation for inference
		input_text = processor.apply_chat_template(
			messages, tokenize=False, add_generation_prompt=True
		)
		inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
  
		streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)

		generation_kwargs = dict(
			**inputs,
			streamer=streamer,
			pad_token_id=tokenizer.eos_token_id,
		)
	else:
		# Preparation for inference
		input_text = processor.apply_chat_template(messages, add_generation_prompt=True)
		inputs = processor(
			images,
			input_text,
			add_special_tokens=False,
			return_tensors="pt"
		).to(model.device)
  
		streamer = TextIteratorStreamer(processor, skip_prompt=True, skip_special_tokens=True)

		generation_kwargs = dict(
			**inputs,
			streamer=streamer,					
		)

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
		 
@spaces.GPU()
@torch.no_grad()
def generate_description(message: dict, history, btn_cancel, system_role, input_use_history, input_temperature, input_top_k, input_top_p, input_repetition_penalty, input_max_new_tokens, input_dosample:bool, input_debug_log, input_image_size) -> Generator[str, None, None]:	
	if not using_gguf_model:
		# Create config
		full_path = MODEL_PATH_TEMPLATE.format(PATH_PREFIX, MODEL_USE)
		config = GenerationConfig.from_pretrained(full_path, temperature = input_temperature, top_p = input_top_p, repetition_penalty = input_repetition_penalty, do_sample = input_dosample, max_new_tokens = input_max_new_tokens)
		model.generation_config = config

	prompt = message['text'].strip()
	if input_debug_log:
		print("{}prompt = {}{}".format(Fore.LIGHTGREEN_EX, prompt, Style.RESET_ALL))
 
	# Create conversation
	input_file = None
	if len(message["files"]) > 0:
		input_file = message["files"][0]
	convo = create_convo(system_role, history, prompt, input_file, input_use_history, input_debug_log, input_image_size)
 
	if not using_gguf_model:
		# Clear the event
		cancel_event.clear() 

		# Load Image
		image = Image.open(input_file)
		resized_image = resize_image(image, MAX_IMAGE_SIZE)
  
		# Generate response
		response = do_chat(convo, resized_image)

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
				stop = ['<|eot_id|>'],
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
	PATH_PREFIX, MODEL_USE, N_THREADS, N_THREADS_BATCH, N_GPU_LAYERS, N_CTX, VERBOSE, using_gguf_model, FINETUNE_PATH, LORA_PATH, LORA_SCALE, TITLE = parse_arguments(PATH_TEMPLATE)
	vision_model = check_vision_support(MODEL_USE)
 
	if using_gguf_model:
		gguf.load_model(prefix=PATH_PREFIX, model_name=MODEL_USE, n_threads=N_THREADS, n_threads_batch=N_THREADS_BATCH, n_gpu_layers=N_GPU_LAYERS, n_ctx=N_CTX, verbose=VERBOSE, lora_path=LORA_PATH, lora_scale=LORA_SCALE)
	else:
		model, processor, tokenizer = load_model(PATH_PREFIX, MODEL_USE)
	
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
				gr.Checkbox(label="Do Sample", value=True, render=False),					
				gr.Checkbox(label="Show prompt in log window", value=True, render=False),	
				gr.Slider(minimum=128, 
							maximum=1280,
							step=64,
							value=512, 
							label="Resize Image (only for multiple images)", 
							render=False ),
			],
		)
  
		gr.HTML(TITLE)

  
	demo.launch()
