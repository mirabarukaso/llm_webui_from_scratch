from llama_cpp import Llama, StoppingCriteriaList
from colorama import Fore, Style
from .basic import GGUF_PATH_TEMPLATE, read_file_content_to_prompt, resize_image
import asyncio
import os
import mimetypes
import tempfile
from PIL import Image
from io import BytesIO
import random
import sys

GGUF_PATH_TEMPLATE 	= '..\\{}\\{}\\{}'

TEMPLATE_SYSTEM 	= 0
TEMPLATE_USER 		= 1
TEMPLATE_ASSISTANT 	= 2
TEMPLATE_REPLY	 	= 3
TEMPLATE_DEEPSEEK	= ['<｜User｜>{}\n', '<｜User｜>{}\n', '<｜Assistant｜>{}', '<｜Assistant｜>']	#The template example gives out the full-width characters....
TEMPLATE_QWEN2	 	= ['<|im_start|>system\n{}<|im_end|>\n','<|im_start|>user\n{}<|im_end|>\n','<|im_start|>assistant\n{}<|im_end|>\n', '<|im_start|>assistant']
TEMPLATE_LLAMA3 	= ['<|start_header_id|>system<|end_header_id|>{}<|eot_id|>\n','<|start_header_id|>user<|end_header_id|>{}<|eot_id|>\n','<|start_header_id|>assistant<|end_header_id|>{}<|eot_id|>\n', '<|start_header_id|>assistant<|end_header_id|>']
TEMPLATE_OPENBUDDY	= ['<|role|>system<|says|>{}<|end|>\n','<|role|>user<|says|>{}<|end|>\n','<|role|>assistant<|says|>{}<|end|>\n', '<|role|>assistant<|says|>']
TEMPLATE_AYA_EXP	= ['<|START_OF_TURN_TOKEN|><|SYSTEM_TOKEN|>{}<|END_OF_TURN_TOKEN|>\n','<|START_OF_TURN_TOKEN|><|USER_TOKEN|>{}<|END_OF_TURN_TOKEN|>\n','<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>{}<|END_OF_TURN_TOKEN|>\n', '<|START_OF_TURN_TOKEN|><|CHATBOT_TOKEN|>']
TEMPLATE_DEFAULT 	= TEMPLATE_LLAMA3

IMG_BASE64_TAG	 	= 'data:image/png;base64,{}'
IMG_FILE_TAG	 	= 'file:///{}'

def get_file_extension(file_path):
    _, file_extension = os.path.splitext(file_path)
    return file_extension

class llama_gguf:
	cancel_event = None
	model = None
	
	def __init__(self):
		self.cancel_event = asyncio.Event()
		self.stopping_criteria = StoppingCriteriaList([
			self.custom_stopping_criteria(self.cancel_event)
		])  
		self.chat_template = TEMPLATE_DEFAULT
		self.vision_model = False
  
		with tempfile.TemporaryDirectory() as self.tmpdirname:
			print('{}Created temporary directory:{}{}'.format(Fore.LIGHTCYAN_EX, self.tmpdirname, Style.RESET_ALL))
  
	def custom_stopping_criteria(self, local_llm_stop_event):
		def f(input_ids, score, **kwargs) -> bool:
				return local_llm_stop_event.is_set()
		return f
  
	def load_model(self, prefix, model_name, n_threads=16, n_threads_batch=16, n_gpu_layers=45, n_ctx = 8192, verbose = False, lora_path = None, lora_scale=1.0, use_mmap = False, use_mlock = False, chat_handler=None):
		# Load the model and processor and tokenizer
		if str(model_name).startswith("GGUF_"):
			gguf_filename = model_name.replace("GGUF_", "") + ".gguf"
			gguf_full_filename = GGUF_PATH_TEMPLATE.format(prefix, model_name, gguf_filename)
			print("{}Loading GGUF: {}\n| n_threads: {} | n_threads_batch: {} | n_gpu_layers: {} | n_ctx: {} | verbose: {} |{}".format(
				Fore.LIGHTGREEN_EX, gguf_full_filename, n_threads, n_threads_batch, n_gpu_layers, n_ctx, verbose, Style.RESET_ALL))
			if lora_path:
				print("{}Loading LORA: {} | Lora Scale: {}|{}".format(Fore.LIGHTRED_EX, lora_path, lora_scale, Style.RESET_ALL))
			
			if not chat_handler:
				self.model = Llama(model_path=gguf_full_filename, n_threads=n_threads, n_threads_batch=n_threads_batch, n_gpu_layers=n_gpu_layers, 
							verbose = verbose, n_ctx=n_ctx, lora_path=lora_path, lora_scale=lora_scale, use_mmap=use_mmap, use_mlock=use_mlock, flash_attn=True, seed=-1,
						)
			else:
				self.model = Llama(model_path=gguf_full_filename, n_threads=n_threads, n_threads_batch=n_threads_batch, n_gpu_layers=n_gpu_layers, 
							verbose = verbose, n_ctx=n_ctx, lora_path=lora_path, lora_scale=lora_scale, use_mmap=use_mmap, use_mlock=use_mlock, flash_attn=True, seed=-1,
							chat_handler=chat_handler
						)
				self.vision_model = True
   
			if str(gguf_filename).lower().__contains__('deepseek'):
				self.chat_template = TEMPLATE_DEEPSEEK
			elif str(gguf_filename).lower().__contains__('llama'):
				self.chat_template = TEMPLATE_LLAMA3
			elif str(gguf_filename).lower().__contains__('qwen'):
				self.chat_template = TEMPLATE_QWEN2
			elif str(gguf_filename).lower().__contains__('openbuddy'):
				self.chat_template = TEMPLATE_OPENBUDDY
			elif str(gguf_filename).lower().__contains__('aya-expanse'):
				self.chat_template = TEMPLATE_AYA_EXP		
			else:
				self.chat_template = TEMPLATE_DEFAULT
		else:
			error_message = "Error: Unsupported model type."
			raise RuntimeError("{}{}{}".format(Fore.LIGHTRED_EX, error_message, Style.RESET_ALL))

	def process_history_for_create_completion(self, history):
		convo = []
		for entry in history:
			human_content = entry['content'] if entry['role'] == 'user' else ''
			human_content = self.process_input_messages(human_content)
			assistant_content = entry['content'] if entry['role'] == 'assistant' else ''
			if human_content:
				convo.append(self.chat_template[TEMPLATE_USER].format(human_content))
			if assistant_content:
				convo.append(self.chat_template[TEMPLATE_ASSISTANT].format(assistant_content))
		return convo
	
	def process_history_for_create_chat_completion(self, history):
		convo = []
		for entry in history:
			human_content = entry['content'] if entry['role'] == 'user' else ''
			if entry['role'] == 'user' and isinstance(human_content, tuple) and len(human_content) == 1 and os.path.isfile(human_content[0]):
				# Ignore image file
				continue
			human_content = self.process_input_messages(human_content)
			assistant_content = entry['content'] if entry['role'] == 'assistant' else ''
			if human_content:
				convo.append({"role": "user", "content": human_content})
			if assistant_content:
				convo.append({"role": "assistant", "content": assistant_content})
		return convo	
 
	def process_input_messages(self, content):
		if isinstance(content, tuple) and len(content) == 1:
			file_path = content[0]
			if os.path.isfile(file_path):
				with open(file_path, 'r', encoding='utf-8') as file:
					file_content = file.read()
				return file_content
		return content

	def create_convo_for_create_completion(self, system_role, history, prompt, use_history, file_list, no_system_prompt):
		convo = []
  
		if not no_system_prompt:
			convo.append(self.chat_template[TEMPLATE_SYSTEM].format(system_role))
   
		if len(history) > 0 and use_history:
			convo.extend(self.process_history_for_create_completion(history))

		if file_list:
			#prefix_data = ''
			for file_data, file_type in file_list:
				if file_type == 'text' or file_type == 'json':
					#prefix_data += file_data
					convo.append(self.chat_template[TEMPLATE_USER].format(f'{file_data}'))
				elif file_type == 'image':
					#BUG: NOT SUPPORT!!!
					pass
			
			if prompt:
				#convo.append(self.chat_template[TEMPLATE_USER].format(f'{prefix_data}\n{prompt}'))
				convo.append(self.chat_template[TEMPLATE_USER].format(f'{prompt}'))
		else:			
			convo.append(self.chat_template[TEMPLATE_USER].format(prompt))
		
		convo.append(self.chat_template[TEMPLATE_REPLY])
		return ''.join(convo)

	def create_convo_for_create_chat_completion(self, system_role, history, prompt, use_history, file_list, no_system_prompt):
		convo = []

		if not no_system_prompt:
			convo.append({"role": "system", "content": system_role})

		if len(history) > 0 and use_history:
			convo.extend(self.process_history_for_create_chat_completion(history))

		if file_list:
			prefix_data = ''
			content_list = [{"type": "text", "text": f'{prefix_data}{prompt}'}]
			for file_data, file_type in file_list:
				if file_type == 'text' or file_type == 'json':
					prefix_data += file_data
				elif file_type == 'image':
					content_list.append({"type": "image_url", "image_url": {"url": file_data }})

			convo.append({
				"role": "user",
				"content": content_list
			})
		else:
			convo.append({"role": "user", "content": prompt})

		return convo

	def create_convo(self, system_role, history, prompt, use_history=False, file_list=None, no_system_prompt=False):
		if not self.vision_model:
			return self.create_convo_for_create_completion(system_role, history, prompt, use_history, file_list, no_system_prompt)
		else:
			return self.create_convo_for_create_chat_completion(system_role, history, prompt, use_history, file_list, no_system_prompt)

	def do_chat_ex(self, convo, input_temperature = 0.8, input_top_k = 40, input_top_p = 0.95, input_repetition_penalty = 1.1, input_max_new_tokens = 2048, stop = None):	   
		return self.model.create_completion(
			prompt=convo,
			stream=True,
			temperature=input_temperature,
			top_k=input_top_k,
			top_p=input_top_p,
			repeat_penalty=input_repetition_penalty,
			max_tokens=input_max_new_tokens,
			stopping_criteria=self.stopping_criteria,
   			stop=stop if stop else self.model.token_eos()
		)

	def do_chat(self, convo, input_temperature = 0.8, input_top_k = 40, input_top_p = 0.95, input_repetition_penalty = 1.1, input_max_new_tokens = 2048, stop = None, input_debug_log = False):
		if input_debug_log:
			self.debug_log_convo(convo)
   
		outputs = ""
		streamer = self.get_streamer(convo, input_temperature, input_top_k, input_top_p, input_repetition_penalty, input_max_new_tokens, stop)
		for msg in streamer:
			outputs = self.process_stream_message(msg, outputs)
			yield outputs
	 
		if input_debug_log:
			self.debug_log_output(outputs)

	def debug_log_convo(self, convo):
		print("{}convo = {}{}".format(Fore.LIGHTGREEN_EX, convo, Style.RESET_ALL))

	def get_streamer(self, convo, input_temperature, input_top_k, input_top_p, input_repetition_penalty, input_max_new_tokens, stop):
		if self.vision_model:
			return self.model.create_chat_completion(
				messages=convo,
				stream=True,
				temperature=input_temperature,
				top_k=input_top_k,
				top_p=input_top_p,
				repeat_penalty=input_repetition_penalty,
				max_tokens=input_max_new_tokens,
				stop=stop if stop else self.model.token_eos()
			)
		else:
			return self.do_chat_ex(convo, input_temperature, input_top_k, input_top_p, input_repetition_penalty, input_max_new_tokens, stop)

	def process_stream_message(self, msg, outputs):
		message = msg['choices'][0]
		if 'delta' in message:
			message = message['delta']
			if 'content' in message:
				outputs += message['content']
		elif 'text' in message:
			new_token = message['text']
			if new_token != "<":
				outputs += new_token
		return outputs

	def debug_log_output(self, outputs):
		print("{}output_text = {}{}".format(Fore.LIGHTRED_EX, outputs, Style.RESET_ALL))

	def process_files(self, files, input_image_size):
		files_list =[]
		for file in files:
			file_content, file_fype = self.process_file(file, input_image_size)
			files_list.append([file_content, file_fype])
		return files_list

	def process_file(self, file_path, input_image_size):
		if not file_path:
			return None, 'None'

		mime_type, _ = mimetypes.guess_type(file_path)
		if mime_type:
			if mime_type == 'text/plain':
				file_content = read_file_content_to_prompt(file_path)	
				return file_content, 'text'
			elif mime_type == 'application/json':
				file_content = read_file_content_to_prompt(file_path)	
				return file_content, 'json'
			elif mime_type.startswith('image') and input_image_size > 0:
				file_extension = get_file_extension(file_path)
				image = Image.open(file_path)				
				resized_image = resize_image(image, input_image_size)

				image_bytes = BytesIO()
				resized_image.save(image_bytes, format=image.format)
				image_bytes = image_bytes.getvalue()
				random_integer = random.randint(0, sys.maxsize)
    
				image_path = '{}{}{}'.format(self.tmpdirname, random_integer, file_extension)
				with open(image_path, 'wb') as f:
					f.write(image_bytes)
				print(f'Saved image to {image_path}')    
    
				return IMG_FILE_TAG.format(file_path), 'image'
		else:
			print("{}Error: Could not determine the file type{}".format(Fore.LIGHTRED_EX, Style.RESET_ALL))
			return None, 'None'
		
