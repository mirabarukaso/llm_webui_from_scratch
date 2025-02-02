from enum import Enum
from llama_cpp import Llama, LLAMA_DEFAULT_SEED
from PIL import Image
from colorama import Fore, Style
from .basic import GGUF_PATH_TEMPLATE, MAX_IMAGE_SIZE, read_file_content_to_prompt, append_file_content_to_prompt, resize_image, image_to_base64
import asyncio
import os
import mimetypes

class ggml_numa_strategy(Enum):
	GGML_NUMA_STRATEGY_DISABLED   = 0
	GGML_NUMA_STRATEGY_DISTRIBUTE = 1
	GGML_NUMA_STRATEGY_ISOLATE    = 2
	GGML_NUMA_STRATEGY_NUMACTL    = 3
	GGML_NUMA_STRATEGY_MIRROR     = 4

    
GGUF_PATH_TEMPLATE = '..\\{}\\{}\\{}'
HISTORY_REPLACE_MARK = '__REPLACE_BY_REAL_HISTORY__'
DEFAULT_SEED = LLAMA_DEFAULT_SEED

class llama_gguf:
	cancel_event = None
	model = None
	
	def __init__(self):
		self.cancel_event = asyncio.Event()
	
	# Stopping criteria for cancel the generation process
	class StopCriteria():
		def __init__(self, event):
			self.event = event

		def __call__(self, *args, **kwargs):
			return self.event.is_set() 
	
	def load_model(self, prefix, model_name, n_threads=16, n_threads_batch=16, n_gpu_layers=45, n_ctx = 8192, verbose = False, lora_path = None, lora_scale=1.0, use_mmap = False, use_mlock = False):
		# Load the model and processor and tokenizer
		if str(model_name).startswith("GGUF_"):
			gguf_filename = model_name.replace("GGUF_", "") + ".gguf"
			gguf_full_filename = GGUF_PATH_TEMPLATE.format(prefix, model_name, gguf_filename)
			print("{}Loading GGUF: {}\n| n_threads: {} | n_threads_batch: {} | n_gpu_layers: {} | n_ctx: {} | verbose: {} |{}".format(
				Fore.LIGHTGREEN_EX, gguf_full_filename, n_threads, n_threads_batch, n_gpu_layers, n_ctx, verbose, Style.RESET_ALL))
			if lora_path:
				print("{}Loading LORA: {} | Lora Scale: {}|{}".format(Fore.LIGHTRED_EX, lora_path, lora_scale, Style.RESET_ALL))
				#self.model = Llama(gguf_full_filename, n_threads=n_threads, n_threads_batch=n_threads_batch, n_gpu_layers=n_gpu_layers, verbose = verbose, n_ctx=n_ctx, lora_path=lora_path, lora_scale=lora_scale, use_mmap=False)  
			#else:
			self.model = Llama(gguf_full_filename, n_threads=n_threads, n_threads_batch=n_threads_batch, n_gpu_layers=n_gpu_layers, verbose = verbose, n_ctx=n_ctx, lora_path=lora_path, lora_scale=lora_scale, use_mmap=use_mmap, use_mlock=use_mlock)  
		else:
			error_message = "Error: Unsupported model type."
			raise RuntimeError("{}{}{}".format(Fore.LIGHTRED_EX, error_message, Style.RESET_ALL))

	def process_history(self, history, combine_mode=False):
		convo = []
		if combine_mode:
			convo = self._process_combined_history(history)
		else:
			convo = self._process_separate_history(history)
		self._clean_history_marks(convo)
		return convo

	def _process_combined_history(self, history):
		convo = []
		for entry in history:
			human_content, system_content = self._extract_contents(entry)
			self._append_human_content(convo, human_content)
			self._append_system_content(convo, system_content)
		return convo

	def _process_separate_history(self, history):
		convo = []
		for entry in history:
			human_content = entry['content'] if entry['role'] == 'user' else ''
			human_content = self.process_input_messages(human_content)
			system_content = entry['content'] if entry['role'] == 'assistant' else ''
			if human_content:
				convo.append({"role": "user", "content": human_content + "\n"})
			if system_content:
				convo.append({"role": "assistant", "content": system_content + "\n"})
		return convo

	def _extract_contents(self, entry):
		human_content = ''
		system_content = ''
		if entry['role'] == 'user':
			human_content = self.process_input_messages(entry['content'])
		if entry['role'] == 'assistant':
			system_content = entry['content']
		return human_content, system_content

	def _append_human_content(self, convo, human_content):
		if human_content:
			skip_append = False
			for i, conv in enumerate(convo):
				if conv['role'] == 'user' and HISTORY_REPLACE_MARK in conv['content']:
					convo[i]['content'] = conv['content'].replace(HISTORY_REPLACE_MARK, human_content)
					skip_append = True
					break
			if not skip_append:
				convo.append({"role": "user", "content": human_content + "\n"})

	def _append_system_content(self, convo, system_content):
		if system_content:
			convo.append({"role": "system", "content": system_content + "\n"})

	def _clean_history_marks(self, convo):
		for i, conv in enumerate(convo):
			convo[i]['content'] = conv['content'].replace(HISTORY_REPLACE_MARK, '')
	
	def process_input_messages(self, content):
		if isinstance(content, tuple) and len(content) == 1:
			file_path = content[0]
			if os.path.isfile(file_path):
				with open(file_path, 'r', encoding='utf-8') as file:
					file_content = file.read()
				return file_content + HISTORY_REPLACE_MARK
		return content

	def create_convo(self, system_role, history, prompt, use_history, file_data=None, file_type=None, combine_mode=False, no_system_prompt=False, use_history_back=False):
		convo = []
  
		if not no_system_prompt:
			convo.append({'role': 'system', 'content': system_role + "\n"})
   
		if len(history) > 0 and use_history:
			convo.extend(self.process_history(history, combine_mode))
   
		if file_data:
			if file_type == 'text' or file_type == 'json':
				convo.append({"role": "user", "content": file_data + prompt + "\n"})
			elif file_type == 'image':
				convo.append({"role": "user", "content": [
						{"type": "image_url", "image_url": {"url": file_data}},
						{"type" : "text", "text": prompt + "\n"}
                	] 
                  })
		else:
			convo.append({"role": "user", "content": prompt + "\n"})
   
		if len(history) > 0 and use_history_back and use_history:
			convo = convo[::-1]

		return convo

	def do_chat(self, convo, input_temperature = 0.8, input_top_k = 40, input_top_p = 0.95, input_repetition_penalty = 1.1, input_max_new_tokens = 2048, stop = None):
		streamer = None
  	
		if not stop:  
			streamer = self.model.create_chat_completion(
				convo, 
				stream=True,
				temperature=input_temperature,
				top_k=input_top_k,
				top_p=input_top_p,
				repeat_penalty=input_repetition_penalty,
				max_tokens=input_max_new_tokens,
			)
		else:
			streamer = self.model.create_chat_completion(
				convo, 
				stream=True,
				temperature=input_temperature,
				top_k=input_top_k,
				top_p=input_top_p,
				repeat_penalty=input_repetition_penalty,
				max_tokens=input_max_new_tokens,
				stop=stop,
			)
								
		return streamer

	def process_file(self, file_path):
		if not file_path:
			return None, 'None'

		mime_type, _ = mimetypes.guess_type(file_path)
		if mime_type:
			if mime_type.startswith('text') or mime_type == 'application/json':
				file_content = read_file_content_to_prompt(file_path)	
				return file_content, 'text'
			elif mime_type.startswith('image'):
				image = Image.open(file_path)
				resized_image = resize_image(image, MAX_IMAGE_SIZE)
				# TODO: This is not a good idea use base64 for image input, too much tokens
				image_base64 = image_to_base64(resized_image)
				return image_base64, 'image'
			else:
				raise ValueError("Unsupported file type")
		else:
			raise ValueError("Could not determine the file type")
		