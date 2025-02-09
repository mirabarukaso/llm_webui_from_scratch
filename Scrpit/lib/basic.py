from colorama import Fore, Style
import os
import base64
from PIL import Image
from io import BytesIO
from transformers import StoppingCriteria
import argparse
import mimetypes

MODEL_PATH_TEMPLATE 	= '..\\{}\\{}'
GGUF_PATH_TEMPLATE 		= '..\\{}\\{}\\{}'

MAX_IMAGE_SIZE = 640

SYSTEM_ROLE = [
	"You are a helpful assistant.",
]

LATEX_DELIMITERS_SET= [
	{'left': "$$", 'right': "$$", 'display': True},
	{'left': "\\(", 'right': "\\)", 'display': False},
	{'left': "\\begin{equation}", 'right': "\\end{equation}", 'display': True},
	{'left': "\\begin{align}", 'right': "\\end{align}", 'display': True},
	{'left': "\\begin{alignat}", 'right': "\\end{alignat}", 'display': True},
	{'left': "\\begin{gather}", 'right': "\\end{gather}", 'display': True},
	{'left': "\\begin{CD}", 'right': "\\end{CD}", 'display': True},
	{'left': "\\[", 'right': "\\]", 'display': True},
]

def check_vision_support(model_path, vision_keyword="-vision-"):	
	if not str(model_path).lower().__contains__(vision_keyword.lower()):
		print("{}INFO: This is NOT Vision model{}".format(Fore.LIGHTGREEN_EX, Style.RESET_ALL))
		return False
	else:
		print("{}INFO: This is Vision model{}".format(Fore.LIGHTMAGENTA_EX, Style.RESET_ALL))
		return True

def process_files(files, vision_model, input_image_size = 0):
	images = ''	
	file_prompt = ''
	if len(files) > 0:
		if not vision_model:
			file_prompt = read_files(files)
		else:
			image_list = []
			file_list = []
			for file in files:
				mime_type, _ = mimetypes.guess_type(file)
				if mime_type and mime_type.startswith('image'):
					image_list.append(file)
				else:
					file_list.append(file)
	
			images = load_images(image_list, input_image_size) 
			file_prompt = read_files(file_list)
	return images, file_prompt

def read_files(files):
    content = ''
    for file in files:
        content += f'{read_file_content_to_prompt(file)}\n'
    return content

def read_file_content_to_prompt(file_path):
	with open(file_path, 'r', encoding='utf-8') as file:
		file_content = file.read()
	return file_content

def image_to_base64(image):
	buffered = BytesIO()
	image.save(buffered, format="JPEG")
	img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
	return img_str

def resize_image(image, max_size=384):
	# Resize the image to ensure its longest side is equal to max_size while maintaining aspect ratio.
	width, height = image.size
	out_image = None
 
	if width > height:
		new_width = max_size
		new_height = int(max_size * height / width)
		out_image = image.resize((new_width, new_height), Image.LANCZOS)
	else:
		new_height = max_size
		new_width = int(max_size * width / height)
		out_image = image.resize((new_width, new_height), Image.LANCZOS)
	 	
	out_image = out_image.convert("RGB")	
	return out_image

def resize_images(images, size = MAX_IMAGE_SIZE):
	resized_image_list = []
	for image in images:
		image = resize_image(image, size)
		resized_image_list.append(image)
	return resized_image_list

def load_images(images, size = MAX_IMAGE_SIZE):
	if not images:
		return None

	image_list = []
	for image in images:
		image = Image.open(image)
		image_list.append(image)
  
	return resize_images(image_list, size)

# Stopping criteria for cancel the generation process
class MyStopCriteria(StoppingCriteria):
	def __init__(self, event):
		self.event = event
  
	def __call__(self, *args, **kwargs):
		return self.event.is_set()					  	

def parse_arguments(path_prefix):
	parser = argparse.ArgumentParser(description="Parse command line arguments.")
	parser.add_argument("-p", "--path_prefix", type=str, default=path_prefix, help="Path prefix for the model.")
	parser.add_argument("model_use", type=str, help="Model to use.")
	parser.add_argument("-n", "--n_gpu_layers", type=int, default=-1, help="Number of GPU layers. Use -1 for max layers.")
	parser.add_argument("-c", "--n_ctx", type=int, default=0, help="Number of Text context.")
	parser.add_argument("-v", "--verbose", action='store_true', help="Enable verbose mode.")
	parser.add_argument("-g", "--gguf_model", action='store_true', help="Use GGUF model.")
	parser.add_argument("-f", "--finetune_path", type=str, default="", help="Path to finetune model.")
	parser.add_argument("-t", "--n_threads", type=int, default=None, help="The number of threads to use when processing.")
	parser.add_argument("-tb", "--n_threads_batch", type=int, default=None, help="The number of threads to use when batch processing.")
	parser.add_argument("-lp", "--lora_patch", type=str, default=None, help="Path to a LoRA file to apply to the model.")
	parser.add_argument("-ls", "--lora_scale", type=float, default=1.0, help="Lora scale, default 1.0.")
	parser.add_argument("-ml", "--mlock", type=bool, default=False, help="Force system to keep model in RAM rather than swapping or compressing.")
	parser.add_argument("-nmm", "--no_mmap", type=bool, default=False, help="Do not memory-map model (slower load but may reduce pageouts if not using mlock.")
	parser.add_argument("-ch", "--chat_handler", type=str, default=None, help="Load chat handler (CLIP) for gguf vision model.")

	args = parser.parse_args()
	title = os.path.basename(args.model_use)
	return args.path_prefix, args.model_use, args.n_threads, args.n_threads_batch, args.n_gpu_layers, args.n_ctx, args.verbose, args.gguf_model, args.finetune_path, args.lora_patch, args.lora_scale, args.mlock, args.no_mmap, args.chat_handler, title, 


