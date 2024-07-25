import gradio as gr
import random
import time
import argparse

import sys
sys.path.append('./nn')

# SewingGPT
from evaluation_scripts.predict_class import infer

# PBR textures generation
from material_gen.test_finetune import gen_texture

import gradio as gr
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import cv2
import potpourri3d as pp3d
import subprocess
from GPT import GPT_api
import json
import re

import customconfig

class UI_interface():
	def __init__(self, shape_config_path, GPT_enable = True, sim_enable = True):

		self.Pattern_infer = infer(shape_config_path)
		self.gen_texture = gen_texture()	

		self.attempt_num = 0
		self.not_copy = True

		self.system_info = customconfig.Properties('./system.json')

		self.GPT_enable = GPT_enable
		self.sim_enable = sim_enable

		if GPT_enable: self.GPT_api = GPT_api()

	def create_mtl_file(self, file_path, texture_path):
		mtl_content = f"""\
newmtl Default_OBJ
Ns 225.000000
Ka 1.000000 1.000000 1.000000
Kd 0.800000 0.800000 0.800000
Ks 0.500000 0.500000 0.500000
Ke 0.000000 0.000000 0.000000
Ni 1.450000
d 1.000000
illum 2
map_Kd {texture_path}
		"""

		with open(file_path, 'w') as mtl_file:
			mtl_file.write(mtl_content)

	def append_mtllib_to_obj(self, obj_file_path, mtl_file_name, obj_file_path_new):
		with open(obj_file_path, 'r') as obj_file:
			obj_content = obj_file.read()

		obj_content = f"mtllib {mtl_file_name}\n" + obj_content

		with open(obj_file_path_new, 'w') as obj_file:
			obj_file.write(obj_content)

	def write_render_info(self, json_path, output_path):
		info = {}
		info["obj"] = self.obj_paths
		info["texture"] = self.texture_paths
		info["output"] = output_path
		with open(json_path, 'w') as f:
			json.dump(info, f)
	
	def texture_combine(self, folder):
		diffuse   = cv2.imread(os.path.join(folder, 'texture_diffuse.png'))
		normal    = cv2.imread(os.path.join(folder, 'texture_normal.png'))
		roughness = cv2.imread(os.path.join(folder, 'texture_roughness.png'))

		H, W = diffuse.shape[:2]
		combined = np.concatenate([diffuse[:, :W // 2], normal[:, W // 2: W // 4 * 3], roughness[:, W // 4 * 3:]], axis = 1)

		cv2.imwrite(os.path.join(folder, 'texture_combined.png'), combined)

		return os.path.join(folder, 'texture_combined.png')

	def run_infer(self, text_prompt, temperature, texture_prompt = None, temp_obj_path = None, num = None, sim_enable = True):
		if temp_obj_path is None: temp_obj_path = self.system_info["human_obj_path"]
		filename = self.system_info["sim_json_path"]
		with open(filename, 'r') as f:
			data = json.load(f)

		data['bodies_path'] = temp_obj_path

		with open(filename, 'w') as f:
			json.dump(data, f)


		obj_path = self.Pattern_infer.forward(text_prompt, temperature / 100., num = num, sim = sim_enable)

		if sim_enable:
			self.obj_paths.append(obj_path)
		folder = os.path.dirname(obj_path)
		
		obj_path_new = None
		video_path = None

		if texture_prompt is not None:
			self.gen_texture.run(texture_prompt, folder)
			texture_name = 'texture_diffuse.png'
			texture_path = os.path.join(folder, texture_name)

			if sim_enable:
				self.texture_paths.append(texture_path)
				render_info_path = os.path.join(folder, 'render_info.json')
				self.write_render_info(render_info_path, folder)
				self.recent_render_info_path = render_info_path
				
				blender_executable = self.system_info["blender_path"]
				script_path = './nn/blender/texture_multi.py'

				command = f'"{blender_executable}" -b -P {script_path} -- "{render_info_path}"'
				process = subprocess.Popen(command)
				process.wait()
				video_path = os.path.join(folder, 'render.mp4')

				obj_path_new = obj_path[:-4] + '_raw.obj'
				V, F = pp3d.read_mesh(obj_path)
				pp3d.write_mesh(V, F, obj_path_new)
			else:
				obj_path_new = obj_path
		
		Pattern_image_path = os.path.join(folder, 'pred_0_pattern.png')
		Texture_image_path = self.texture_combine(folder)

		return obj_path_new, video_path, Pattern_image_path, Texture_image_path
	
	
	def merge_objs(self, file1, file2, output_file):
		vertices = []
		faces = []

		with open(file1, 'r') as f:
			for line in f:
				elements = line.split()
				if elements[0] == 'v':
					vertices.append([float(elements[1]), float(elements[2]), float(elements[3])])
				elif elements[0] == 'f':
					faces.append([int(elements[1]), int(elements[2]), int(elements[3])])

		offset = len(vertices)
		with open(file2, 'r') as f:
			for line in f:
				elements = line.split()
				if len(elements) > 0:
					if elements[0] == 'v':
						vertices.append([float(elements[1]), float(elements[2]), float(elements[3])])
					elif elements[0] == 'f':
						faces.append([int(elements[1]) + offset, int(elements[2]) + offset, int(elements[3]) + offset])

		with open(output_file, 'w') as f:
			for vertex in vertices:
				f.write(f"v {vertex[0]} {vertex[1]} {vertex[2]}\n")
			for face in faces:
				f.write(f"f {face[0]} {face[1]} {face[2]}\n")

	def respond(self, message, chat_history):
		if self.GPT_enable:
			response = self.GPT_api.GPT_response(message)
		else:
			response = message

		chat_history.append((message, response))
		
		if self.GPT_enable:
			output = json.loads(re.search(r"{.*?}", response, flags=re.DOTALL).group())
			for key in output.keys():
				garments = output[key]
		else:
			prompts = response.split(';')
			garments = []
			for prompt in prompts:
				garment = prompt.split('/')
				if len(garment) == 1: garment.append('') # if no texture prompt
				garments.append(garment)
			

		output_images_history = []

		obj_path = self.system_info["human_obj_path"]
		self.obj_paths = []
		self.texture_paths = []
		for i, garment in enumerate(garments):
			text_prompt = garment[0]
			texture_prompt = garment[1]

			obj_path_new, video_path, Pattern_image_path, Texture_image_path = self.run_infer(text_prompt, 70, texture_prompt, obj_path, i, sim_enable = self.sim_enable)
			obj_path_combine = obj_path_new
			if self.sim_enable: self.merge_objs(obj_path, obj_path_new, obj_path_combine)
			obj_path = obj_path_combine
			
			output_images_history.append((None, f"[Attempt {self.attempt_num}] Generate No.{i}... garment prompt: {text_prompt}; texture prompt: {texture_prompt}"))
			output_images_history.append((None, (Pattern_image_path, )))

			if len(texture_prompt) > 0: output_images_history.append((None, (Texture_image_path, )))
			
		self.attempt_num += 1

		gpt_response_txt_path = os.path.join(os.path.dirname(obj_path), 'gpt_response.txt')
		
		with open(gpt_response_txt_path, 'w') as f:
			f.write(response)

		UV_path = None
		if self.sim_enable:
			UV_path = os.path.join(os.path.dirname(obj_path), 'pred_0_sim_uv.png')
			self.UV_path = UV_path
		

		return "", chat_history, output_images_history, video_path, UV_path
	
	def predict(self, im):

		if self.not_copy:
			self.not_copy = False
			os.system('copy ' + self.texture_paths[-1] + ' ' + self.texture_paths[-1].split('.')[0] + '_backup.png')

		back_image = cv2.imread(self.texture_paths[-1].split('.')[0] + '_backup.png')
		 
		if len(im["layers"]) > 0:
			draw = cv2.resize(im["layers"][0], (back_image.shape[1], back_image.shape[0]))
			if draw[..., 3].max() > 0:
				cv2.imwrite(self.texture_paths[-1], draw[..., :3][..., ::-1] * (draw[..., 3:] / 255.) + back_image * (1. - draw[..., 3:] / 255.) )

				blender_executable = self.system_info["blender_path"]
				script_path = 'nn/blender/texture_multi.py'

				command = f'"{blender_executable}" -b -P {script_path} -- "{self.recent_render_info_path}"'
				print(command)
				process = subprocess.Popen(command)
				process.wait()

		video_path = os.path.join(os.path.dirname(self.texture_paths[-1]), 'render.mp4')
		return video_path

	def run(self):

		with gr.Blocks() as demo1:
			with gr.Row() as row:
				with gr.Column():
					chatbot = gr.Chatbot(height=500)
					msg = gr.Textbox()
					clear = gr.ClearButton([msg, chatbot])
				with gr.Column():
					output_images = gr.Chatbot(height=500)
					output_video = gr.Video(label="Out")
					output_UV = gr.Image()

			msg.submit(self.respond, [msg, chatbot], [msg, chatbot, output_images, output_video, output_UV])


		with gr.Blocks() as demo2:
			with gr.Row():
				im = gr.ImageEditor(
					type="numpy",
					crop_size="1:1",
				)
				im_preview = gr.Video(label="Out")
			
			
			im.change(self.predict, outputs=im_preview, inputs=im)

		demo = gr.TabbedInterface([demo1, demo2], ["Generation", "Texture Editing"])
		
		
		demo.launch()

if __name__ == "__main__":
	parser = argparse.ArgumentParser()

	parser.add_argument('--sim', help='enable simulation', action='store_true')
	parser.add_argument('--GPT', help='enable GPT interaction', action='store_true')
	args = parser.parse_args()

	UI = UI_interface(shape_config_path = './models/infer.yaml', GPT_enable = args.GPT, sim_enable = args.sim)
	UI.run()



