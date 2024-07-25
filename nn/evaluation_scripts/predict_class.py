import argparse
from datetime import datetime
import numpy as np
from pathlib import Path
import torch
import yaml
import torch.nn as nn
import torch.nn.functional as F
import json

import subprocess

import os, sys
from tqdm import tqdm

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# My modules
import customconfig
import data
from experiment import ExperimentWrappper
from pattern.wrappers import VisPattern
import net_blocks as blocks

SOS = 2001
EOS = 2002
PAD = 2003
C_outline    = 50
C_rotation   = 1000
C_transl     = 1000
C_stitch_tag = 1000


def get_values_from_args(shape_config_path = './models/infer.yaml'):
	system_info = customconfig.Properties('./system.json')

	with open(shape_config_path, 'r') as f:
		shape_config = yaml.safe_load(f)

	saving_path = Path(system_info['output']) / (datetime.now().strftime('%y%m%d-%H-%M-%S'))
	saving_path.mkdir(parents=True)

	return shape_config, saving_path


def temperature_sampling(logits, temperature=1.0):
	adjusted_logits = logits / temperature
	probabilities = F.softmax(adjusted_logits, dim=-1)
	sampled_index = torch.multinomial(probabilities, num_samples=1)
	return sampled_index

def multiple_garments_offset(data_path, num):

	offset = num * 5. + 5.

	with open(data_path, 'r') as f:
		jsondata = json.load(f)

	with open(data_path[:-5] + '_backup.json', 'w') as f: # for backup
		json.dump(jsondata, f)

	for key in jsondata["pattern"]["panels"].keys():
		v = jsondata["pattern"]["panels"][key]["translation"][2]
		jsondata["pattern"]["panels"][key]["translation"][2] += offset * v / np.abs(v)

	with open(data_path, 'w') as f:
		json.dump(jsondata, f)



class infer():
	
	def __init__(self, shape_config_path):
		self.system_info = customconfig.Properties('./system.json')
		self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
		shape_config, save_to = get_values_from_args(shape_config_path)

		# --------------- Experiment to evaluate on ---------
				
		shape_experiment = ExperimentWrappper(shape_config, self.system_info['wandb_username'])
		if not shape_experiment.is_finished():
			print('Warning::Evaluating unfinished experiment')

		# data stats from training 
		_, _, self.data_config = shape_experiment.data_info()  # need to get data stats

		
		self.CLIP_embedding = blocks.StableDiffusion(torch.device('cuda'), False, True)
		self.output_folder = save_to

		
		self.shape_model = shape_experiment.load_model().to(self.device)
		self.shape_model.eval()

		self.num = 0


	def obj_resize(self, obj_file_path, scale_factor_v = 1. / 100, scale_factor_vt = 1. / 10):
		with open(obj_file_path, 'r') as obj_file:
			lines = obj_file.readlines()

		scaled_lines = []
		for line in lines:
			if line.startswith('v '):
				vertices = line.split()[1:]
				scaled_vertices = [str(float(vertex) * scale_factor_v) for vertex in vertices]
				scaled_line = 'v ' + ' '.join(scaled_vertices) + '\n'
				scaled_lines.append(scaled_line)
			else:
				scaled_lines.append(line)

		with open(obj_file_path, 'w') as obj_file:
			obj_file.writelines(scaled_lines)

	def forward(self, caption, temperature = 1.0, num = None, sim = True):
		captions = [caption]
		for caption in captions:
			captions_batch = [caption]

			save_to = Path(os.path.join(self.output_folder, str(self.num).zfill(3)))
			os.makedirs(save_to, exist_ok=True)
			self.num += 1

			# ----- Model (Pattern Shape) architecture -----

			batch_size = len(captions_batch)

			prompts_batch = []
			for i in range(batch_size):
				prompts_batch.append(self.CLIP_embedding.get_text_embeds(captions_batch[i])[0])
			prompts_batch = torch.stack(prompts_batch)
			
			# -------- Predict Shape ---------
			with torch.no_grad():
				indices_value_ar = torch.tensor([SOS]).to(self.device)[None]
				indices_axis_ar = torch.tensor([0]).to(self.device)[None]
				indices_pos_ar = torch.tensor([0]).to(self.device)[None]

				outlines     = torch.zeros((1, 0, 14, 4)).to(self.device)
				rotations    = torch.zeros((1, 0, 4)).to(self.device)
				translations = torch.zeros((1, 0, 3)).to(self.device)
				stitch_tags  = torch.zeros((1, 0, 14, 3)).to(self.device)
				free_edges_mask = torch.zeros((1, 0, 14)).to(self.device)
				
				with torch.no_grad():
					CLIP_feature  = self.shape_model.proj_feature_txt(prompts_batch)

				for i in tqdm(range(1500)):
					with torch.no_grad():
						logits = self.shape_model(indices_value_ar, indices_axis_ar, indices_pos_ar, CLIP_feature.clone())

					next_token = temperature_sampling(logits[0, -1:], temperature=temperature)

					indices_value_ar = torch.cat([indices_value_ar, next_token], dim=1)

					next_axis = torch.tensor([i % 119 + 1]).to(self.device)
					indices_axis_ar = torch.cat([indices_axis_ar, next_axis[None]], dim=1)

					next_pos = torch.tensor([i // 119 + 1]).to(self.device)
					indices_pos_ar = torch.cat([indices_pos_ar, next_pos[None]], dim=1)

					if next_token == EOS:
						break 

				indices_value_ar = indices_value_ar[:, 1:]
				
				j = 0

				all_num = indices_value_ar.shape[1] // 119
				for i in range(all_num):
					outlines = torch.cat((outlines, (indices_value_ar[0, j:j + 14 * 4].reshape(1, 1, 14, 4) - (SOS - 1) // 2) / C_outline), dim = 1)
					j = j + 14 * 4
					rotations = torch.cat((rotations, (indices_value_ar[0, j:j + 4].reshape(1, 1, 4) - (SOS - 1) // 2) / C_rotation), dim = 1)
					j = j + 4
					translations = torch.cat((translations, (indices_value_ar[0, j:j + 3].reshape(1, 1, 3) - (SOS - 1) // 2) / C_transl), dim = 1)
					j = j + 3
					stitch_tags = torch.cat((stitch_tags, (indices_value_ar[0, j:j + 14 * 3].reshape(1, 1, 14, 3) - (SOS - 1) // 2) / C_stitch_tag), dim = 1)
					j = j + 14 * 3
					free_edges_mask = torch.cat((free_edges_mask, (indices_value_ar[0, j:j + 14].reshape(1, 1, 14) - (SOS - 1) // 2)), dim = 1)
					j = j + 14
				
				predictions = {"outlines": outlines, "rotations": rotations, "translations": translations, "stitch_tags": stitch_tags, "free_edges_mask": free_edges_mask}
				
			# ---- save shapes ----
			saving_path = save_to
			saving_path.mkdir(parents=True, exist_ok=True)
			names = None
			data.save_garments_prediction(
				predictions, saving_path, self.data_config, names,
				stitches_from_stitch_tags=True)

			with open(saving_path / "caption.txt", "w") as file:
				file.write(caption)

			print(f'Pattern shape saved to {saving_path}')

			data_path = os.path.join(saving_path, 'pred_0', 'specification.json')

			if sim:
				
				if num is not None: multiple_garments_offset(data_path, num) # Add offsets for multiple garments simulation to avoid overlap

				mayapy_executable = self.system_info["maya_path"]
				script_path = './nn/simulation/sim.py'          

				command = f'"{mayapy_executable}" "{script_path}" --data "{data_path}"'

				process = subprocess.Popen(command)
				process.wait()

				obj_path = os.path.join(saving_path, 'pred_0', 'pred_0_sim.obj')
				if os.path.exists(obj_path):
					self.obj_resize(obj_path)
				else:
					self.num -= 1
					print('[Generation Error!! Re-generating...]')
					obj_path = self.forward(caption, temperature)

				return obj_path

			return data_path


if __name__ == "__main__":

	parser = argparse.ArgumentParser()
	parser.add_argument('-c', '--config', help='YAML configuration file', type=str, default='./models/infer.yaml')
	parser.add_argument('-s', '--sim', help='Simulation the results', action='store_true')	
	args = parser.parse_args()

	Pattern_infer = infer(args.config)
	system_info = customconfig.Properties('./system.json')

	prompt_list = ["dress, sleeveless", "jacket, cropped length", "pants"]
	for prompt in prompt_list:
		
		temp_obj_path = system_info["human_obj_path"]
		filename = system_info["sim_json_path"]

		with open(filename, 'r') as f:
			jsondata = json.load(f)

		jsondata['bodies_path'] = temp_obj_path

		with open(filename, 'w') as f:
			json.dump(jsondata, f)

		Pattern_infer.forward(prompt, temperature = 0.7, sim = args.sim)