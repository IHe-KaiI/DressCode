import argparse
from datetime import datetime
import numpy as np
from pathlib import Path
import torch
import traceback
import yaml
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import json
from glob import glob

import subprocess

# Do avoid a need for changing Evironmental Variables outside of this script
import os, sys

from tqdm import tqdm
import potpourri3d as pp3d

currentdir = os.path.dirname(os.path.realpath(__file__))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 

# My modules
import customconfig
import data
from experiment import ExperimentWrappper
from pattern.wrappers import VisPattern
import net_blocks as blocks

system_info = customconfig.Properties('./system.json')

def sim_single(data_path, c = 0):
    mayapy_executable = system_info["maya_path"]
    script_path = './nn/simulation/sim.py'

    command = f'"{mayapy_executable}" "{script_path}" --data "{data_path}"'

    process = subprocess.Popen(command)
    process.wait()
    
    obj_path = os.path.join(os.path.dirname(data_path), 'pred_0_sim.obj')
    
    V, F = pp3d.read_mesh(obj_path)
    V /= 100.

    pp3d.write_mesh(V, F, obj_path[:-4] + '_raw.obj')
    return obj_path[:-4] + '_raw.obj'

def vis_single(data_path, c = 0):
    mayapy_executable = system_info["maya_path"]
    script_path = './nn/simulation/sim_no_stitch.py'

    command = f'"{mayapy_executable}" "{script_path}" --data "{data_path}"'

    process = subprocess.Popen(command)
    process.wait()
    
    obj_path = os.path.join(os.path.dirname(data_path), 'specification_no_stitch_sim.obj')
    
    V, F = pp3d.read_mesh(obj_path)
    V[..., 2] += 5. * c * V[..., 2] / np.abs(V[..., 2])
    pp3d.write_mesh(V, F, obj_path)
    return obj_path

def merge_objs(file1, file2, output_file):
    vertices = []
    faces = []

    with open(file1, 'r') as f:
        for line in f:
            elements = line.split()
            if len(elements) > 0:
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


def vis(dir_path, sim = False, single_garment=False):
    
    if single_garment: paths = sorted(glob(os.path.join(dir_path, '*', '*', 'specification.json')))
    else: paths = sorted(glob(os.path.join(dir_path, '*', '*', '*', 'specification.json')))

    combined_path = os.path.join(dir_path, 'vis_combined.obj')
    combined_path_sim = os.path.join(dir_path, 'sim_combined.obj')

    c = 0
    temp_obj_path = system_info["human_obj_path"]
    for i, path in enumerate(paths):
        obj_path = vis_single(path, c = c)
        c = c + 1

        if sim:
            filename = system_info["sim_json_path"]
            with open(filename, 'r') as f:
                data = json.load(f)

            data['bodies_path'] = temp_obj_path

            with open(filename, 'w') as f:
                json.dump(data, f)

            obj_path_sim = sim_single(path)

            merge_objs(obj_path_sim, temp_obj_path, combined_path_sim)
            temp_obj_path = combined_path_sim


        if i == 0: 
            print('copy ' + obj_path + ' ' + combined_path)
            os.system('copy ' + obj_path + ' ' + combined_path)
        else:
            print('combined_path = ', combined_path, ' obj_path = ', obj_path)
            merge_objs(combined_path, obj_path, combined_path)


if __name__ == "__main__":

    vis(r'D:\hekai\multiple_garments', sim=True, single_garment=False)