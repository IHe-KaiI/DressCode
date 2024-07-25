import argparse
import os
import sys
from importlib import reload
import json

from maya import cmds
import maya.standalone 	

# My modules
import customconfig
# reload in case we are in Maya internal python environment
reload(customconfig)


def get_command_args():
    """command line arguments to control the run"""
    # https://stackoverflow.com/questions/40001892/reading-named-command-arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', '-d', help='name of dataset folder', type=str)
    parser.add_argument('--config', '-c', help='name of .json file with desired simulation&rendering config', type=str, default=None)
    parser.add_argument('--minibatch', '-b', help='number of examples to simulate in this run', type=int, default=None)

    args = parser.parse_args()
    # print(args)

    return args


def init_mayapy():
    try: 
        print('Initilializing Maya tools...')
        maya.standalone.initialize()
        print('Load plugins')
        cmds.loadPlugin('mtoa.mll')  # https://stackoverflow.com/questions/50422566/how-to-register-arnold-render
        cmds.loadPlugin('objExport.mll')  # same as in https://forums.autodesk.com/t5/maya-programming/invalid-file-type-specified-atomimport/td-p/9121166
        
    except Exception as e: 
        print(e)
        print('Init failed')
        pass


def stop_mayapy():  
    maya.standalone.uninitialize() 
    print("Maya stopped")

init_mayapy()
import mayaqltools as mymaya  # has to import after maya is loaded
reload(mymaya)  # reload in case we are in Maya internal python environment

class sim():
    def __init__(self, system_config_path, props_path):
        self.system_config = customconfig.Properties(system_config_path)
        self.props = customconfig.Properties(props_path)

    def simulation_single(self, pattern_path):
        self.system_config['pattern_path'] = pattern_path
        self.props["frozen"] = False
        mymaya.simulation.single_file_sim(self.system_config, self.props, no_stitch=False)

    def empty_stitch(self, path):

        with open(path, 'r') as file:
            json_data = json.load(file)
        
        json_data['pattern']['stitches'] = []

        new_path = path[:-5] + '_no_stitch.json'
        with open(new_path, 'w') as file:
            json.dump(json_data, file)
        
        return new_path

    def simulation_single_no_stitch(self, pattern_path):

        new_path = self.empty_stitch(pattern_path)
        self.system_config['pattern_path'] = new_path
        self.props["frozen"] = False
        mymaya.simulation.single_file_sim(self.system_config, self.props, no_stitch=True)


if __name__ == "__main__":

    command_args = get_command_args()
    system_info = customconfig.Properties('./system.json')

    Sim = sim(system_info['sim_json_path'], system_info['dataset_properties_path'])
    Sim.simulation_single_no_stitch(command_args.data)