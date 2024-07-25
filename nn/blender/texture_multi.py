import bpy
import os
import math
import sys
import mathutils
import json

bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

json_path = sys.argv[-1]

with open(json_path, "r") as f:
    data = json.load(f)

output_path = data["output"]
objs = data["obj"]
textures = data['texture']

sys_json_path = './system.json'
with open(sys_json_path, "r") as f:
    system_info = json.load(f)


input_human_obj_path = system_info["human_obj_path"]
bpy.ops.import_scene.obj(filepath=input_human_obj_path)

for i in range(len(objs)):

    input_obj_path = objs[i]
    texture_path = textures[i]
    
    bpy.ops.image.open(filepath=texture_path)

    diffuse_map_path = texture_path
    normal_map_path = texture_path[:-12] + '_normal.png'
    roughness_map_path = texture_path[:-12] + '_roughness.png'

    bpy.ops.import_scene.obj(filepath=input_obj_path)
    model = bpy.context.selected_objects[-1]

    imported_object_name = bpy.context.selected_objects[-1].name
    mesh_name = bpy.context.selected_objects[-1].data.name

    bpy.context.view_layer.objects.active = bpy.data.objects[imported_object_name]

    material = bpy.data.materials.new(name=imported_object_name + "Material")
    model.data.materials[0] = material

    material.use_nodes = True
    nodes = material.node_tree.nodes

    bsdf = material.node_tree.nodes.get("Principled BSDF")

    def create_image_texture_node(image_path, node_tree, label, color_space='sRGB'):
        tex_image_node = node_tree.nodes.new('ShaderNodeTexImage')
        tex_image_node.image = bpy.data.images.load(image_path)
        tex_image_node.label = label
        tex_image_node.image.colorspace_settings.name = color_space
        return tex_image_node


    diffuse_tex_node = create_image_texture_node(diffuse_map_path, material.node_tree, "Diffuse Map", 'sRGB')
    material.node_tree.links.new(bsdf.inputs['Base Color'], diffuse_tex_node.outputs['Color'])

    normal_map_tex_node = create_image_texture_node(normal_map_path, material.node_tree, "Normal Map", 'Non-Color')
    normal_map_node = material.node_tree.nodes.new('ShaderNodeNormalMap')
    material.node_tree.links.new(normal_map_node.inputs['Color'], normal_map_tex_node.outputs['Color'])
    material.node_tree.links.new(bsdf.inputs['Normal'], normal_map_node.outputs['Normal'])


    roughness_tex_node = create_image_texture_node(roughness_map_path, material.node_tree, "Roughness Map", 'Non-Color')
    material.node_tree.links.new(bsdf.inputs['Roughness'], roughness_tex_node.outputs['Color'])



bpy.ops.object.mode_set(mode="VERTEX_PAINT")
bpy.ops.paint.vertex_paint_toggle()
bpy.ops.object.mode_set(mode="OBJECT")


def eraseAllKeyframes(scene,passedOB = None):

    if passedOB != None:

        ad = passedOB.animation_data

        if ad != None:
            print ('ad=',ad)
            passedOB.animation_data_clear()

            #scene.update()


def look_at(obj_camera, point):
    loc_camera = obj_camera.matrix_world.to_translation()
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    obj_camera.rotation_euler = rot_quat.to_euler()


bpy.ops.object.camera_add(location=(3, 0, 1))
camera = bpy.context.object
camera.data.lens = 35
bpy.context.scene.camera = camera

scene = bpy.data.scenes["Scene"]

eraseAllKeyframes(scene, passedOB = None)


hdr_path = system_info["HDR_path"]
world = bpy.context.scene.world
world.use_nodes = True
nodes = world.node_tree.nodes

for node in nodes:
    nodes.remove(node)

env_texture = nodes.new('ShaderNodeTexEnvironment')
background = nodes.new('ShaderNodeBackground')
output_node = nodes.new('ShaderNodeOutputWorld')

env_texture.image = bpy.data.images.load(hdr_path)

background.inputs['Strength'].default_value = 0.7 

links = world.node_tree.links
links.new(env_texture.outputs['Color'], background.inputs['Color'])
links.new(background.outputs['Background'], output_node.inputs['Surface'])


cp = mathutils.Vector((0.0, 0.0, 0.0))
cam = camera 

frame_num = 30
scene.frame_start = 1
scene.frame_end = frame_num

radius = 3.
for i in range(frame_num):
    bpy.context.scene.frame_set(i)

    angle = (i / frame_num) * 2 * math.pi
    x = radius * math.cos(angle)
    y = radius * math.sin(angle)
    cam.location = (x, y, 0)
    cam.keyframe_insert('location')


for i in range(frame_num):
    bpy.context.scene.frame_set(i)

    look_at(cam, cp)
    cam.keyframe_insert('rotation_euler')


scene.render.filepath = os.path.join(output_path, 'render.mp4')
scene.render.image_settings.file_format = "FFMPEG"
bpy.context.scene.render.ffmpeg.format = 'MPEG4'
bpy.ops.render.render(animation=True)