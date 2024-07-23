#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

# python full_eval.py -m360 <mipnerf360 folder> -tat <tanks and temples folder> -db <deep blending folder>

import os
from argparse import ArgumentParser

blender_scenes = ["chair", "drums", "lego", "mic", "materials", "ship", "hotdog", "ficus"]
#shiny_blender_scenes = ["car", "ball", "helmet", "teapot", "toaster", "coffee"]

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--output_path", default="./wbg_eval_lpgm_blender")
args, _ = parser.parse_known_args()

all_scenes = []
all_scenes.extend(blender_scenes)
#all_scenes.extend(shiny_blender_scenes)


if not args.skip_training or not args.skip_rendering:
    parser.add_argument('--blender', "-blender", type=str,default="./data/nerf_synthetic" )
    #parser.add_argument("--shinyblender", "-shiny", required=True, type=str, default='./data/shiny')
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval -w --blender_bool " #--test_iterations -1 "
    for scene in blender_scenes:
        source = args.blender + "/" + scene
        os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)
    #for scene in shiny_blender_scenes:
    #    source = args.shinyblender + "/" + scene
    #    os.system("python train.py -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_rendering:
    all_sources = []
    for scene in blender_scenes:
        all_sources.append(args.blender + "/" + scene)
    #for scene in shiny_blender_scenes:
    #    all_sources.append(args.shinyblender + "/" + scene)

    common_args = " --quiet --eval -w --blender_bool --skip_train"
    for scene, source in zip(all_scenes, all_sources):
        os.system("python render.py --iteration 24000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)
        os.system("python render.py --iteration 30000 -s " + source + " -m " + args.output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene in all_scenes:
        scenes_string += "\"" + args.output_path + "/" + scene + "\" "

    os.system("python metrics.py -m " + scenes_string)
