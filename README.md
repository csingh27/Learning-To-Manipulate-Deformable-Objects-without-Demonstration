<img align="right" width="205" height="109" src="/images/bair_logo.png">

# Learning to Manipulate Deformable Objects without Demonstrations

This is the code that corresponds to the [paper](https://arxiv.org/abs/1910.13439)

## rlpyt Usage
See the [original library](https://github.com/astooke/rlpyt) for more information on the design of the library

## Installation

1) Install Mujoco

• Register for license on https://www.roboti.us/license.html
• conda create -n Masterarbeit python=3.7
• conda activate Masterarbeit
• create /.mujoco folder
• wget https://www.roboti.us/download/mujoco200_linux.zip
• unzip mujoco200_linux.zip
• Add license.txt and mjkey.txt to .mujoco/mucoco200_linux/bin folder and also separately to .mujoco/ folder (as mujoco-py and dm_control look in /mujoco/) (IMPORTANT !)
• cp mjkey.txt /home/chandandeep/.mujoco/mujoco200_linux/bin/
• cp LICENSE.txt /home/chandandeep/.mujoco/mujoco200_linux/bin/
• Run this “ ./simulate ../model/humanoid.xml ” from “.mujocomucoco200_linux/bin ” to test if mujoco is successful

2) Install mujoco-py with mujoco200

• pip3 install mujoco-py==2.0.2.8
• export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/chandandeep/.mujoco/mujoco200_linux/bin
• export MUJOCO_PY_MUJOCO_PATH=/home/chandandeep/.mujoco/mujoco200_linux/
• export MUJOCO_PY_MJKEY_PATH=/home/chandandeep/.mujoco/mjkey.txt
• sudo apt install libosmesa6-dev libgl1-mesa-glx libglfw3      
• sudo apt install patchelf
• Make a python file trial.py with the following contents : 
'''
*import mujoco_py
import os
mj_path, _ = mujoco_py.utils.discover_mujoco()
xml_path = os.path.join(mj_path, 'model', 'humanoid.xml')
model = mujoco_py.load_model_from_path(xml_path)
sim = mujoco_py.MjSim(model)
print(sim.data.qpos)
# [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
sim.step()
print(sim.data.qpos)
# [-2.09531783e-19  2.72130735e-05  6.14480786e-22 -3.45474715e-06
#   7.42993721e-06 -1.40711141e-04 -3.04253586e-04 -2.07559344e-04
#   8.50646247e-05 -3.45474715e-06  7.42993721e-06 -1.40711141e-04
#  -3.04253586e-04 -2.07559344e-04 -8.50646247e-05  1.11317030e-04
#  -7.03465386e-05 -2.22862221e-05 -1.11317030e-04  7.03465386e-05
#  -2.22862221e-05]*
'''

3) Install a custom version of [dm_control](https://github.com/wilson1yan/dm_control)

4) Install a custom version of [dm_env](https://github.com/wilson1yan/dm_env)

• git clone https://github.com/wilson1yan/dm_control.git
• sudo apt-get install python3-setuptools
• sudo python3 setup.py install
• sudo ln -s /usr/lib/x86_64-linux-gnu/libGL.so.1 /usr/lib/x86_64-linux-gnu/libGL.so
• sudo pip3 install -r requirements.txt
• pip3 install dm-control
• export MJLIB_PATH=/home/chandandeep/.mujoco/mujoco200_linux/bin/libmujoco200.so
• git clone https://github.com/wilson1yan/dm_env.git
• sudo python3 setup.py install
• pip3 install dm_env

Install the original rlpyt environment

## Running

All launch scripts are in rlpyt/experiments/scripts/dm_control/qpg/sac/launch

### Cloth

For Cloth (State), see launch_dm_control_sac_state_cloth_point.py

For Cloth (Pixel), see launch_dm_control_sac_pixels_cloth_point.py

For Cloth-Simplified (State), see launch_dm_control_sac_state_cloth_corner.py

### Rope

For Rope (State), see launch_dm_control_state_rope.py

For Rope (Pixel), see launch_dm_control_pixels_rope.py
