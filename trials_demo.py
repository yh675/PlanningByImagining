import gflags
import sys
import os

import cv2

import load_params_demo
import utils_demo
from utils_demo import Resize_params
import test_mod_demo

argv = gflags.FLAGS(sys.argv)

#time step and sequence
time_step = '000150'
seq = '10'

# process map images for astar (obstacle definition)
input_dir = os.path.join(gflags.FLAGS.save_data_path, 'input/')
output_dir = os.path.join(gflags.FLAGS.save_data_path, 'output/')
result_dir = os.path.join(gflags.FLAGS.save_data_path, 'all_tests/')
test_save_dir = os.path.join(gflags.FLAGS.save_data_path, 'test_results')

# maps: RGB
in_img = cv2.imread(os.path.join(input_dir, '{}_s{}_input.png'.format(time_step, seq)))
out_img = cv2.imread(os.path.join(output_dir, '{}_s{}_output.png'.format(time_step, seq)))
res_img = cv2.imread(os.path.join(result_dir, '{}_s{}_test.png'.format(time_step, seq)))

# resizing parameters
scale_percent = 25  # percent of original size
resize_params = Resize_params(scale_percent, in_img)

width = resize_params.dim[0]
height = resize_params.dim[1]

print("resizing images and converting to obstacle map")
# create map and resize
obs_in_img = utils_demo.process_for_astar(in_img.copy())
resized_obs_in = cv2.resize(obs_in_img.copy(), resize_params.dim, interpolation=cv2.INTER_AREA)
obs_out_img = utils_demo.process_for_astar(out_img.copy())
resized_obs_out = cv2.resize(obs_out_img.copy(), resize_params.dim, interpolation=cv2.INTER_AREA)
obs_res_img = utils_demo.process_for_astar(res_img.copy())
resized_obs_res = cv2.resize(obs_res_img.copy(), resize_params.dim, interpolation=cv2.INTER_AREA)

assert resized_obs_in.shape == resized_obs_out.shape == resized_obs_res.shape

print("getting start and goal locations")
#NOTE: this is to read from existing trials file for start and goal location data
# a_file = open(os.path.join(test_save_dir, '../results.txt'), "r")
# start_locs, goal_locs = test_mod_demo.locs_from_file(a_file) #get start and goal locs from file

#NOTE: generate start and goal locations
n_trials = 20
# start_locs, goal_locs = test_mod_demo.get_locs(n_trials, width, height, resized_obs_in.copy(), resized_obs_out.copy(), resized_obs_res.copy())
#
# start_cent = (5, 83)
# start_sigma = 10
# start_locs, goal_locs = test_mod_demo.get_locs_gauss(n_trials, start_cent, start_sigma, width, height,
#                                                 resized_obs_in.copy(), resized_obs_out.copy(), resized_obs_res.copy())

# test_mod_demo.vis_locs(start_locs, goal_locs, resized_obs_res.copy())
# exit(1)

# # NOTE: GT trials
# obs_map = resized_obs_res.copy()
# test_mod_demo.GT_trials(start_locs, goal_locs, test_save_dir, obs_map)
# exit(1)

#NOTE: edit this
# obs_map = resized_obs_res.copy()
# gt_map = resized_obs_out.copy()
# test_mod_demo.run_trials(start_locs, goal_locs, obs_map, test_save_dir, output_dir, resize_params, time_step, seq, gt_map)


