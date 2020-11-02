import cv2
import numpy as np
import random
from ast import literal_eval
import os

import load_params_demo
import utils_demo
from astar_mod_demo import astar_map
import astar_mod_demo


#colors
blue = [255, 0, 0]
red = [0, 0, 255]
green = [0, 255, 0]
white = [255, 255, 255]
black = [0, 0, 0]


def add_path_only(path, mymap):

    blue = [255, 0, 0]
    white = [255, 255, 255]
    green = [0, 255, 0]

    for p in path:  # write path to all_obs
        if mymap[p[1], p[0], :].tolist() == white:
            mymap[p[1], p[0], :] = blue
        else:
            mymap[p[1], p[0], :] = green

    return mymap


def get_locs_gauss(n_trials, start_cent, start_sigma, width, height, resized_obs_in, resized_obs_out, resized_obs_res):

    trial = 0

    start_locs = []
    goal_locs = []
    while trial < n_trials:

        start_loc = (
        round(np.random.normal(start_cent[0], start_sigma)), round(np.random.normal(start_cent[1], start_sigma)))
        goal_loc = (random.randint(0, width - 1), random.randint(0, height - 1))  # (column, row)

        if start_loc[0] < 0 or start_loc[1] < 0:  # make sure start_loc and goal is positive
            continue
        if start_loc == goal_loc:  # if it samples same start and goal location
            continue

        bool1 = utils_demo.check_astar_loc(start_loc, goal_loc, astar_map(resized_obs_in))
        bool2 = utils_demo.check_astar_loc(start_loc, goal_loc, astar_map(resized_obs_out))
        bool3 = utils_demo.check_astar_loc(start_loc, goal_loc, astar_map(resized_obs_res))

        if bool1 + bool2 + bool3 != 3:  # if not all three conditions are true
            continue

        start_locs.append(start_loc)
        goal_locs.append(goal_loc)

        trial += 1

    return start_locs, goal_locs


def get_locs(n_trials, width, height, resized_obs_in, resized_obs_out, resized_obs_res):

    trial = 0

    start_locs = []
    goal_locs = []
    while trial < n_trials:

        start_loc = (random.randint(0, width - 1), random.randint(0, height - 1))  # (column, row)
        goal_loc = (random.randint(0, width - 1), random.randint(0, height - 1))  # (column, row)

        if start_loc == goal_loc:  # if it samples same start and goal location
            continue

        bool1 = utils_demo.check_astar_loc(start_loc, goal_loc, astar_map(resized_obs_in))
        bool2 = utils_demo.check_astar_loc(start_loc, goal_loc, astar_map(resized_obs_out))
        bool3 = utils_demo.check_astar_loc(start_loc, goal_loc, astar_map(resized_obs_res))

        if bool1 + bool2 + bool3 != 3:  # if not all three conditions are true
            continue

        start_locs.append(start_loc)
        goal_locs.append(goal_loc)

        trial += 1

    return start_locs, goal_locs


def run_trials(start_locs, goal_locs, obs_map, test_save_dir, output_dir, resize_params, time_step, seq, gt_map):

    trial = 0
    radius = 2
    thickness = 1

    my_map = astar_map(obs_map.copy())  # here using the resized_obs_res map

    # myfile = open(test_save_dir + r'\results.txt', 'w')  # where to write data
    myfile = open(os.path.join(test_save_dir, 'results.txt'), 'w')
    myfile.close()

    for start_loc, goal_loc in zip(start_locs, goal_locs):

        print("trial number ", trial)

        # initial astar path
        path = astar_mod_demo.astar(my_map, start_loc, goal_loc)

        if len(path) == 0:  # if no path can be found
            times_replan = 0

            # with open(test_save_dir + r'\results.txt', 'a') as file:
            with open(os.path.join(test_save_dir, 'results.txt'), 'a') as file:
                file.write(
                    "({},{}) ({},{}) {} {}\n".format(start_loc[0], start_loc[1], goal_loc[0], goal_loc[1], len(path),
                                                     times_replan))
            file.close()

            write_img = cv2.circle(obs_map.copy(), start_loc, radius, blue, thickness)
            write_img = cv2.circle(write_img, goal_loc, radius, blue, thickness)

            # cv2.imwrite(test_save_dir + r'\trial_{}.png'.format(trial), write_img)
            cv2.imwrite(os.path.join(test_save_dir, 'trial_{}.png'.format(trial)), write_img)

            trial += 1
            continue

        all_obs = obs_map.copy()  # initialize all_obs

        times_replan, real_path, all_obs = astar_mod_demo.replan_loop(all_obs, path, goal_loc, resize_params, time_step, seq,
                                                                 test_save_dir, output_dir, gt_map)  # run replanner loop
        print("real_path", real_path)
        # with open(test_save_dir + r'\results.txt', 'a') as file:
        with open(os.path.join(test_save_dir, 'results.txt'), 'a') as file:
            file.write(
                "({},{}) ({},{}) {} {}\n".format(start_loc[0], start_loc[1], goal_loc[0], goal_loc[1], len(real_path),
                                                 times_replan))
        file.close()

        write_img = add_path_only(real_path, all_obs.copy())
        write_img = cv2.circle(write_img, start_loc, radius, blue, thickness)
        write_img = cv2.circle(write_img, goal_loc, radius, blue, thickness)

        # cv2.imwrite(test_save_dir + r'\trial_{}.png'.format(trial), write_img)
        cv2.imwrite(os.path.join(test_save_dir, 'trial_{}.png'.format(trial)), write_img)

        trial += 1


def GT_trials(start_locs, goal_locs, test_save_dir, obs_map):

    trial = 0
    radius = 2
    thickness = 1

    my_map = astar_map(obs_map.copy())

    # myfile = open(test_save_dir + r'\results.txt', 'w')  # where to write data
    myfile = open(os.path.join(test_save_dir, 'results.txt'), 'w')
    myfile.close()

    for i in range(len(start_locs)):
        start_loc = start_locs[i]
        goal_loc = goal_locs[i]

        print("trial number ", trial)

        path = astar_mod_demo.astar(my_map, start_loc, goal_loc)

        times_replan = 0

        # with open(test_save_dir + r'\results.txt', 'a') as file:
        with open(os.path.join(test_save_dir, 'results.txt'), 'a') as file:
            file.write("({},{}) ({},{}) {} {}\n".format(start_loc[0], start_loc[1], goal_loc[0], goal_loc[1], len(path),
                                                        times_replan))
        file.close()

        write_img = add_path_only(path, obs_map.copy())
        write_img = cv2.circle(write_img, start_loc, radius, blue, thickness)
        write_img = cv2.circle(write_img, goal_loc, radius, blue, thickness)

        # cv2.imwrite(test_save_dir + r'\trial_{}.png'.format(trial), write_img)
        cv2.imwrite(os.path.join(test_save_dir, 'trial_{}.png'.format(trial)), write_img)

        trial += 1


def locs_from_file(file):
    list_of_lists = []
    for line in file:
        stripped_line = line.strip()
        line_list = stripped_line.split()
        list_of_lists.append(line_list)

    file.close()

    start_locs = []
    goal_locs = []
    for line in list_of_lists:
        l1 = literal_eval(line[0])
        l2 = literal_eval(line[1])
        start_locs.append(l1)
        goal_locs.append(l2)

    return start_locs, goal_locs


def vis_locs(start_locs, goal_locs, map):

    for start_loc, goal_loc in zip(start_locs, goal_locs):
        map[start_loc[1], start_loc[0], :] = red
        map[goal_loc[1], goal_loc[0], :] = green

    cv2.imshow("Start and goal locations", map)
    cv2.waitKey(0)
