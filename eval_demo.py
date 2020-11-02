import gflags
import sys
import os

import numpy as np

from ast import literal_eval

import load_params


argv = gflags.FLAGS(sys.argv)

test_save_dir = os.path.join(gflags.FLAGS.save_data_path, 'test_results/results.txt')
a_file = open(test_save_dir)

list_of_lists = []
for line in a_file:
  stripped_line = line.strip()
  line_list = stripped_line.split()
  list_of_lists.append(line_list)

a_file.close()

lengths = []
replans = []
for line in list_of_lists:
    l1 = literal_eval(line[2])
    l2 = literal_eval(line[3])

    lengths.append(l1)
    replans.append(l2)

#idx where no paths are found
no_path = np.where(np.asarray(lengths) == 0)[0]
replans = np.asarray(replans)

if len(no_path) > 0:
    replans_np = replans[no_path] #replans where no paths are found
    print("replans with no path found: mean (std)", np.average(replans_np), np.std(replans_np))
else:
    print("no trials with no path")

lengths_path = np.delete(lengths, no_path)
replans_path = np.delete(replans, no_path)

print("path length: mean (std)", np.average(lengths_path), np.std(lengths_path))
print("replans: mean (std)", np.average(replans_path), np.std(replans_path))
