import gflags

#save data
gflags.DEFINE_string('save_data_path', '', 'where to save generated data')
#yaml config file
gflags.DEFINE_string('labels_yaml', 'semantic-kitti.yaml', 'yaml file with labels and colormap')