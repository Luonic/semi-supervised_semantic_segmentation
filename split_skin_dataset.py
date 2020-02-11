import os
from glob import glob
from random import shuffle
import json
test_size = 0.4

dataset_dir = '/home/alex/Code/instascraped/dataset_1'
part_config_path = os.path.join(dataset_dir, 'part_config.json')

dir_names = [os.path.split(filename)[-1] for filename in glob(os.path.join(dataset_dir, '*')) if os.path.isdir(filename)]
shuffle(dir_names)

partitions = []
partitions.append(dir_names[0:int(len(dir_names) * test_size)])
partitions.append(dir_names[int(len(dir_names) * test_size):])

with open(part_config_path, 'w') as f:
    json.dump(partitions, f, sort_keys=True, allow_nan=False, indent=4)