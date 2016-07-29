import os


# Model settings
alpha = 0.1
beta = 0.1
K = 20
iter_max = 500


# Project settings
BASE_DIR = os.path.dirname(__file__)
RES_DIR = os.path.join(BASE_DIR, 'res')

demo_dataset_dir = os.path.join(RES_DIR, 'wiki20')
dict_file = os.path.join(RES_DIR, 'dictionary.txt')
