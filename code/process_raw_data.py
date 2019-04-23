from glob import glob
import numpy as np
import pandas as pd
from keras.preprocessing.image import img_to_array, load_img
import os
import shutil
from definitions import REPO_ROOT_PATH, PROCESSED_DATA_PATH

# set random seed
np.random.seed(10)

###############################################################################
# Set up processed output directories
###############################################################################

# make train validation test subdirectories
train_dir = PROCESSED_DATA_PATH + "train_data/"
validation_dir = PROCESSED_DATA_PATH + "validation_data/"
test_dir = PROCESSED_DATA_PATH + "test_data/"

# if processed data directory exists, remove it
if os.path.exists(PROCESSED_DATA_PATH):
    shutil.rmtree(PROCESSED_DATA_PATH)

# make new processed data directory
os.makedirs(PROCESSED_DATA_PATH)

# make new processed train validation test subdirectories
os.makedirs(train_dir)
os.makedirs(validation_dir)
os.makedirs(test_dir)

###############################################################################
# Separate raw image files into train, validation, and test sets
###############################################################################

# set path for raw data directory
raw_data_path = REPO_ROOT_PATH + "data/raw/"

# load raw test data files
raw_test_files = glob(raw_data_path + "chest_xray/test/*/*.jpeg")

# load raw train data files
raw_train_files = glob(raw_data_path + "chest_xray/train/*/*.jpeg")

# separate raw train files by class
raw_train_bacteria_files = [fn for fn in raw_train_files if 'BACTERIA' in fn]
raw_train_virus_files = [fn for fn in raw_train_files if 'VIRUS' in fn]
raw_train_normal_files = [fn for fn in raw_train_files if 'NORMAL' in fn]

# make list of files in test set
test_files = raw_test_files

# randomly sample image files for validation set
bacteria_val_files = np.random.choice(raw_train_bacteria_files,
                                      size=len(raw_train_bacteria_files) // 4,
                                      replace=False).tolist()

virus_val_files = np.random.choice(raw_train_virus_files,
                                   size=len(raw_train_virus_files) // 4,
                                   replace=False).tolist()

normal_val_files = np.random.choice(raw_train_normal_files,
                                    size=len(raw_train_normal_files) // 4,
                                    replace=False).tolist()

# make lists of files for train set
bacteria_train_files = list(set(raw_train_bacteria_files)
                            - set(bacteria_val_files))
virus_train_files = list(set(raw_train_virus_files) - set(virus_val_files))
normal_train_files = list(set(raw_train_normal_files) - set(normal_val_files))

# concatenate the lists for the train and validation sets
train_files = bacteria_train_files + virus_train_files + normal_train_files
validation_files = bacteria_val_files + virus_val_files + normal_val_files

###############################################################################
# Copy image files into processed train, validation, test subdirectories
###############################################################################

# make image file subdirectories
img_subdir = "image_files/"

os.makedirs(train_dir + img_subdir)
os.makedirs(validation_dir + img_subdir)
os.makedirs(test_dir + img_subdir)

# copy the files to the processed data subdirectories
print("copying image files")

for fn in train_files:
    shutil.copy(fn, train_dir + img_subdir)

for fn in validation_files:
    shutil.copy(fn, validation_dir + img_subdir)

for fn in test_files:
    shutil.copy(fn, test_dir + img_subdir)

###############################################################################
# Process image file data into dataframes
###############################################################################


# function to get class from filename
def get_class_from_filename(filename):
    """Takes a string with the base name of a filepath as input.
    returns the class as a string."""

    if 'NORMAL' in filename:
        str_class = 'normal'
    elif 'BACTERIA' in filename:
        str_class = 'bacterial_pneumonia'
    elif 'VIRUS' in filename:
        str_class = 'viral_pneumonia'
    else:
        raise Exception('Filename not valid!  Class not found.')

    return str_class


# function to make dataframe using list of image filepaths
def df_from_filepath_list(filepath_list, img_size):
    """Takes a list of filepath strings and a tuple of ints
    with image pixel dimensions as input.
    Returns a dataframe with the base path, image class,
    and pixel intensity array for each filepath."""

    list_of_tuples = []

    for fp in filepath_list:
        f_name = os.path.basename(fp)
        f_class = get_class_from_filename(f_name)
        pixel_array = img_to_array(load_img(fp, target_size=img_size))
        list_of_tuples.append((f_name, f_class, pixel_array))

    return pd.DataFrame(list_of_tuples,
                        columns=['image_file_base_path',
                                 'image_class',
                                 'pixel_array_custom_image_size'])


# Set the image size
IMG_SIZE = (224, 224)

# create the train validation test dataframes
print("loading image data into dataframes")

df_train = df_from_filepath_list(train_files, IMG_SIZE)
df_validation = df_from_filepath_list(validation_files, IMG_SIZE)
df_test = df_from_filepath_list(test_files, IMG_SIZE)

###############################################################################
# Save processed data to disk
###############################################################################

print("saving data files")

# save the dataframes to pickle files -- preserves numpy array format
df_validation.to_pickle(validation_dir + "validation_data.pickle")
df_test.to_pickle(test_dir + "test_data.pickle")
df_train.to_pickle(train_dir + "train_data.pickle")

###############################################################################
# Save list of class names
###############################################################################

print("saving class names")

class_list = ['bacterial_pneumonia', 'viral_pneumonia', 'normal']

class_filepath = PROCESSED_DATA_PATH + "y_ohe_class_names.txt"

with open(class_filepath, "w") as f:
    for c in class_list:
        f.write(c + "\n")

print("done")
