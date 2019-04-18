from glob import glob
import numpy as np
import pandas as pd

###############################################################################
# Set up processed input directories
###############################################################################
print("defining pathnames")

# NOTE: The root repository directory is set for my machine!
#       Users may need to update the path below before running the code.
#
# set path for repository root directory
repo_root_path = "C:/Users/Jon/Springboard/Capstone_2_repo/"

# set path for processed data directory
processed_data_path = repo_root_path + "data/processed/"

# make train validation test subdirectories
train_dir = processed_data_path + "train_data/"
validation_dir = processed_data_path + "validation_data/"
test_dir = processed_data_path + "test_data/"

# set image file subdirectory
img_subdir = "image_files/"

###############################################################################
# Load processed data into dataframes
###############################################################################
print("loading data")

# load validation and test data
df_val = pd.read_pickle(validation_dir + "validation_data.pickle")
df_test = pd.read_pickle(test_dir + "test_data.pickle")

# load the train data in parts
train_data_files = glob(train_dir + "train_data_part*.pickle")
df_train_parts = []

for fn in train_data_files:
    df_train_parts.append(pd.read_pickle(fn))

df_train = pd.concat(df_train_parts).sort_index()

###############################################################################
# prepare features and target variable arrays
###############################################################################
print("preparing feature and target variable arrays")

# make 4-D numpy arrays with pixel intensities
X_train = np.array(df_train.pixel_array.tolist())
X_val = np.array(df_val.pixel_array.tolist())
X_test = np.array(df_test.pixel_array.tolist())

# make one hot encoded target variable arrays
y_train_ohe = pd.get_dummies(df_train.image_class).values
y_val_ohe = pd.get_dummies(df_val.image_class).values
y_test_ohe = pd.get_dummies(df_test.image_class).values

# list of image classes
# list indices correspond to indices of target variable arrays
y_ohe_classes = pd.get_dummies(df_test.image_class).columns.tolist()

print("done")
