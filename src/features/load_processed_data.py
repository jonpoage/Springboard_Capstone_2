from glob import glob
import numpy as np
import pandas as pd

def get_repo_root_path():
	"""returns the root path of the repository as a string"""
	
	# NOTE: The root repository path below is set for my machine!
	#       Users may need to update the path before running the code.
	
	repo_root_path = "C:/Users/Jon/Springboard/Capstone_2_repo/"
	return repo_root_path

def get_input_data():
	###############################################################################
	# Set up root path for repository
	###############################################################################
	print("defining pathnames")

	# set path for repository root directory
	repo_root_path = get_repo_root_path()

	###############################################################################
	# Load processed data into dataframes
	###############################################################################
	print("loading data")

	# load validation and test data
	df_val = pd.read_pickle(repo_root_path
	                    + "data/processed/validation_data/validation_data.pickle")
	df_test = pd.read_pickle(repo_root_path
	                    + "data/processed/test_data/test_data.pickle")

	# load the train data in parts
	train_data_files = glob(repo_root_path
	                    + "data/processed/train_data/train_data_part*.pickle")
	
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

	print("returning input data")
	
	return (df_train, df_val, df_test,
	        X_train, X_val, X_test,
			y_train_ohe, y_val_ohe, y_test_ohe,
			y_ohe_classes)