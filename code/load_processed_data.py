import numpy as np
import pandas as pd


def get_repo_root_path():
    """returns the root path of the repository as a string."""

    # NOTE: The root repository path below is set for my machine!
    #       Users may need to update the path before running the code.

    repo_root_path = "C:/Users/Jon/Springboard/Capstone_2_repo/"

    return repo_root_path


def get_y_ohe_class_names():
    """returns a list with the target variable class names
    in an order that is consistent every time the function is called."""

    # set filepath for file that contains list of class names
    repo_root_path = get_repo_root_path()
    filepath = repo_root_path + "data/processed/y_ohe_class_names.txt"

    # load the class names into a list
    class_list = []

    with open(filepath, "r") as f:
        for line in f:
            class_list.append(line.strip())

    return class_list


def get_input_data_from_pickle_file(filepath):
    """This function takes a filepath string as input.
    It returns a dataframe with image file data,
    a 4-D numpy array with pixel intensity values,
    and an array of one hot encoded target variable values.
    The columns of the returned target variable array are sorted
    according to the order of classes in the class list text file."""

    # dataframe with image file data
    df = pd.read_pickle(filepath)

    # 4-D array of pixel intensity values
    X = np.array(df.pixel_array.tolist())

    # one hot encoded target variable array
    y_ohe_df = pd.get_dummies(df.image_class)
    class_order = get_y_ohe_class_names()
    y_ohe = y_ohe_df[class_order].values

    return (X, y_ohe, df)


def load_train_data():
    """This function calls get_input_data_from_pickle_file()
    on the train data pickle file and returns (X, y_ohe, df)"""

    repo_root_path = get_repo_root_path()
    filepath = repo_root_path + "data/processed/train_data/train_data.pickle"

    X, y_ohe, df = get_input_data_from_pickle_file(filepath)

    return (X, y_ohe, df)


def load_validation_data():
    """This function calls get_input_data_from_pickle_file()
    on the validation data pickle file and returns (X, y_ohe, df)."""

    repo_root_path = get_repo_root_path()
    filepath = repo_root_path \
            + "data/processed/validation_data/validation_data.pickle"

    X, y_ohe, df = get_input_data_from_pickle_file(filepath)

    return (X, y_ohe, df)


def load_test_data():
    """This function calls get_input_data_from_pickle_file()
    on the test data pickle file and returns (X, y_ohe, df)."""

    repo_root_path = get_repo_root_path()
    filepath = repo_root_path + "data/processed/test_data/test_data.pickle"

    X, y_ohe, df = get_input_data_from_pickle_file(filepath)

    return (X, y_ohe, df)
