import numpy as np
import pandas as pd
from definitions import REPO_ROOT_PATH


def get_y_ohe_class_names():
    """returns a list with the target variable class names
    in an order that is consistent every time the function is called."""

    # set filepath for file that contains list of class names
    filepath = REPO_ROOT_PATH + "data/processed/y_ohe_class_names.txt"

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
    on the train data pickle file and returns (X, y_ohe, df)."""

    filepath = REPO_ROOT_PATH + "data/processed/train_data/train_data.pickle"

    X, y_ohe, df = get_input_data_from_pickle_file(filepath)

    return (X, y_ohe, df)


def load_validation_data():
    """This function calls get_input_data_from_pickle_file()
    on the validation data pickle file and returns (X, y_ohe, df)."""

    filepath = REPO_ROOT_PATH \
            + "data/processed/validation_data/validation_data.pickle"

    X, y_ohe, df = get_input_data_from_pickle_file(filepath)

    return (X, y_ohe, df)


def load_test_data():
    """This function calls get_input_data_from_pickle_file()
    on the test data pickle file and returns (X, y_ohe, df)."""

    filepath = REPO_ROOT_PATH + "data/processed/test_data/test_data.pickle"

    X, y_ohe, df = get_input_data_from_pickle_file(filepath)

    return (X, y_ohe, df)
