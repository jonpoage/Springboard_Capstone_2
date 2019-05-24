import numpy as np
import seaborn as sns
import load_processed_data as ld
import utils_model_evaluation as ume
import os
from definitions import MODELS_PATH, MODEL_NAMES, FIGURE_OUTPUT_PATH
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input

# adjust seaborn font scale
sns.set(font_scale=1.2)

# load test data
X, y_ohe, _ = ld.load_test_data()

# create a 1-D array of class labels encoded as integers
# NOTE: this implementation assumes y_ohe is correct without checking
y_true = np.argmax(y_ohe,
                   axis=1)

# get list of class labels
y_labels = ld.get_y_ohe_class_names()

# iterate over model names
for k, mn in MODEL_NAMES.items():
    # initialize model as a precaution
    m = None

    # load the model
    model_file_path = MODELS_PATH + mn + '.h5'
    m = load_model(model_file_path)

    # get model predictions
    if k == 1:
        y_pred = m.predict_classes(X/255.0)
    elif k in [2, 3]:
        y_pred = m.predict_classes(preprocess_input(X))
    else:
        raise ValueError('Invalid model key')

    # get confusion matrix
    df_cm = ume.get_confusion_matrix_df(y_true=y_true,
                                        y_pred=y_pred,
                                        y_labels=y_labels)

    # make heatmap of confusion matrix
    fig_cm = ume.get_confusion_matrix_heatmap_figure(df_cm)

    # save the heatmap figure
    heatmap_file_path = (FIGURE_OUTPUT_PATH
                         + mn
                         + '_confusion_matrix_heatmap.png')
    if os.path.exists(heatmap_file_path):
        os.remove(heatmap_file_path)
    fig_cm.savefig(heatmap_file_path,
                   bbox_inches='tight')

    # get classification report
    df_cr = ume.get_classification_report_df(y_true=y_true,
                                             y_pred=y_pred,
                                             y_labels=y_labels)

    # make table from classification report
    fig_cr = ume.get_classification_report_figure(df_cr, len(y_labels))

    # save the table figure
    table_file_path = FIGURE_OUTPUT_PATH + mn + '_classification_report.png'
    if os.path.exists(table_file_path):
        os.remove(table_file_path)
    fig_cr.savefig(table_file_path,
                   bbox_inches='tight')
