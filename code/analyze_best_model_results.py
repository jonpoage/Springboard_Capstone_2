import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import load_processed_data as ld
import utils_model_evaluation as ume
import os
from definitions import MODELS_PATH, MODEL_NAMES, FIGURE_OUTPUT_PATH
from keras.models import load_model
from keras.applications.vgg16 import preprocess_input
from sklearn.metrics import roc_auc_score, average_precision_score, \
                            roc_curve, precision_recall_curve

###############################################################################
# Configuration
###############################################################################

# adjust seaborn settings
sns.set(font_scale=1.2,
        style='whitegrid')

# key used to select which model to analyze
model_key = 1

###############################################################################
# Load and preprocess data
###############################################################################

# load test data
X, y_ohe, _ = ld.load_test_data()

# create a 1-D array of class labels encoded as integers
# NOTE: this implementation assumes y_ohe is correct without checking
y_true = np.argmax(y_ohe,
                   axis=1)

# get list of class labels
y_labels = ld.get_y_ohe_class_names()

# load the model
mn = MODEL_NAMES[model_key]
model_file_path = MODELS_PATH + mn + '.h5'
m = load_model(model_file_path)

# preprocess the image data
if model_key in [2, 3]:
    X_preprocessed = preprocess_input(X)
else:
    X_preprocessed = X / 255.0

# get predictions
y_pred = m.predict_classes(X_preprocessed)
y_pred_proba = m.predict_proba(X_preprocessed)

###############################################################################
# Get data for analysis of pneumonia vs normal
###############################################################################

# create a 1-D array of encoded class labels
ynp_true = (y_true != y_labels.index('normal')).astype(np.uint8)

# class labels
ynp_labels = ['normal',
              'pneumonia']

# get predictions
ynp_pred = (y_pred != y_labels.index('normal')).astype(np.uint8)
ynp_pred_proba_p = (y_pred_proba[:, y_labels.index('bacterial_pneumonia')]
                    + y_pred_proba[:, y_labels.index('viral_pneumonia')])

###############################################################################
# Confusion matrix (pneumonia vs normal)
###############################################################################

# get confusion matrix
df_np_cm = ume.get_confusion_matrix_df(y_true=ynp_true,
                                       y_pred=ynp_pred,
                                       y_labels=ynp_labels)

# make heatmap of confusion matrix
fig_np_cm = ume.get_confusion_matrix_heatmap_figure(df_np_cm)

# save the heatmap figure
heatmap_file_path = (FIGURE_OUTPUT_PATH
                     + mn
                     + '_normalVSpneumonia_confusion_matrix_heatmap.png')
if os.path.exists(heatmap_file_path):
    os.remove(heatmap_file_path)
fig_np_cm.savefig(heatmap_file_path,
                  bbox_inches='tight')

###############################################################################
# Table of scores (pneumonia vs normal)
###############################################################################

# get classification report
df_np_cr = ume.get_classification_report_df(y_true=ynp_true,
                                            y_pred=ynp_pred,
                                            y_labels=ynp_labels)

# create dataframe with scores
df_p_scores = df_np_cr.loc['Pneumonia', ['f1-score', 'precision', 'recall']]
df_p_scores['ROC AUC'] = roc_auc_score(ynp_true, ynp_pred_proba_p)
df_p_scores['average precision'] = average_precision_score(ynp_true,
                                                           ynp_pred_proba_p)
df_p_scores = df_p_scores.astype(str)

# create a figure
fig_np_scores = plt.figure(figsize=(6.4, 4))

# get axis
ax = fig_np_scores.gca()

# do not display the axis
ax.axis('off')

# create table
t = ax.table(cellText=list(zip(df_p_scores.index, df_p_scores.values)),
             cellLoc='center',
             loc='right',
             bbox=[0.1, 0, 0.8, 1])

# change settings for index cells
for y in range(df_p_scores.size):
    t[y, 0].set_facecolor('#d1e5fa')  # light blue
    t[y, 0].set_text_props(fontweight='bold')

# set number of digits to display
for y in range(df_p_scores.size):
    t[y, 1].get_text().set_text('%.2f'
                                % float(t[y, 1].get_text().get_text()))

# set the font size
t.auto_set_font_size(False)
t.set_fontsize(14)

# save the table figure
table_file_path = (FIGURE_OUTPUT_PATH
                   + mn
                   + '_normalVSpneumonia_scores_table.png')
if os.path.exists(table_file_path):
    os.remove(table_file_path)
fig_np_scores.savefig(table_file_path,
                      bbox_inches='tight')

###############################################################################
# ROC curve (pneumonia vs normal)
###############################################################################

# create a figure
fig_roc = plt.figure(figsize=(6.4, 4))

# get axis
ax_roc = fig_roc.gca()

# plot ROC curve
fpr, tpr, _ = roc_curve(ynp_true, ynp_pred_proba_p)
ax_roc.plot(fpr, tpr)

# plot baseline
ax_roc.plot([0, 1],
            [0, 1],
            '--k')

# set labels and title
ax_roc.set_xlabel('False Positive Rate')
ax_roc.set_ylabel('True Positive Rate')
ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve'
                 + '\nfor Pneumonia vs. Normal Classification')

# configure axes
ax_roc.axis('square')

# save the ROC curve image file
roc_curve_file_path = (FIGURE_OUTPUT_PATH
                       + mn
                       + '_normalVSpneumonia_ROC_curve.png')
if os.path.exists(roc_curve_file_path):
    os.remove(roc_curve_file_path)
fig_roc.savefig(roc_curve_file_path,
                bbox_inches='tight')

###############################################################################
# Precision-recall curve (pneumonia vs normal)
###############################################################################

# create a figure
fig_prc = plt.figure(figsize=(6.4, 4))

# get axis
ax_prc = fig_prc.gca()

# plot precision-recall curve
precision, recall, _ = precision_recall_curve(ynp_true, ynp_pred_proba_p)
ax_prc.plot(recall, precision)

# plot baseline
ax_prc.hlines(np.sum(ynp_true) / ynp_true.size,
              xmin=0,
              xmax=1,
              colors='k',
              linestyles='--')

# set labels and title
ax_prc.set_xlabel('Recall')
ax_prc.set_ylabel('Precision')
ax_prc.set_title('Precision-Recall Curve'
                 + '\nfor Pneumonia vs. Normal Classification')

# configure axes
ax_prc.axis('square')
buffer = 0.05
ax_prc.set_xlim([0 - buffer, 1 + buffer])
ax_prc.set_ylim([0 - buffer, 1 + buffer])

# save the precision-recall curve image file
pr_curve_file_path = (FIGURE_OUTPUT_PATH
                      + mn
                      + '_normalVSpneumonia_precision_recall_curve.png')
if os.path.exists(pr_curve_file_path):
    os.remove(pr_curve_file_path)
fig_prc.savefig(pr_curve_file_path,
                bbox_inches='tight')
