import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import load_processed_data as ld
from definitions import PROCESSED_DATA_PATH, FIGURE_OUTPUT_PATH
import cv2
import shutil

# configure matplotlib settings
rcParams.update({'font.size': 18})

###############################################################################
# Create Dataframe with all processed image data
###############################################################################

# load train, validation, test dataframes with the processed data
_, _, df_train = ld.load_train_data()
_, _, df_val = ld.load_validation_data()
_, _, df_test = ld.load_test_data()

df_train['data_set'] = 'train'
df_val['data_set'] = 'validation'
df_test['data_set'] = 'test'

# concatenate the dataframes into one with all processed data
df = pd.concat([df_train, df_val, df_test],
               ignore_index=True)

# list of class names
df_class_list = df.image_class.unique().tolist()

###############################################################################
# Copy example image files to figure output directory
###############################################################################

# filepath for training image files
train_image_filepath = PROCESSED_DATA_PATH + "train_data/image_files/"

# make dict with filepaths of example image files
filepath_examples = {}

filepath_examples['normal'] = train_image_filepath + "NORMAL-2504415-0001.jpeg"
filepath_examples['bacterial_pneumonia'] = train_image_filepath \
                                           + "BACTERIA-558657-0001.jpeg"
filepath_examples['viral_pneumonia'] = train_image_filepath \
                                       + "VIRUS-1801584-0006.jpeg"

# copy the example image files to the figure output directory
for k, fp in filepath_examples.items():
    shutil.copy(fp, FIGURE_OUTPUT_PATH + k + '_example.jpeg')

###############################################################################
# Create csv file with file counts grouped by data set and class
###############################################################################

# copy the dataframe that has all processed image data
df_customize = df.copy()

# modify the image class values
df_customize['image_class'] = df_customize['image_class'].str.replace('_', ' ')

# make dataframe with file counts grouped by data set and class
df_counts = df_customize.groupby(
        ['data_set', 'image_class']).count()[['pixel_array_custom_image_size']]

# rename the count column
df_counts.columns = ['Count']

# rename the MultiIndex levels
df_counts.index.rename(['Data Set', 'Image Class'],
                       inplace=True)

# reorder the index in custom order
df_counts = df_counts.reindex(index=['train',
                                     'validation',
                                     'test'],
                              level=0).reindex(index=['bacterial pneumonia',
                                                      'viral pneumonia',
                                                      'normal'],
                                               level=1)

# save the dataframe to a csv file
df_counts.to_csv(FIGURE_OUTPUT_PATH + 'file_counts.csv')

###############################################################################
# Create mean image pixel arrays
###############################################################################

# initialize dicts of pixel arrays
X_rgb = {}
X_avg_rgb = {}
X_avg_bw = {}

# fill dicts with pixel arrays keyed by class
for c in df_class_list:
    # create 4-D arrays of pixel intensities (RGB)
    X_rgb[c] = np.array(
            df.pixel_array_custom_image_size[df.image_class == c].tolist())

    # get the mean image pixel arrays (RGB)
    X_avg_rgb[c] = np.mean(X_rgb[c], axis=0)

    # get the mean image pixel arrays converted to greyscale
    X_avg_bw[c] = cv2.cvtColor(X_avg_rgb[c], cv2.COLOR_RGB2GRAY)

###############################################################################
# Create image file with mean images
###############################################################################

# create the figure and subplot axes
fig, ax = plt.subplots(nrows=1,
                       ncols=3,
                       figsize=(13, 4))

# plot the mean images
for ic, c in enumerate(df_class_list):
    ax[ic].imshow(X_avg_bw[c], cmap='gray')
    ax[ic].set_yticks([])
    ax[ic].set_xticks([])
    ax[ic].set_xlabel(c.replace("_", " ").title())

# add a title to the figure
fig.suptitle('Mean Images (%i x %i)' % (
        X_avg_bw['normal'].shape[0], X_avg_bw['normal'].shape[1]))

# save the figure as an image file
fig.savefig(FIGURE_OUTPUT_PATH + 'mean_images.png')

###############################################################################
# Create image file with intensity histograms
###############################################################################

# create the figure and subplot axes
fig, ax = plt.subplots(nrows=3,
                       ncols=1,
                       sharex=True,
                       sharey=True,
                       figsize=(8, 10))

# plot the intensity histograms
for ic, c in enumerate(df_class_list):
    ax[ic].hist(X_avg_bw[c].ravel(),
                bins=256,
                range=[0, 256],
                color='grey')
    ax[ic].text(-5, 600, c.replace("_", " "), color='grey')

# add title and axis labels to figure
ax[0].set_title('Intensity Histograms of Mean Images')
ax[-1].set_xlabel('Pixel Intensity')
fig.text(0.02, 0.5, 'Frequency Count',
         ha='center',
         va='center',
         rotation='vertical')

# save the figure as an image file
fig.savefig(FIGURE_OUTPUT_PATH + 'mean_images_intensity_histograms.png')