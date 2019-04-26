import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import rcParams
import load_processed_data as ld
from definitions import PROCESSED_DATA_PATH, FIGURE_OUTPUT_PATH
import cv2
import shutil
from keras.preprocessing.image import load_img

# configure matplotlib settings
rcParams.update({'font.size': 18})

###############################################################################
# Create Dataframe with all processed image data
###############################################################################

print('loading image data')

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

###############################################################################
# Copy example image files to figure output directory
###############################################################################

print('copying example image files')

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
# Create png file with example image files
###############################################################################

print('creating png file with example images')

# make a figure
fig = plt.figure(figsize=(8.5, 8.5))

# plot the normal image top center
plt.subplot(2, 4, (2, 3))
plt.imshow(load_img(filepath_examples['normal'], 0))
plt.axis('off')
plt.title('Normal',
          fontsize=18)

# plot the bacterial image lower left
plt.subplot(2, 4, (5, 6))
plt.imshow(load_img(filepath_examples['bacterial_pneumonia'], 0))
plt.axis('off')
plt.title('Bacterial Pneumonia',
          fontsize=18)

# plot the viral image lower right
plt.subplot(2, 4, (7, 8))
plt.imshow(load_img(filepath_examples['viral_pneumonia'], 0))
plt.axis('off')
plt.title('Viral Pneumonia',
          fontsize=18)

# adjust the subplots size in the figure
plt.subplots_adjust(left=0,
                    bottom=0,
                    right=1,
                    top=0.9)

# save the figure as a png file
fig.savefig(FIGURE_OUTPUT_PATH + 'example_images.png')

###############################################################################
# Create png file with pivot table of file counts by class and data set
###############################################################################

print('creating png file with file counts')

# copy the dataframe that has all processed image data
df_customize = df.copy()

# modify the image class values
df_customize['image_class'] = df_customize['image_class'].str.replace('_', ' ')

# make dataframe with file counts by data set and class
df_counts = pd.pivot_table(data=df_customize,
                           index='image_class',
                           columns='data_set',
                           values='image_file_base_path',
                           aggfunc='count',
                           margins=True,
                           margins_name='Total')

# rename the column index
df_counts.columns.name = 'Data Set'

# rename the index
df_counts.index.name = 'Image Class'

# reorder the index and columns in custom order
df_counts = df_counts.reindex(columns=['train',
                                       'validation',
                                       'test',
                                       'Total'],
                              index=['bacterial pneumonia',
                                     'viral pneumonia',
                                     'normal',
                                     'Total'])

# create a figure
fig = plt.figure(figsize=(6.4, 3))

# get the figure's axis
ax = fig.gca()

# do not display the axis
ax.axis('off')

# create a table from the pivot table
t = ax.table(cellText=df_counts.values,
             cellLoc='center',
             rowLabels=df_counts.index,
             colLabels=df_counts.columns,
             loc='center',
             bbox=[0, 0, 1, 1])

# format the cells
for x in range(3):
    t[0, x].set_facecolor('#ffdccc')  # light orange
    t[0, x].set_text_props(fontweight='bold')

for y in np.arange(1, 4):
    t[y, -1].set_facecolor('#d1e5fa')  # light blue
    t[y, -1].set_text_props(fontweight='bold')

for x in np.arange(-1, 4):
    t[4, x].set_facecolor('#e6e6e6')  # light grey
    t[4, x].set_text_props(fontweight='bold')

for y in range(4):
    t[y, 3].set_facecolor('#e6e6e6')  # light grey
    t[y, 3].set_text_props(fontweight='bold')

# set the font size
t.auto_set_font_size(False)
t.set_fontsize(12)

# save the table as a png file
fig.savefig(FIGURE_OUTPUT_PATH + 'file_counts.png',
            bbox_inches='tight')

###############################################################################
# Create mean image pixel arrays
###############################################################################

print('making mean images')

# initialize dicts of pixel arrays
X_rgb = {}
X_avg_rgb = {}
X_avg_bw = {}

# list of class names
class_list = df.image_class.unique().tolist()

# fill dicts with pixel arrays keyed by class
for c in class_list:
    # create 4-D arrays of pixel intensities (RGB)
    X_rgb[c] = np.array(
            df.pixel_array_custom_image_size[df.image_class == c].tolist())

    # get the mean image pixel arrays (RGB)
    X_avg_rgb[c] = np.mean(X_rgb[c],
                           axis=0)

    # get the mean image pixel arrays converted to greyscale
    X_avg_bw[c] = cv2.cvtColor(X_avg_rgb[c],
                               cv2.COLOR_RGB2GRAY)

###############################################################################
# Create image file with mean images
###############################################################################

print('creating png file with mean images')

# create the figure and subplot axes
fig, ax = plt.subplots(nrows=1,
                       ncols=3,
                       figsize=(13, 4))

# plot the mean images
for ic, c in enumerate(class_list):
    ax[ic].imshow(X_avg_bw[c],
                  cmap='gray')
    ax[ic].set_yticks([])
    ax[ic].set_xticks([])
    ax[ic].set_xlabel(c.replace("_", " ").title())

# add a title to the figure
fig.suptitle('Mean Images (%i x %i)'
             % (X_avg_bw['normal'].shape[0],
                X_avg_bw['normal'].shape[1]))

# save the figure as an image file
fig.savefig(FIGURE_OUTPUT_PATH + 'mean_images.png')

###############################################################################
# Create image file with intensity histograms
###############################################################################

print('creating png file with mean image intensity histograms')

# create the figure and subplot axes
fig, ax = plt.subplots(nrows=3,
                       ncols=1,
                       sharex=True,
                       sharey=True,
                       figsize=(8, 10))

# plot the intensity histograms
for ic, c in enumerate(class_list):
    ax[ic].hist(X_avg_bw[c].ravel(),
                bins=256,
                range=[0, 256],
                color='grey')
    ax[ic].text(x=-5,
                y=600,
                s=c.replace("_", " "),
                color='grey')

# add title and axis labels to figure
ax[0].set_title('Intensity Histograms of Mean Images')
ax[-1].set_xlabel('Pixel Intensity')
fig.text(x=0.02,
         y=0.5,
         s='Frequency Count',
         ha='center',
         va='center',
         rotation='vertical')

# save the figure as an image file
fig.savefig(FIGURE_OUTPUT_PATH + 'mean_images_intensity_histograms.png')

print('done')
