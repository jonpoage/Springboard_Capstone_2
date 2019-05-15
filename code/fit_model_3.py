import models as mdl
import utils_model_fitting as umf
import utils_training_history as uth
import os
from definitions import MODELS_PATH, FIGURE_OUTPUT_PATH
from keras.optimizers import Adam

# set model name
model_name = 'model_FINETUNINGvgg16base_myNNclf'

# get model base
base = mdl.get_vgg16_fine_tuning()

# get input dimensions for classifier
input_dim = base.output_shape[1]

# get classifier
clf = mdl.get_my_NN_classifier(input_dim=input_dim,
                               n_output=3)

# compile the model
model = umf.compile_model(base=base,
                          classifier=clf,
                          loss='categorical_crossentropy',
                          optimizer=Adam(lr=1e-5),
                          metrics=['accuracy'])

# pathname for output model file
output_HDF5_file_path = MODELS_PATH + model_name + '.h5'
if os.path.exists(output_HDF5_file_path):
    os.remove(output_HDF5_file_path)

# train the model
history = umf.fit_and_save_model(model,
                                 output_file_path=output_HDF5_file_path,
                                 batch_size=30,
                                 class_weight=None)

# pathname for output image file
output_png_file_path = (FIGURE_OUTPUT_PATH + model_name
                        + '_training_history_plots.png')
if os.path.exists(output_png_file_path):
    os.remove(output_png_file_path)

# save figure of training history losses and accuracies
uth.save_training_history_plots(history,
                                output_file_path=output_png_file_path)
