import models as mdl
import utils_model_fitting as umf
import utils_training_history as uth
from definitions import MODELS_PATH, FIGURE_OUTPUT_PATH
from keras.optimizers import Adam

# set model name
model_name = 'model_myCNNbase_myNNclf'

# get model base
base = mdl.get_my_conv_layers()

# get input dimensions for classifier
input_dim = base.output_shape[1]

# get classifier
clf = mdl.get_my_NN_classifier(input_dim=input_dim,
                               n_output=3)

# compile the model
model = umf.compile_model(base=base,
                          classifier=clf,
                          loss='categorical_crossentropy',
                          optimizer=Adam(),
                          metrics=['accuracy'])

# train the model
output_HDF5_file_path = MODELS_PATH + model_name + '.h5'
history = umf.fit_and_save_model(model,
                                 output_file_path=output_HDF5_file_path,
                                 batch_size=30,
                                 class_weight=None)

# save figure of training history losses and accuracies
output_png_file_path = FIGURE_OUTPUT_PATH + model_name \
                       + '_training_history_plots.png'
uth.save_training_history_plots(history,
                                output_file_path=output_png_file_path)
