import numpy as np
import load_processed_data as ld
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Sequential


def compile_model(base, classifier, loss='binary_crossentropy',
                  optimizer=Adam(), metrics=None):
    """This function builds a model and compiles it.

    Input arguments:
        base - a keras model that takes the image data as input and outputs
               an array of flattened bottleneck features.
        classifier - a keras model that takes the array of
                     flattened bottleneck features as input and outputs
                     the class predictions.
        loss - (optional) loss for the model compilation.
        optimizer - (optional) optimizer for the model compilation.
        metrics - (optional) list of metrics for the model compilation.
                  If metrics is None, this function will pass the list
                  ['accuracy'] to the metrics keyword of the compile method.

    This function returns a compiled Sequential object."""

    # set default metrics if necessary
    if metrics is None:
        metrics = ['accuracy']

    # instantiate model
    model = Sequential()

    # add base and classifier
    model.add(base)
    model.add(classifier)

    # compile the model
    model.compile(optimizer=optimizer,
                  loss=loss,
                  metrics=metrics)

    # return the model
    return model


def fit_and_save_model(model, output_file_path, batch_size=30,
                       class_weight=None, epochs=100):
    """This function fits a model and then saves it to an HDF5 file.

    Input arguments:
        model - a keras model that has been compiled.
        output_file_path - pathname of the output .h5 file.
        batch_size - (optional) batch size for model training, expressed
                     as an integer.
            NOTE: this implementation of the function uses the same batch size
                  for the train and validation data generation.
        class_weight - (optional) dictionary mapping class indices (integers)
                       to weight values (floats), to be used during
                       model training.
        epochs - (optional) number of epochs for model training.

    This function returns a History object that contains the
    model training history."""

    # load the train data
    X_train, y_train, _ = ld.load_train_data()

    # load the validation data
    X_val, y_val, _ = ld.load_validation_data()

    # training image data generator
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       zoom_range=0.3,
                                       rotation_range=10,
                                       width_shift_range=0.2,
                                       height_shift_range=0.2,
                                       shear_range=0.1,
                                       fill_mode='nearest')
    train_generator = train_datagen.flow(X_train,
                                         y=y_train,
                                         batch_size=batch_size,
                                         seed=10)

    # validation image data generator
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow(X_val,
                                     y=y_val,
                                     batch_size=batch_size,
                                     seed=10)

    # fit the model
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=np.ceil(X_train.shape[0]
                                                          / batch_size),
                                  epochs=epochs,
                                  validation_data=val_generator,
                                  validation_steps=np.ceil(X_val.shape[0]
                                                           / batch_size),
                                  verbose=1,
                                  class_weight=class_weight)

    # save model to HDF5 file
    model.save(output_file_path)

    # return the training history
    return history
