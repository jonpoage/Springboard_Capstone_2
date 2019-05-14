import numpy as np
import load_processed_data as ld
from definitions import MODELS_PATH
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.models import Sequential


def compile_model(base, classifier, loss):

    # instantiate model
    model = Sequential()

    # add base and classifier
    model.add(base)
    model.add(classifier)

    # compile the model
    model.compile(optimizer=Adam(),
                  loss=loss,
                  metrics=['accuracy'])

    # return the model
    return model


def fit_and_save_model(model, str_model_name, batch_size=30):

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
                                         y_train,
                                         batch_size=batch_size,
                                         seed=10)

    # validation image data generator
    val_datagen = ImageDataGenerator(rescale=1./255)
    val_generator = val_datagen.flow(X_val,
                                     y_val,
                                     batch_size=batch_size,
                                     seed=10)

    # fit the model
    history = model.fit_generator(train_generator,
                                  steps_per_epoch=np.ceil(X_train.shape[0]
                                                          / batch_size),
                                  epochs=100,
                                  validation_data=val_generator,
                                  validation_steps=np.ceil(X_val.shape[0]
                                                           / batch_size),
                                  verbose=1,
                                  class_weight=None)

    # save model to HDF5 file
    model.save(MODELS_PATH + str_model_name + '.h5')

    return history
