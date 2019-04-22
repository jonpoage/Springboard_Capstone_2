from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.models import Sequential, Model
from keras.applications import vgg16


def get_my_NN_classifier(input_dim,
                         dense_activation='relu',
                         dropout_rate=0.3,
                         output_activation='softmax'):
    """This function creates a NN classifier.

    Input arguments:
        input_dim - integer input dimension.
        dense_activation - (optional) string indicating the activation
                            function to be used in the hidden Dense layers.
        dropout_rate - (optional) float indicating the dropout rate,
                            as a proportion between 0 and 1.
        output_activation - (optional) string indicating the activation
                            function to be used in the output layer.

    This function returns a Sequential object."""

    my_NN_classifier = Sequential()

    my_NN_classifier.add(Dense(512,
                               activation=dense_activation,
                               input_dim=input_dim))

    my_NN_classifier.add(Dropout(rate=dropout_rate))

    my_NN_classifier.add(Dense(512,
                               activation=dense_activation))

    my_NN_classifier.add(Dropout(rate=dropout_rate))

    my_NN_classifier.add(Dense(3, activation=output_activation))

    return my_NN_classifier


def get_my_conv_layers(input_shape=(224, 224, 3),
                       kernel_size=(3, 3),
                       pool_size=(2, 2),
                       activation='relu'):
    """This function creates a CNN feature extractor.

    Input arguments:
        input_shape - (optional) tuple of integers indicating the input shape.
        kernel_size - (optional) tuple of integers indicating the size of the
                                convolution window.
        pool_size - (optional) tuple of integers indicating the size of the
                                max pooling window.
        activation - (optional) string indicating the activation
                            function to be used in the Conv2D layers.

    This function returns a Sequential object."""

    my_conv_layers = Sequential()

    my_conv_layers.add(Conv2D(32,
                       kernel_size=kernel_size,
                       activation=activation,
                       input_shape=input_shape))

    my_conv_layers.add(MaxPooling2D(pool_size=pool_size))

    my_conv_layers.add(Conv2D(64,
                       kernel_size=kernel_size,
                       activation=activation))

    my_conv_layers.add(MaxPooling2D(pool_size=pool_size))

    my_conv_layers.add(Conv2D(128,
                       kernel_size=kernel_size,
                       activation=activation))

    my_conv_layers.add(MaxPooling2D(pool_size=pool_size))

    my_conv_layers.add(Conv2D(128,
                       kernel_size=kernel_size,
                       activation=activation))

    my_conv_layers.add(MaxPooling2D(pool_size=pool_size))

    my_conv_layers.add(Flatten())

    return my_conv_layers


def get_vgg16_model(input_shape=(224, 224, 3)):
    """This function creates an instance of the VGG16 model, with the 3 top
    FC layers excluded and a final Flatten layer added.

    Input arguments:
        input_shape - (optional) tuple of integers indicating the input shape.

    This function returns a Model object."""

    vgg = vgg16.VGG16(include_top=False,
                      weights='imagenet',
                      input_shape=input_shape)

    # add a flatten layer
    output = vgg.layers[-1].output
    output = Flatten()(output)
    vgg_model = Model(vgg.input, output)

    return vgg_model


def get_vgg16_frozen(input_shape=(224, 224, 3)):
    """This function calls get_vgg16_model() and freezes all layers of the
    resulting Model object.

    Input arguments:
        input_shape - (optional) tuple of integers indicating the input shape.

    This function returns a Model object."""

    vgg_model_frozen = get_vgg16_model(input_shape=input_shape)

    # freeze all layers
    vgg_model_frozen.trainable = False
    for layer in vgg_model_frozen.layers:
        layer.trainable = False

    return vgg_model_frozen


def get_vgg16_fine_tuning(input_shape=(224, 224, 3),
                          idx_first_trainable_layer=11):
    """This function calls get_vgg16_model() and freezes some amount of
    layers of the resulting Model object.

    Input arguments:
        input_shape - (optional) tuple of integers indicating the input shape.
        idx_first_trainable_layer - (optional) index of the first trainable
                        layer. This layer and all layers above it are
                        trainable. All layers below it are not trainable.

    This function returns a Model object."""

    vgg_model_fine_tuning = get_vgg16_model(input_shape=input_shape)

    # freeze certain layers only
    vgg_model_fine_tuning.trainable = True
    for idx_layer, layer in enumerate(vgg_model_fine_tuning.layers):
        if idx_layer < idx_first_trainable_layer:
            layer.trainable = False
        elif idx_layer >= idx_first_trainable_layer:
            layer.trainable = True
        else:
            raise Exception("Cannot freeze VGG16 layers! Invalid layer index.")

    return vgg_model_fine_tuning
