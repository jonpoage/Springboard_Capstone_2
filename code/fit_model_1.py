import models as mdl

# get model base
base = mdl.get_my_conv_layers()

# get input dimensions for classifier
input_dim = base.output_shape[1]

# get classifier
clf = mdl.get_my_NN_classifier(input_dim=input_dim,
                               n_output=3)
