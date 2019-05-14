import matplotlib.pyplot as plt


def plot_training_history(history):
    """This function creates a figure that displays the accuracy and loss for
    the train and validation sets over all epochs of the model training
    history.

    Input arguments:
        history - a keras.callbacks.History object.

    This function returns a Figure object."""

    # accuracy scores
    train_acc = history.history['acc']
    val_acc = history.history['val_acc']

    # loss values
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    # create figure with subplots
    fig, ax = plt.subplots(nrows=2,
                           ncols=1,
                           sharex=True,
                           figsize=(6.4, 10))

    # plot accuracy scores
    ax[0].plot(train_acc,
               'b',
               label='Train')
    ax[0].plot(val_acc,
               'm',
               label='Validation')
    ax[0].set_ylim([0, 1])
    ax[0].set_ylabel('Accuracy')
    ax[0].set_xlabel('Epoch')
    ax[0].legend()

    # plot loss scores
    ax[1].plot(train_loss,
               'b',
               label='Train')
    ax[1].plot(val_loss,
               'm',
               label='Validation')
    ax[1].set_ylim([0,
                    max(train_loss + val_loss)*1.05])
    ax[1].set_ylabel('Loss: Categorical Cross-Entropy')
    ax[1].set_xlabel('Epoch')
    ax[1].legend()

    # set title for figure
    fig.suptitle('Training History')

    # return the Figure object
    return fig
