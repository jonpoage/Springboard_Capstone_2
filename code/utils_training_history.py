import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams


def save_training_history_plots(history, output_file_path):
    """This function calls plot_training_history() and saves the returned
    figure to an image file.

    Input arguments:
        history - a keras.callbacks.History object.
        output_file_path - pathname of the output image file.

    This function does not return anything."""

    # get training history figure
    fig = plot_training_history(history)

    # save figure to image file
    fig.savefig(output_file_path)


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

    # configure matplotlib settings
    rcParams.update({'font.size': 18})

    # create figure with subplots
    fig, ax = plt.subplots(nrows=2,
                           ncols=1,
                           sharex=True,
                           figsize=(6.4, 10))

    # legend colors
    lc = {'train': 'b',
          'val': 'm'}

    # plot accuracy scores
    ax[0].plot(train_acc,
               lc['train'],
               label='Train')
    ax[0].plot(val_acc,
               lc['val'],
               label='Validation')
    ax[0].set_ylim([0, 1])
    ax[0].set_ylabel('Accuracy')
    ax[0].grid()

    # plot loss scores
    ax[1].plot(train_loss,
               lc['train'],
               label='Train')
    ax[1].plot(val_loss,
               lc['val'],
               label='Validation')
    ax[1].set_ylim([0, np.ceil(max(train_loss + val_loss))])
    ax[1].set_ylabel('Loss')
    ax[1].set_xlabel('Epoch')
    ax[1].grid()

    # adjust subplots
    fig.subplots_adjust(hspace=0.16)

    # place the legend above the axes
    ax[0].legend(bbox_to_anchor=(0.1, 1.1, 0.8, 0.1),
                 ncol=2)

    # set title for figure
    fig.suptitle('Training History')

    # return the Figure object
    return fig
