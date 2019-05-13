import numpy as np
from keras.callbacks import Callback
import sklearn.metrics as metrics


class Custom_Metrics(Callback):

    def __init__(self, averaging='binary', p_threshold=0.5):
        self.averaging = averaging
        self.p_threshold = p_threshold

    def on_train_begin(self, logs={}):
        # initialize lists for metrics
        self.precisions = []
        self.recalls = []
        self.f1s = []

    def on_epoch_end(self, epoch, logs={}):
        # get predicted target values
        y_pred_probas = np.asarray(
                            self.model.predict(self.model.validation_data[0]))
        y_pred = (y_pred_probas >= self.p_threshold).astype('int')

        # get true target values
        y_true = self.model.validation_data[1]

        # evaluate metrics for epoch
        _precision = metrics.precision_score(y_true,
                                             y_pred,
                                             average=self.averaging)
        _recall = metrics.recall_score(y_true,
                                       y_pred,
                                       average=self.averaging)
        _f1 = metrics.f1_score(y_true,
                               y_pred,
                               average=self.averaging)

        # print summary for epoch
        print(" - precision: %f    - recall: %f    - f1 score: %f"
              % (_precision, _recall, _f1))

        # append epoch metrics to summary lists
        self.precisions.append(_precision)
        self.recalls.append(_recall)
        self.f1s.append(_f1)

        # return nothing
        return
