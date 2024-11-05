from keras.optimizers import Adam
from tensorflow import keras
from .drawing import DrawPlot, plot_confusion_matrix
import params as PARAMS
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from tensorflow.python.keras.utils.np_utils import to_categorical
from abc import ABC, abstractmethod


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, draw, val_data=None, train_data=None):
        super().__init__()
        self.validation_data = val_data
        self.train_data = train_data
        self.draw = draw

    def on_epoch_end(self, epoch, logs={}):
        self.draw.save(epoch, logs, self.model)
        return


class BaseModel(ABC):
    def __init__(self, df):
        self.model = None
        self.fn = PARAMS.MODEL_PATH

        (self.train_df, self.val_df) = df

        self.train_labels = None
        self.val_labels = None
        self.train_inputs = None
        self.val_inputs = None

        self.data_loaded = False

    def make_dataset(self):
        self.train_labels = to_categorical(self.train_df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)
        self.val_labels = to_categorical(self.val_df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)

        self.train_labels = tf.convert_to_tensor(self.train_labels, dtype=tf.float32)
        self.val_labels = tf.convert_to_tensor(self.val_labels, dtype=tf.float32)

    @abstractmethod
    def build(self):
        pass

    def train(self):
        if self.model is None:
            self.build()

        optimizer = Adam(learning_rate=PARAMS.LEARNING_RATE)
        loss_fn = keras.losses.CategoricalCrossentropy()
        metrics = [keras.metrics.CategoricalAccuracy(),
                   keras.metrics.Precision(class_id=0),
                   keras.metrics.Precision(class_id=1),
                   keras.metrics.Recall(class_id=0),
                   keras.metrics.Recall(class_id=1),
                   ]
        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        self.model.summary()

        self.make_dataset()
        try:
            draw = DrawPlot(fn=self.fn)
            self.model.fit(
                self.train_inputs,
                self.train_labels,
                validation_data=(self.val_inputs, self.val_labels),
                epochs=PARAMS.EPOCHS,
                batch_size=PARAMS.BATCH_SIZE,
                callbacks=[PlotLosses(draw)]
            )

        except KeyboardInterrupt:
            pass

    def test(self):
        if not self.data_loaded:
            self.make_dataset()

        o_pred = self.model.predict(self.val_inputs)

        pred = to_categorical(tf.argmax(o_pred, axis=1), num_classes=2)
        y_pred = np.argmax(pred, axis=1)
        y_true = np.argmax(self.val_labels, axis=1)

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

        print(skm.classification_report(y_true, y_pred))
        plot_confusion_matrix(cm=cm, classes=PARAMS.CLASSES, title='confusion_matrix')

        diff_idx = np.where(y_pred != y_true)[0]
        print("Misclassifications")
        print(self.val_df.iloc[diff_idx][["Patient_ID", "Type", "Age"]])
