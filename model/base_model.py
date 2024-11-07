from keras.optimizers import Adam
from tensorflow import keras
from .drawing import DrawPlot, plot_confusion_matrix, plot_attention_scores
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

        first_decay_steps = PARAMS.EPOCHS_PER_CYCLE * int(len(self.train_df) / PARAMS.BATCH_SIZE)

        cos_decay_ann = tf.keras.optimizers.schedules.CosineDecayRestarts(initial_learning_rate=PARAMS.LEARNING_RATE,
                                                                          first_decay_steps=first_decay_steps,
                                                                          t_mul=1.2, m_mul=0.99, alpha=0)
        optimizer = tf.keras.optimizers.SGD(learning_rate=cos_decay_ann)
        # optimizer = Adam(learning_rate=PARAMS.LEARNING_RATE)

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

        train_tensor = tf.random.normal((16, 4, 1, 1))
        val_tensor = tf.random.normal((4, 4, 1, 1))
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

    def get_attention_scores(self, input_data):
        # Get the attention scores from the model by running inference
        _, attention_scores = self.model.predict(input_data)
        return attention_scores

    def test(self):
        if not self.data_loaded:
            self.make_dataset()

        intermediate_layer_model = tf.keras.Model(inputs=self.model.input,
                                                  outputs=self.model.get_layer("multi_head_attention").output)

        intermediate_output = intermediate_layer_model.predict(self.train_inputs)

        attention_score = np.mean(intermediate_output[1], axis=(0,1))
        np.savetxt("attention_score.csv", attention_score, delimiter=",", fmt='%f')
        plot_attention_scores(attention_score, feature_labels=PARAMS.FEATURES[1:])

        o_pred = self.model.predict(self.val_inputs)

        pred = to_categorical(tf.argmax(o_pred, axis=1), num_classes=2)
        y_pred = np.argmax(pred, axis=1)
        y_true = np.argmax(self.val_labels, axis=1)

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

        print(skm.classification_report(y_true, y_pred))
        plot_confusion_matrix(cm=cm, classes=PARAMS.CLASSES, title='confusion_matrix')

        diff_idx = np.where(y_pred != y_true)[0]
        print("Misclassifications")
        print(self.val_df.iloc[diff_idx][["Patient_ID", "Label", "Age"]].to_string(index=False))
