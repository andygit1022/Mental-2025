# model/base_model.py
import os
from keras.src.optimizers import Adam
from keras.src.callbacks import Callback
from .drawing import DrawPlot, plot_confusion_matrix, plot_attention_scores
import params as PARAMS
import sklearn.metrics as skm
from sklearn.metrics import confusion_matrix
import numpy as np
import tensorflow as tf
from keras.src.utils import to_categorical
from abc import ABC, abstractmethod
import matplotlib
from matplotlib import pyplot as plt

from keras.src.optimizers import Adam, SGD
import keras
from keras import Model

from keras.src.optimizers.schedules.learning_rate_schedule import CosineDecayRestarts
from keras.src.metrics import Precision, Recall, CategoricalAccuracy
from keras.src.losses import CategoricalCrossentropy


class PlotLosses(Callback):
    def __init__(self, fn, val_data=None, train_data=None):
        super().__init__()
        self.validation_data = val_data
        self.train_data = train_data
        self.fn = fn  # Model saving filename

        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.f1 = []
        self.val_f1 = []

        # Store per-class precision, recall, and F1 scores dynamically
        print("DEBUG:", PARAMS.NUM_CLASSES, type(PARAMS.NUM_CLASSES))
        self.precision = {i: [] for i in range(PARAMS.NUM_CLASSES)}
        self.recall = {i: [] for i in range(PARAMS.NUM_CLASSES)}
        self.f1_per_class = {i: [] for i in range(PARAMS.NUM_CLASSES)}
        self.val_precision = {i: [] for i in range(PARAMS.NUM_CLASSES)}
        self.val_recall = {i: [] for i in range(PARAMS.NUM_CLASSES)}
        self.val_f1_per_class = {i: [] for i in range(PARAMS.NUM_CLASSES)}

        self.logs = []
        self.best_loss = float('inf')
        self.best_f1 = 0
        self.best_f1_per_class = {i: 0 for i in range(PARAMS.NUM_CLASSES)}
        self.wait = 0

        # 결과 폴더 생성
        os.makedirs("results", exist_ok=True)

        self.initialize_plot()

    def initialize_plot(self):
        plt.ion()  # Enable interactive mode
        num_plots = 3 + (PARAMS.NUM_CLASSES * 3)  # Loss, Accuracy, Macro F1 + (3 metrics per class)
        self.fig, self.axs = plt.subplots(1, num_plots, figsize=(num_plots * 3, 5))

        # Titles for each subplot
        titles = ["Loss", "Accuracy", "Macro F1"] + \
                 [f"F1_{i}" for i in range(PARAMS.NUM_CLASSES)] + \
                 [f"Precision_{i}" for i in range(PARAMS.NUM_CLASSES)] + \
                 [f"Recall_{i}" for i in range(PARAMS.NUM_CLASSES)]

        for ax, title in zip(self.axs, titles):
            ax.set_title(title)
            ax.set_xlabel("Epochs")
            ax.set_ylim([0, 1])  # Set y-limits for metrics plots


    def on_epoch_end(self, epoch, logs=None):
        if logs is None:
            logs = {}

        self.x.append(epoch + 1)
        self.logs.append(logs)

        # Record loss and accuracy
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('categorical_accuracy'))
        self.val_acc.append(logs.get('val_categorical_accuracy'))

        # Compute per-class metrics
        macro_f1 = 0
        val_macro_f1 = 0
        for i in range(PARAMS.NUM_CLASSES):
            pr = logs.get(f'precision_{i}', 0)
            rc = logs.get(f'recall_{i}', 0)
            val_pr = logs.get(f'val_precision_{i}', 0)
            val_rc = logs.get(f'val_recall_{i}', 0)

            # Compute F1 scores
            f1 = (2 * pr * rc) / (pr + rc) if (pr + rc) > 0 else 0
            val_f1 = (2 * val_pr * val_rc) / (val_pr + val_rc) if (val_pr + val_rc) > 0 else 0

            # Store values
            self.precision[i].append(pr)
            self.recall[i].append(rc)
            self.f1_per_class[i].append(f1)

            self.val_precision[i].append(val_pr)
            self.val_recall[i].append(val_rc)
            self.val_f1_per_class[i].append(val_f1)

            macro_f1 += f1
            val_macro_f1 += val_f1

        # Compute Macro F1 score
        macro_f1 /= PARAMS.NUM_CLASSES
        val_macro_f1 /= PARAMS.NUM_CLASSES
        self.f1.append(macro_f1)
        self.val_f1.append(val_macro_f1)

        # Save best model
        if self.val_losses[-1] < self.best_loss or val_macro_f1 > self.best_f1:
            print(f"\nModel saved at epoch {epoch + 1}")
            self.best_loss = min(self.best_loss, self.val_losses[-1])
            self.best_f1 = max(self.best_f1, val_macro_f1)
            for i in range(PARAMS.NUM_CLASSES):
                self.best_f1_per_class[i] = max(self.best_f1_per_class[i], self.val_f1_per_class[i][-1])
            self.model.save(self.fn)
            self.wait = 0
        else:
            self.wait += 1
            if self.wait >= 100:
                self.model.stop_training = True
                print(f"\nEarly stopping at epoch {epoch + 1}")

        # 업데이트된 플롯 그리기 & 저장
        self.update_plot(epoch+1)

    def update_plot(self, epoch_number):
        if self.fig is None or self.axs is None:
            self.initialize_plot()

        # Clear previous data for all subplots
        for ax in self.axs:
            ax.cla()
            ax.set_ylim([0, 1])

        # 1) Loss
        self.axs[0].plot(self.x, self.losses, 'b', label="Loss")
        self.axs[0].plot(self.x, self.val_losses, 'r', label="Val Loss")
        self.axs[0].legend()

        # 2) Accuracy
        self.axs[1].plot(self.x, self.acc, 'b', label="Accuracy")
        self.axs[1].plot(self.x, self.val_acc, 'r', label="Val Accuracy")
        self.axs[1].legend()

        # 3) Macro F1
        self.axs[2].plot(self.x, self.f1, 'b', label="Macro F1")
        self.axs[2].plot(self.x, self.val_f1, 'r', label="Val Macro F1")
        self.axs[2].legend()

        # 4~ : Per-class metrics
        plot_idx = 3
        for i in range(PARAMS.NUM_CLASSES):
            # F1 Score
            self.axs[plot_idx].plot(self.x, self.f1_per_class[i], 'b', label=f"F1_{i}")
            self.axs[plot_idx].plot(self.x, self.val_f1_per_class[i], 'r', label=f"Val F1_{i}")
            self.axs[plot_idx].legend()
            plot_idx += 1

            # Precision
            self.axs[plot_idx].plot(self.x, self.precision[i], 'b', label=f"PR_{i}")
            self.axs[plot_idx].plot(self.x, self.val_precision[i], 'r', label=f"Val PR_{i}")
            self.axs[plot_idx].legend()
            plot_idx += 1

            # Recall
            self.axs[plot_idx].plot(self.x, self.recall[i], 'b', label=f"RC_{i}")
            self.axs[plot_idx].plot(self.x, self.val_recall[i], 'r', label=f"Val RC_{i}")
            self.axs[plot_idx].legend()
            plot_idx += 1

        # 그래프 업데이트
        plt.pause(0.01)
        # 매 epoch마다 저장
        plt.savefig(f"results/loss_plot_epoch_{epoch_number}.png")

    def on_train_end(self, logs=None):
        plt.ioff()  # Disable interactive mode
        # 학습 끝난 시점에 최종 그림 저장
        plt.savefig("results/loss_plot_final.png")
        plt.show()  # Keep the plot open


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
        self.make_dataset()

        if self.model is None:
            self.build()

        first_decay_steps = PARAMS.EPOCHS_PER_CYCLE * int(len(self.train_df) / PARAMS.BATCH_SIZE)

        cos_decay_ann = CosineDecayRestarts(initial_learning_rate=PARAMS.LEARNING_RATE,
                                            first_decay_steps=first_decay_steps,
                                            t_mul=1.2, m_mul=0.99, alpha=0)
        optimizer = SGD(learning_rate=cos_decay_ann)
        # optimizer = Adam(learning_rate=PARAMS.LEARNING_RATE)

        loss_fn = CategoricalCrossentropy()
        metrics = [CategoricalAccuracy()] + \
                  [Precision(class_id=i, name=f'precision_{i}') for i in range(PARAMS.NUM_CLASSES)] + \
                  [Recall(class_id=i, name=f'recall_{i}') for i in range(PARAMS.NUM_CLASSES)]

        self.model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)
        self.model.summary()

        # Dataset 준비
        train_dataset = tf.data.Dataset.from_tensor_slices((
            (
                self.train_inputs[0],
                self.train_inputs[1],
                self.train_inputs[2],
                self.train_inputs[3],
            ),
            self.train_labels
        ))
        train_dataset = train_dataset.shuffle(len(self.train_labels))\
                                     .batch(PARAMS.BATCH_SIZE)\
                                     .prefetch(tf.data.experimental.AUTOTUNE)

        val_dataset = tf.data.Dataset.from_tensor_slices((
            (
                self.val_inputs[0],
                self.val_inputs[1],
                self.val_inputs[2],
                self.val_inputs[3],
            ),
            self.val_labels
        )).batch(PARAMS.BATCH_SIZE)

        try:
            self.model.fit(
                train_dataset,
                validation_data=val_dataset,
                epochs=PARAMS.EPOCHS,
                callbacks=[PlotLosses(fn=self.fn)]
            )
        except KeyboardInterrupt:
            pass  # Allow manual stopping

    def get_attention_scores(self, input_data):
        # Get the attention scores from the model by running inference
        _, attention_scores = self.model.predict(input_data)
        return attention_scores

    def test(self):
        if not self.data_loaded:
            self.make_dataset()

        intermediate_layer_model = Model(inputs=self.model.input,
                                         outputs=self.model.get_layer("multi_head_attention").output)

        intermediate_output = intermediate_layer_model.predict(self.train_inputs,verbose=10)
        
        att_scores = intermediate_output[1]
        print("[DEBUG] att_scores.shape =", att_scores.shape)

        attention_score = np.mean(intermediate_output[1], axis=(0,1))
        np.savetxt("attention_score.csv", attention_score, delimiter=",", fmt='%f')
        plot_attention_scores(attention_score, feature_labels=PARAMS.FEATURES[1:])

        o_pred = self.model.predict(self.val_inputs)

        pred = to_categorical(tf.argmax(o_pred, axis=1), num_classes=PARAMS.NUM_CLASSES)
        y_pred = np.argmax(pred, axis=1)
        y_true = np.argmax(self.val_labels, axis=1)

        cm = confusion_matrix(y_true=y_true, y_pred=y_pred)

        print(skm.classification_report(y_true, y_pred))
        plot_confusion_matrix(cm=cm, classes=PARAMS.CLASSES, title='confusion_matrix')

        diff_idx = np.where(y_pred != y_true)[0]
        print("Misclassifications")
        print(self.val_df.iloc[diff_idx][["Patient_ID", "Label", "Age"]].to_string(index=False))
