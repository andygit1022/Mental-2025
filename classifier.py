import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import itertools
from transformers import DistilBertTokenizer, TFDistilBertModel
import pandas as pd
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.layers import Input, Dense, Lambda, Concatenate, Layer
from keras.optimizers import Adam
from tensorflow.keras.models import Model, load_model, save_model

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        gpu_number = 0  # Change this to the desired GPU number (e.g., 1, 2, etc.)
        tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU')

        # Optional: Set memory growth to avoid memory allocation issues
        tf.config.experimental.set_memory_growth(gpus[gpu_number], True)

        print(f"Using GPU: {gpu_number}")
    except RuntimeError as e:
        # Visible devices must be set at program startup
        print(e)

print("TensorFlow version:", tf.__version__)
print("Keras version:", keras.__version__)

from matplotlib import pyplot as plt

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

class DrawPlot:
    def __init__(self, fn):
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.f1 = []
        self.f1_0 = []
        self.f1_1 = []
        self.val_f1_0 = []
        self.val_f1_1 = []
        self.logs = []
        self.legend = False
        self.best_acc = 0
        self.best_f1_0 = 0
        self.best_f1_1 = 0
        self.best_pr0 = 0
        self.best_pr1 = 0
        self.best_loss = 1
        self.best_f1 = 0
        self.val_f1 = []
        self.val_pr0 = []
        self.val_pr1 = []
        self.val_rc0 = []
        self.val_rc1 = []
        self.pr0 = []
        self.pr1 = []
        self.rc0 = []
        self.rc1 = []
        self.epoch = 0
        plt.ion()
        self.fig = plt.figure()
        self.fn = fn

    def save(self, epoch, logs, model):
        self.epoch += 1
        self.x.append(self.epoch)
        self.logs.append(logs)
        self.losses.append(logs.get('loss'))

        self.pr0.append(logs.get('precision'))
        self.rc0.append(logs.get('recall'))
        if self.pr0[-1] + self.rc0[-1] == 0:
            f1_0 = 0
        else:
            f1_0 = 2 * self.pr0[-1] * self.rc0[-1] / (self.pr0[-1] + self.rc0[-1])
        self.f1_0.append(f1_0)

        self.pr1.append(logs.get('precision_1'))
        self.rc1.append(logs.get('recall_1'))
        self.acc.append(logs.get('categorical_accuracy'))
        if self.pr1[-1] + self.rc1[-1] == 0:
            f1_1 = 0
        else:
            f1_1 = 2 * self.pr1[-1] * self.rc1[-1] / (self.pr1[-1] + self.rc1[-1])

        f1 = (f1_0 + f1_1) * 0.5

        self.f1_1.append(f1_1)
        self.f1.append(f1)

        self.val_pr0.append(logs.get('val_precision'))
        self.val_rc0.append(logs.get('val_recall'))
        self.val_losses.append(logs.get('val_loss'))

        if self.val_pr0[-1] + self.val_rc0[-1] == 0:
            val_f1_0 = 0
        else:
            val_f1_0 = 2 * self.val_pr0[-1] * self.val_rc0[-1] / (self.val_pr0[-1] + self.val_rc0[-1])
        self.val_f1_0.append(val_f1_0)

        self.val_acc.append(logs.get('val_categorical_accuracy'))
        self.val_pr1.append(logs.get('val_precision_1'))
        self.val_rc1.append(logs.get('val_recall_1'))

        if self.val_pr1[-1] + self.val_rc1[-1] == 0:
            val_f1_1 = 0
        else:
            val_f1_1 = 2 * self.val_pr1[-1] * self.val_rc1[-1] / (self.val_pr1[-1] + self.val_rc1[-1])
        self.val_f1_1.append(val_f1_1)
        val_f1 = (val_f1_0 + val_f1_1) * 0.5
        self.val_f1.append(val_f1)
        test1 = (
            (self.val_losses[-1] < self.best_loss)
        )
        test2 = (
                (self.val_f1[-1] > self.best_f1) and
                (self.val_f1_0[-1] > self.best_f1_0) and
                (self.val_f1_1[-1] > self.best_f1_1) and
                (self.val_pr0[-1] > 0.5) and
                (self.val_pr1[-1] > 0.5)
        )
        if ((test1 or test2)):
            print(
                "\nmodel save [%s, %s]: [loss: %f]\t[f1: %f --> %f]\t[f1_0: %f --> %f]\t[f1_1: %f --> %f]\t[pr0: %f --> %f]\t[pr1: %f --> %f]" % (
                    test1, test2,
                    self.val_losses[-1],
                    self.best_f1, self.val_f1[-1],
                    self.best_f1_0, self.val_f1_0[-1], self.best_f1_1, self.val_f1_1[-1], self.best_pr0,
                    self.val_pr0[-1],
                    self.best_pr1, self.val_pr1[-1]))
            if test2:
                self.best_f1 = self.val_f1[-1]
                self.best_f1_0 = self.val_f1_0[-1]
                self.best_f1_1 = self.val_f1_1[-1]
            self.best_pr0 = self.val_pr0[-1]
            self.best_pr1 = self.val_pr1[-1]
            if test1:
                self.best_loss = self.val_losses[-1]

            # keras.models.save_model(model, CONSTANT.DEV_MODEL_NAME)
            model.save(self.fn)

        if self.epoch % 10 == 0:
            plt.close()
            self.legend = False
            return

        plt.subplot(1, 7, 1)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.losses, 'b', label="loss")
        plt.plot(self.x, self.val_losses, 'r', label="val_loss")
        if not self.legend:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 2)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.f1, 'b', label="f1")

        plt.plot(self.x, self.val_f1, 'r', label="val_f1")
        plt.ylim([0, 1])
        if not self.legend:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 3)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.f1_0, 'b', label="f1_0")

        plt.plot(self.x, self.val_f1_0, 'r', label="val_f1_0")
        plt.ylim([0, 1])
        if not self.legend:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 4)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.f1_1, 'b', label="f1_1")

        plt.plot(self.x, self.val_f1_1, 'r', label="val_f1_1")
        plt.ylim([0, 1])
        if not self.legend:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 5)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.pr0, 'b', label="pr0")

        plt.plot(self.x, self.val_pr0, 'r', label="val_pr0")
        plt.ylim([0, 1])
        if self.legend == False:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 6)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.pr1, 'b', label="pr1")

        plt.plot(self.x, self.val_pr1, 'r', label="val_pr1")
        plt.ylim([0, 1])
        if self.legend == False:
            plt.legend(loc='upper left')
        plt.subplot(1, 7, 7)  # nrows=2, ncols=1, index=1
        plt.plot(self.x, self.acc, 'b', label="acc")

        plt.plot(self.x, self.val_acc, 'r', label="val_acc")
        plt.ylim([0, 1])
        if self.legend == False:
            plt.legend(loc='upper left')
            self.legend = True
        plt.pause(0.01)


class PlotLosses(keras.callbacks.Callback):
    def __init__(self, draw, val_data=None, train_data=None):
        super().__init__()
        self.validation_data = val_data
        self.train_data = train_data
        # plt.ion()
        # self.fig = plt.figure()
        self.draw = draw

    def on_epoch_end(self, epoch, logs={}):
        self.draw.save(epoch, logs, self.model)
        return

class DistilBERTEmbeddingLayer(Layer):
    def __init__(self, bert_model, **kwargs):
        super(DistilBERTEmbeddingLayer, self).__init__(**kwargs)
        self.bert_model = bert_model  # Reuse the passed DistilBERT model

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding

    def get_config(self):
        # This is required for model serialization
        config = super(DistilBERTEmbeddingLayer, self).get_config()
        # We cannot directly serialize bert_model, so we exclude it
        return config

    @classmethod
    def from_config(cls, config):
        # Recreate the BERT model when loading the layer
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        return cls(bert_model=bert_model, **config)

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.ioff()
    plt.close()
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="black" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

def test(val_inputs, val_labels):
    import sklearn.metrics as skm
    from sklearn.metrics import confusion_matrix
    CLASSES=["NC", "PRD"]

    # Loading the model with custom loss if needed
    custom_objects = {"CategoricalCrossentropy": tf.keras.losses.CategoricalCrossentropy()}
    model = load_model('model.keras', custom_objects={'DistilBERTEmbeddingLayer': DistilBERTEmbeddingLayer})

    # Evaluate the model
    # loss, accuracy = model.evaluate(val_inputs, val_labels)
    # print(f"Validation Loss: {loss}, Validation Accuracy: {accuracy}")


    o_pred = model.predict(val_inputs)

    pred = to_categorical(tf.argmax(o_pred, axis=1), num_classes=2)
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(val_labels, axis=1)

    cm = confusion_matrix(y_true=y_true, y_pred=y_pred)
    # xx = np.concatenate([y_true.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)
    # np.savetxt('result_test_SOL.csv', xx, delimiter=",")
    # if mode == "test":
    #     xx = np.concatenate([self.t_test.reshape(-1, 1), y_true.reshape(-1, 1), y_pred.reshape(-1, 1)], axis=1)
    #     # np.savetxt('result_test.csv', xx, delimiter=",")

    print(skm.classification_report(y_true, y_pred))
    plot_confusion_matrix(cm=cm, classes=CLASSES, title='confusion_matrix')

# Load data
df = pd.read_csv("template.csv")

# Define features and label columns
# features = [
#     "Literacy and Numeracy", "Medical History", "Medications", "Surgeries", "Stroke",
#     "Other History", "Vision", "Hearing", "Diet", "Sleep", "Alcohol", "Smoking",
#     "Family History", "Main Complaints", "Memory", "Language", "Orientation",
#     "Judgment and Problem Solving", "Social Activities", "Home and Hobbies",
#     "Daily Living", "Personality and Behavior"
# ]
features = [
    "Memory", "Language",
]
columns = ["Type"] + features
df[features] = df[features].astype(str)
df = df[columns]

# Encode labels
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['Type'])
num_classes = len(label_encoder.classes_)

# Split data
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42)

# Load DistilBERT tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Tokenize each feature separately with fixed max_length
def tokenize_feature(texts, max_length=512):
    encoding = tokenizer(
        list(texts),
        max_length=max_length,
        padding="max_length",  # Ensures fixed length of max_length
        truncation=True,
        return_tensors="tf"
    )
    # Return tensors directly
    return encoding["input_ids"], encoding["attention_mask"]

# Process train and validation encodings
train_encodings = {feature: tokenize_feature(train_df[feature]) for feature in features}
val_encodings = {feature: tokenize_feature(val_df[feature]) for feature in features}

# Convert labels to tensors
train_labels = to_categorical(train_df['label_encoded'], num_classes=num_classes)
val_labels = to_categorical(val_df['label_encoded'], num_classes=num_classes)

train_labels = tf.convert_to_tensor(train_labels, dtype=tf.float32)
val_labels = tf.convert_to_tensor(val_labels, dtype=tf.float32)

# Load DistilBERT model
# Initialize the DistilBERT model once
bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
shared_embedding_layer = DistilBERTEmbeddingLayer(bert_model=bert_model)

# Replace the Lambda layer with the custom DistilBERT embedding layer
# Create input and embedding layers for each feature
feature_embeddings = []
model_inputs = []
for feature in features:
    input_ids = Input(shape=(512,), dtype=tf.int32, name=f"{feature.replace(' ', '_')}_input_ids")
    attention_mask = Input(shape=(512,), dtype=tf.int32, name=f"{feature.replace(' ', '_')}_attention_mask")

    # Get BERT embeddings for the feature using Lambda wrapper
    feature_embedding = shared_embedding_layer([input_ids, attention_mask])
    feature_embeddings.append(feature_embedding)

    # Add to model inputs
    model_inputs.extend([input_ids, attention_mask])

# Concatenate all feature embeddings and create classification layer
concatenated_features = Concatenate()(feature_embeddings)
# output = Dense(100, activation='relu')(concatenated_features)
# output = Dense(100, activation='relu')(output)
output = Dense(num_classes, activation='softmax')(concatenated_features)

# Define inputs for model and create final model
model = Model(inputs=model_inputs, outputs=output)

# Compile model
optimizer = Adam(learning_rate=3e-5)
loss_fn = keras.losses.CategoricalCrossentropy()
metrics = [keras.metrics.CategoricalAccuracy(),
           keras.metrics.Precision(class_id=0),
           keras.metrics.Precision(class_id=1),
           keras.metrics.Recall(class_id=0),
           keras.metrics.Recall(class_id=1),
           ]
model.compile(optimizer=optimizer, loss=loss_fn, metrics=metrics)

# Print model summary to check structure
model.summary()

# Prepare input data for training and validation with exact key names
train_inputs = {}
val_inputs = {}

for feature in features:
    # Remove spaces in feature names for compatibility with input layer names
    feature_key = feature.replace(" ", "_")
    train_inputs[f"{feature_key}_input_ids"] = train_encodings[feature][0]
    train_inputs[f"{feature_key}_attention_mask"] = train_encodings[feature][1]
    val_inputs[f"{feature_key}_input_ids"] = val_encodings[feature][0]
    val_inputs[f"{feature_key}_attention_mask"] = val_encodings[feature][1]

test(val_inputs, val_labels)

try:
    draw = DrawPlot(fn="model.keras")
    model.fit(
        train_inputs,
        train_labels,
        validation_data=(val_inputs, val_labels),
        epochs=1,
        batch_size=4,
        callbacks=[PlotLosses(draw)]
    )

except KeyboardInterrupt:
    pass

test(val_inputs, val_labels)


