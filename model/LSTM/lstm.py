from ..base_model import BaseModel
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Layer, Flatten, MultiHeadAttention, LayerNormalization, Dropout
from transformers import DistilBertTokenizer, TFDistilBertModel
import params as PARAMS
from keras.models import Model
from tensorflow.python.keras.utils.np_utils import to_categorical
import numpy as np
from keras.models import load_model
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras import regularizers
from keras.initializers import Constant


def compute_class_biases(labels):
    class_totals = np.sum(labels, axis=0)
    class_probs = class_totals / np.sum(class_totals)
    return np.log(class_probs / (1 - class_probs))


class DistilBERTEmbeddingLayer(Layer):
    def __init__(self, bert_model, **kwargs):
        super(DistilBERTEmbeddingLayer, self).__init__(**kwargs)
        self.bert_model = bert_model

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]  # [CLS] Token Embedding

    def get_config(self):
        config = super(DistilBERTEmbeddingLayer, self).get_config()
        return config

    @classmethod
    def from_config(cls, config):
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        return cls(bert_model=bert_model, **config)


class BertLSTM(BaseModel):

    def __init__(self, df):
        super().__init__(df)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_feature(self, texts, max_length=PARAMS.MAX_LEN):
        encoding = self.tokenizer(
            list(texts),
            max_length=max_length,
            padding="max_length",
            truncation=True,
            return_tensors="tf"
        )
        return encoding["input_ids"], encoding["attention_mask"]

    def make_dataset(self):
        super().make_dataset()

        train_encodings = {
            feature: self.tokenize_feature(self.train_df[feature])
            for feature in PARAMS.FEATURES if PARAMS.FULL_FEATURES[feature] == 'str'
        }

        val_encodings = {
            feature: self.tokenize_feature(self.val_df[feature])
            for feature in PARAMS.FEATURES if PARAMS.FULL_FEATURES[feature] == 'str'
        }

        self.train_inputs = []
        self.val_inputs = []

        # String features
        for feature in PARAMS.FEATURES:
            if feature == "Patient_ID":
                continue
            feature_key = feature.replace(" ", "_")

            if PARAMS.FULL_FEATURES[feature] == 'str':
                self.train_inputs.append(train_encodings[feature][0])  # Input IDs
                self.train_inputs.append(train_encodings[feature][1])  # Attention Masks
                self.val_inputs.append(val_encodings[feature][0])
                self.val_inputs.append(val_encodings[feature][1])

            elif PARAMS.FULL_FEATURES[feature] == 'int32':
                self.train_inputs.append(tf.convert_to_tensor(self.train_df[feature], dtype=tf.float32))
                self.val_inputs.append(tf.convert_to_tensor(self.val_df[feature], dtype=tf.float32))

        self.data_loaded = True

    def build(self):
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        shared_embedding_layer = DistilBERTEmbeddingLayer(bert_model=bert_model)

        feature_embeddings = []
        model_inputs = []

        for feature in PARAMS.FEATURES:
            if feature == "Patient_ID":
                continue
            feature_key = feature.replace(" ", "_")
            if PARAMS.FULL_FEATURES[feature] == 'str':
                input_ids = Input(shape=(PARAMS.MAX_LEN,), dtype=tf.int32, name=f"{feature_key}_input_ids")
                attention_mask = Input(shape=(PARAMS.MAX_LEN,), dtype=tf.int32, name=f"{feature_key}_attention_mask")
                feature_embedding = shared_embedding_layer([input_ids, attention_mask])
                model_inputs.extend([input_ids, attention_mask])
            elif PARAMS.FULL_FEATURES[feature] == 'int32':
                feature_embedding = Input(shape=(1,), dtype=tf.float32, name=f"{feature_key}")
                model_inputs.append(feature_embedding)

            feature_proj = Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01))(feature_embedding)
            feature_embeddings.append(feature_proj)

        concatenated_features = tf.stack(feature_embeddings, axis=1)

        # LSTM layers
        x = LSTM(256, return_sequences=True, kernel_regularizer=regularizers.l2(0.01))(concatenated_features)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)
        x = LSTM(128, return_sequences=False, kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.3)(x)
        x = BatchNormalization()(x)

        # Dense layers
        x = Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
        x = Dropout(0.3)(x)
        output = Dense(PARAMS.NUM_CLASSES, activation='softmax')(x)

        self.model = Model(inputs=model_inputs, outputs=output)



    def test(self):
        self.model = load_model('model.keras', custom_objects={'DistilBERTEmbeddingLayer': DistilBERTEmbeddingLayer})
        super().test()
