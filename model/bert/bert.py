from ..base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.layers import Input, Dense, Lambda, Concatenate, Layer
from transformers import DistilBertTokenizer, TFDistilBertModel
import params as PARAMS
from tensorflow.keras.models import Model

from transformers import AutoTokenizer, TFAutoModel


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
        # bert_model = TFAutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
        return cls(bert_model=bert_model, **config)


class Bert(BaseModel):

    def __init__(self, df):
        super().__init__(df)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # self.tokenizer = AutoTokenizer.from_pretrained("microsoft/MiniLM-L12-H384-uncased")

    def tokenize_feature(self, texts, max_length=PARAMS.MAX_LEN):
        encoding = self.tokenizer(
            list(texts),
            max_length=max_length,
            padding="max_length",  # Ensures fixed length of max_length
            truncation=True,
            return_tensors="tf"
        )
        # Return tensors directly
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

        self.train_inputs = {}
        self.val_inputs = {}

        self.token_counts = {
            'train': {},
            'val': {}
        }

        # string feautres
        for feature in PARAMS.FEATURES:
            if feature == "Patient_ID":
                continue
            # Remove spaces in feature names for compatibility with input layer names
            feature_key = feature.replace(" ", "_")
            if PARAMS.FULL_FEATURES[feature] == 'str':
                self.train_inputs[f"{feature_key}_input_ids"] = train_encodings[feature][0]
                self.train_inputs[f"{feature_key}_attention_mask"] = train_encodings[feature][1]
                self.val_inputs[f"{feature_key}_input_ids"] = val_encodings[feature][0]
                self.val_inputs[f"{feature_key}_attention_mask"] = val_encodings[feature][1]

                self.token_counts['train'][feature_key] = [
                    len(self.tokenizer.tokenize(text)) for text in self.train_df[feature]
                ]
                self.token_counts['val'][feature_key] = [
                    len(self.tokenizer.tokenize(text)) for text in self.val_df[feature]
                ]

            elif PARAMS.FULL_FEATURES[feature] == 'int32':
                self.train_inputs[f"{feature_key}"] = tf.convert_to_tensor(self.train_df[feature], dtype=tf.float32)
                self.val_inputs[f"{feature_key}"] = tf.convert_to_tensor(self.val_df[feature], dtype=tf.float32)

        self.data_loaded = True

    def build(self):
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        # bert_model = TFAutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
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

                # Get BERT embeddings for text features
                feature_embedding = shared_embedding_layer([input_ids, attention_mask])
                feature_embeddings.append(feature_embedding)
                model_inputs.extend([input_ids, attention_mask])

            elif PARAMS.FULL_FEATURES[feature] == 'int32':
                int_input = Input(shape=(1,), dtype=tf.float32, name=f"{feature_key}")
                feature_embeddings.append(int_input)
                model_inputs.append(int_input)

            # Concatenate all feature embeddings
        concatenated_features = Concatenate()(feature_embeddings)

        # Add dense layers for classification
        output = Dense(100, activation='relu')(concatenated_features)
        # output = Dense(50, activation='relu')(output)
        output = Dense(PARAMS.NUM_CLASSES, activation='softmax')(output)

        self.model = Model(inputs=model_inputs, outputs=output)

    def test(self):
        self.model = load_model('model.keras', custom_objects={'DistilBERTEmbeddingLayer': DistilBERTEmbeddingLayer})

        super().test()
