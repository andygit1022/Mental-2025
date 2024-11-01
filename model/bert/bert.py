from ..base_model import BaseModel
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.layers import Input, Dense, Lambda, Concatenate, Layer
from transformers import DistilBertTokenizer, TFDistilBertModel
import params as PARAMS
from tensorflow.keras.models import Model


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


class Bert(BaseModel):

    def __init__(self, df):
        super().__init__(df)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    def tokenize_feature(self, texts, max_length=512):
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

        train_encodings = {feature: self.tokenize_feature(self.train_df[feature]) for feature in PARAMS.FEATURES}
        val_encodings = {feature: self.tokenize_feature(self.val_df[feature]) for feature in PARAMS.FEATURES}

        self.train_inputs = {}
        self.val_inputs = {}

        for feature in PARAMS.FEATURES:
            # Remove spaces in feature names for compatibility with input layer names
            feature_key = feature.replace(" ", "_")
            self.train_inputs[f"{feature_key}_input_ids"] = train_encodings[feature][0]
            self.train_inputs[f"{feature_key}_attention_mask"] = train_encodings[feature][1]
            self.val_inputs[f"{feature_key}_input_ids"] = val_encodings[feature][0]
            self.val_inputs[f"{feature_key}_attention_mask"] = val_encodings[feature][1]

        self.data_loaded = True

    def build(self):
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        shared_embedding_layer = DistilBERTEmbeddingLayer(bert_model=bert_model)

        feature_embeddings = []
        model_inputs = []
        for feature in PARAMS.FEATURES:
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
        output = Dense(PARAMS.NUM_CLASSES, activation='softmax')(concatenated_features)

        self.model = Model(inputs=model_inputs, outputs=output)

    def test(self):
        self.model = load_model('model.keras', custom_objects={'DistilBERTEmbeddingLayer': DistilBERTEmbeddingLayer})

        super().test()
