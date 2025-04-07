# Distil - Bert 복구버전

from ..base_model import BaseModel
import tensorflow as tf
import sys
from tensorflow.python.keras.models import load_model

#from tensorflow.keras.layers import InputLayer as OriginalInputLayer
from tensorflow.python.keras.engine.input_layer import InputLayer as OriginalInputLayer

from keras.src.layers import Input, Dense, Concatenate, Layer, MultiHeadAttention, LayerNormalization, Flatten, Dropout

# Distil Bert 사용시
from transformers import DistilBertTokenizer, TFDistilBertModel
# Bert Large 사용시
#from transformers import BertTokenizer, TFBertModel

import params as PARAMS
from keras.src.models import Model
import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from keras.src.initializers import Constant
from transformers import AutoTokenizer, TFAutoModel

import keras
from keras.src.initializers import Constant
import keras.src.ops as K

#from tensorflow.python.keras.layers import LayerNormalization

    
def compute_class_biases(labels):
    # Assuming labels are one-hot encoded, calculate the class distribution
    class_totals = np.sum(labels, axis=0)
    class_probs = class_totals / np.sum(class_totals)

    # Calculate logit bias: log(p / (1 - p)) for each class
    initial_bias = np.log(class_probs / (1 - class_probs))
    return initial_bias
# BERT Large 사용시 + pooler -> use_pooler_output=True
class BERTEmbeddingLayer(Layer):
    def __init__(self, bert_model,use_pooler_output=False ,**kwargs):
        super(BERTEmbeddingLayer, self).__init__(**kwargs)
        self.bert_model = bert_model  # Reuse the passed BERT model
        # pooler layer 추가
        self.use_pooler_output = use_pooler_output
        # self.bert_model.trainable = False

    def call(self, inputs):
        input_ids, attention_mask = inputs
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        # DistilBERT에는 pooler_output이 없음 → last_hidden_state[:, 0, :] 사용
        # BERT-Large 등에는 pooler_output이 있으므로, use_pooler_output=True면 그걸 사용
        if self.use_pooler_output and hasattr(output, 'pooler_output') and (output.pooler_output is not None):
            return output.pooler_output
        else:
            # pooler를 사용하지 않거나 모델에 pooler가 없는 경우
            return output.last_hidden_state[:, 0, :]
        #return output.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding

    def get_config(self):
        # This is required for model serialization
        config = super(BERTEmbeddingLayer, self).get_config()
        print(config)
        # We cannot directly serialize bert_model, so we exclude it
        return config

    @classmethod
    def from_config(cls, config):
        # Recreate the BERT model when loading the layer
        #Distil Bert 사용시
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        # bert_model = TFAutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
        # bert_model.trainable = False
        return cls(bert_model=bert_model, **config)
# 버전 안맞아서 추가


class Bert(BaseModel):
    """
    BaseModel을 상속하여, 데이터셋 생성부터 모델 구성(build), 
    학습(train), 평가(test)까지 담당하는 클래스입니다.
    """
    def __init__(self, df):
        super().__init__(df)
        # Distil Bert 사용시
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        # Bert Large 사용시
        #self.tokenizer = BertTokenizer.from_pretrained('bert-large-uncased')


    def tokenize_feature(self, texts, max_length=PARAMS.MAX_LEN):
        """
        입력 문장을 토크나이저로 토큰화하고, 
        최대 길이(max_length)에 맞춰 잘라내거나(padding='max_length'),
        필요한 경우 truncation을 적용해줍니다.
        """
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
        """
        BaseModel의 make_dataset을 상속받아 실제 학습/검증용 데이터를 준비.
        """
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

        self.token_counts = {
            'train': {},
            'val': {}
        }

        # string features
        for feature in PARAMS.FEATURES:
            if feature == "Patient_ID":
                continue
            # Remove spaces in feature names for compatibility with input layer names
            feature_key = feature.replace(" ", "_")

            if PARAMS.FULL_FEATURES[feature] == 'str':
                # Add input IDs and attention masks for train and val to their respective lists
                self.train_inputs.append(train_encodings[feature][0])  # input IDs for train
                self.train_inputs.append(train_encodings[feature][1])  # attention mask for train
                self.val_inputs.append(val_encodings[feature][0])  # input IDs for val
                self.val_inputs.append(val_encodings[feature][1])  # attention mask for val

                # Update token counts as a dictionary if needed for analysis
                self.token_counts['train'][feature_key] = [
                    len(self.tokenizer.tokenize(text)) for text in self.train_df[feature]
                ]
                self.token_counts['val'][feature_key] = [
                    len(self.tokenizer.tokenize(text)) for text in self.val_df[feature]
                ]

            elif PARAMS.FULL_FEATURES[feature] == 'int32':
                # Add integer features to train and val lists
                self.train_inputs.append(tf.convert_to_tensor(self.train_df[feature], dtype=tf.float32))
                self.val_inputs.append(tf.convert_to_tensor(self.val_df[feature], dtype=tf.float32))

        self.data_loaded = True

    def build(self):
        # Distil Bert 사용시
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        # Bert Large 사용시
        #bert_model = TFBertModel.from_pretrained('bert-large-uncased')
        # bert_model = TFAutoModel.from_pretrained("microsoft/MiniLM-L12-H384-uncased")
        shared_embedding_layer = BERTEmbeddingLayer(bert_model=bert_model)

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
                model_inputs.extend([input_ids, attention_mask])

            elif PARAMS.FULL_FEATURES[feature] == 'int32':
                feature_embedding = Input(shape=(1,), dtype=tf.float32, name=f"{feature_key}")
                model_inputs.append(feature_embedding)
            
            feature_proj = Dense(256)(feature_embedding)
            # Add Dropout after Attention layer
            #attention_output = Dropout(0.3)(attention_output)  # Apply Dropout with rate=0.3
            feature_embeddings.append(feature_proj)

        concatenated_features = K.stack(feature_embeddings, axis=1)

        # Attention layer to capture attention scores
        attention_layer = MultiHeadAttention(num_heads=PARAMS.NUM_HEAD, key_dim=256//PARAMS.NUM_HEAD)
        attention_output, attention_scores = attention_layer(concatenated_features, concatenated_features, return_attention_scores=True)
        attention_output = LayerNormalization()(attention_output + concatenated_features)
        
        # drop-out Add Dropout after Attention layer
        #attention_output = Dropout(0.3)(attention_output)  # Apply Dropout with rate=0.3

        # Add dense layers for classification
        fl_output = Flatten()(attention_output)
        output = Dense(64, activation='relu')(fl_output)
        # output = Dense(64, activation='relu')(output)
        output = Dense(16, activation='relu')(output)

        train_labels = to_categorical(self.train_df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)
        initial_bias = compute_class_biases(train_labels)
        output = Dense(PARAMS.NUM_CLASSES, activation='softmax',
                       bias_initializer= Constant(initial_bias))(output)

        # Model outputs main classification output and attention scores
        self.model = Model(inputs=model_inputs, outputs=output)
        

    def test(self):
        #self.model = load_model('model.keras', custom_objects={'BERTEmbeddingLayer': BERTEmbeddingLayer})
        self.model = keras.saving.load_model("model.keras", custom_objects={'BERTEmbeddingLayer': BERTEmbeddingLayer})

        super().test()
