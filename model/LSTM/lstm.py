from base_model import BaseModel  # base_model.py 경로를 반영
import tensorflow as tf
from keras.layers import Input, Dense, LSTM, Concatenate, LayerNormalization, Flatten
from transformers import AutoTokenizer, TFAutoModel
import params as PARAMS
import numpy as np
from keras.utils import to_categorical
from tensorflow.python.keras import models


def compute_class_biases(labels):
    # Assuming labels are one-hot encoded, calculate the class distribution
    class_totals = np.sum(labels, axis=0)
    class_probs = class_totals / np.sum(class_totals)

    # Calculate logit bias: log(p / (1 - p)) for each class
    initial_bias = np.log(class_probs / (1 - class_probs))
    return initial_bias


class BertLSTM(BaseModel):

    def __init__(self, df):
        super().__init__(df)
        self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")  # BERT 토크나이저

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
            feature: self.tokenize_feature(self.train_df[feature], max_length=PARAMS.MAX_LEN)
            for feature in PARAMS.FEATURES if PARAMS.FULL_FEATURES[feature] == 'str'
        }

        val_encodings = {
            feature: self.tokenize_feature(self.val_df[feature], max_length=PARAMS.MAX_LEN)
            for feature in PARAMS.FEATURES if PARAMS.FULL_FEATURES[feature] == 'str'
        }

        self.train_inputs = []
        self.val_inputs = []

        # Process string and integer features
        for feature in PARAMS.FEATURES:
            if feature == "Patient_ID":
                continue
            feature_key = feature.replace(" ", "_")
            if PARAMS.FULL_FEATURES[feature] == 'str':
                self.train_inputs.extend(train_encodings[feature])
                self.val_inputs.extend(val_encodings[feature])
            elif PARAMS.FULL_FEATURES[feature] == 'int32':
                self.train_inputs.append(tf.convert_to_tensor(self.train_df[feature], dtype=tf.float32))
                self.val_inputs.append(tf.convert_to_tensor(self.val_df[feature], dtype=tf.float32))

        self.data_loaded = True

    def build(self):
        # BERT 모델 로드
        bert_model = TFAutoModel.from_pretrained("bert-base-uncased")
        bert_model.trainable = False  # BERT는 고정된 상태로 사용

        # BERT 입력 레이어
        input_ids = Input(shape=(PARAMS.MAX_LEN,), dtype=tf.int32, name="input_ids")
        attention_mask = Input(shape=(PARAMS.MAX_LEN,), dtype=tf.int32, name="attention_mask")

        # BERT 출력 임베딩
        bert_output = bert_model(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = bert_output.last_hidden_state  # Shape: [batch_size, seq_len, hidden_size]

        # LSTM 레이어 추가
        lstm_output = LSTM(256, return_sequences=False, name="lstm")(sequence_output)  # 마지막 출력만 사용

        # 정수형 특징 처리
        int_features = []
        for feature in PARAMS.FEATURES:
            if PARAMS.FULL_FEATURES[feature] == 'int32' and feature != "Patient_ID":
                input_layer = Input(shape=(1,), dtype=tf.float32, name=f"{feature.replace(' ', '_')}_input")
                dense_layer = Dense(128, activation="relu")(input_layer)
                int_features.append(dense_layer)

        # 모든 특징 병합
        all_features = [lstm_output] + int_features if int_features else [lstm_output]
        concatenated_features = Concatenate()(all_features)

        # Dense 레이어
        x = Dense(128, activation="relu")(concatenated_features)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)

        # 분류 레이어
        train_labels = to_categorical(self.train_df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)
        initial_bias = compute_class_biases(train_labels)
        output = Dense(
            PARAMS.NUM_CLASSES,
            activation="softmax",
            bias_initializer=tf.keras.initializers.Constant(initial_bias),
            name="output"
        )(x)

        # 모델 정의
        model_inputs = [input_ids, attention_mask] + [
            Input(shape=(1,), dtype=tf.float32, name=f"{feature.replace(' ', '_')}_input")
            for feature in PARAMS.FEATURES if PARAMS.FULL_FEATURES[feature] == 'int32' and feature != "Patient_ID"
        ]
        self.model = models(inputs=model_inputs, outputs=output)

    def test(self):
        # 가중치 로드 문제 해결
        try:
            self.model.load_weights(PARAMS.MODEL_PATH)
            print("Model weights loaded successfully.")
        except Exception as e:
            print(f"Error loading model weights: {e}")

        # 평가 메서드 실행
        super().test()
