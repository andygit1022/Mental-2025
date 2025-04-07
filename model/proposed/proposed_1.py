# model/proposed/proposed.py
from ..base_model import BaseModel
import tensorflow as tf
from tensorflow.python.keras.models import load_model
from keras.src.layers import Input, Dense, Concatenate, Layer, MultiHeadAttention, LayerNormalization, Flatten, TimeDistributed, Lambda, Reshape
from transformers import DistilBertTokenizer, TFDistilBertModel
import params as PARAMS
from keras.src.models import Model
import numpy as np
from keras.src.utils import to_categorical
from keras.src.initializers import Constant
from transformers import AutoTokenizer, TFAutoModel


#from tf_keras.src.initializers.initializers import Constant

def compute_class_biases(labels):
    # Assuming labels are one-hot encoded, calculate the class distribution
    class_totals = np.sum(labels, axis=0)
    class_probs = class_totals / np.sum(class_totals)

    # Calculate logit bias: log(p / (1 - p)) for each class
    initial_bias = np.log(class_probs / (1 - class_probs))
    return initial_bias

class DistilBERTEmbeddingLayer(Layer):
    def __init__(self, bert_model, **kwargs):
        super(DistilBERTEmbeddingLayer, self).__init__(**kwargs)
        self.bert_model = bert_model  # Reuse the passed DistilBERT model
        self.bert_model.trainable = True

    def call(self, inputs):
        input_ids, attention_mask = inputs
        input_ids = tf.cast(input_ids, dtype=tf.int32)  # 🔥 float32 → int32 변환
        attention_mask = tf.cast(attention_mask, dtype=tf.int32)  # 🔥 float32 → int32 변환
        output = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
        return output.last_hidden_state[:, 0, :]

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
        bert_model.trainable = True
        return cls(bert_model=bert_model, **config)

class Proposed(BaseModel):
    def __init__(self, df):
        super().__init__(df)
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
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
        def make_inputs(df):
            """
            Processes the dataframe and creates tokenized inputs with hierarchical structure.

            Returns:
            - input_ids: (batch_size, categories, sentences_per_category, max_length)
            - attention_masks: (batch_size, categories, sentences_per_category, max_length)
            - labels: One-hot encoded labels for classification
            """
            batch_input_ids = []
            batch_attention_masks = []
            batch_int_features = []
            batch_stat_features = []

            for index, row in df.iterrows():
                category_input_ids = []
                category_attention_masks = []
                category_int_features = []
                category_stat_features = []

                for category in PARAMS.FEATURES:
                    if category == "Patient_ID":
                        continue

                    if PARAMS.FULL_FEATURES[category] == 'str':
                        sentences = row[category] if isinstance(row[category], list) else []  # Ensure it's a list

                        # Prepare arrays for category-level input
                        category_sentences_input_ids = []
                        category_sentences_attention_masks = []

                        for i in range(PARAMS.NUM_SENTENCES):  # Handling up to 10 sentences per category
                            if i >= len(sentences):
                                # Padding with zeros for missing sentences
                                category_sentences_input_ids.append(np.zeros((PARAMS.MAX_LEN,), dtype=np.int32))
                                category_sentences_attention_masks.append(np.zeros((PARAMS.MAX_LEN,), dtype=np.int32))
                            else:
                                sentence = sentences[i]
                                train_encodings = self.tokenize_feature([sentence])  # Tokenize the sentence
                                category_sentences_input_ids.append(train_encodings[0].numpy()[0])
                                category_sentences_attention_masks.append(train_encodings[1].numpy()[0])

                        # Stack all sentences within a category → Shape: (10, MAX_LEN)
                        category_input_ids.append(np.stack(category_sentences_input_ids))
                        category_attention_masks.append(np.stack(category_sentences_attention_masks))

                        category_ind_stat_features = []

                        for f in ["_avg_words", "_total_sentences", "_avg_chars", "_total_numbers", "_avg_idf_weight"]:
                            c_name = category + f
                            category_ind_stat_features.append(tf.convert_to_tensor(row[c_name], dtype=tf.float32)) #
                        category_stat_features.append(np.stack(category_ind_stat_features))

                    elif PARAMS.FULL_FEATURES[category] == 'int32':
                        category_int_features.append(tf.convert_to_tensor(row[category], dtype=tf.int32))

                # Stack all categories → Shape: (categories, sentences, max_length)
                batch_input_ids.append(np.stack(category_input_ids))
                batch_attention_masks.append(np.stack(category_attention_masks))
                batch_int_features.append(np.stack(category_int_features))
                batch_stat_features.append(np.stack(category_stat_features))

            # Convert entire batch to tensors → Shape: (batch_size, categories, sentences, max_length)
            input_ids_tensor = tf.convert_to_tensor(np.stack(batch_input_ids), dtype=tf.int32)
            attention_mask_tensor = tf.convert_to_tensor(np.stack(batch_attention_masks), dtype=tf.int32)
            int_feature_tensor = tf.convert_to_tensor(np.stack(batch_int_features), dtype=tf.int32)
            stat_feature_tensor = tf.convert_to_tensor(np.stack(batch_stat_features), dtype=tf.float32)

            # Labels → One-hot encoding
            labels = to_categorical(df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)

            return input_ids_tensor, attention_mask_tensor, int_feature_tensor, stat_feature_tensor, labels

        # Call superclass function
        super().make_dataset()

        # Generate dataset tensors
        self.train_input_ids, self.train_attention_masks, self.train_int_feature, self.train_stat_feature, self.train_labels = make_inputs(self.train_df)
        self.val_input_ids, self.val_attention_masks, self.val_int_feature, self.val_stat_feature, self.val_labels = make_inputs(self.val_df)

        self.train_inputs = [self.train_input_ids, self.train_attention_masks, self.train_int_feature, self.train_stat_feature]
        self.val_inputs = [self.val_input_ids, self.val_attention_masks, self.val_int_feature, self.val_stat_feature]


        self.data_loaded = True

    def build(self):
        """
        Build a hierarchical attention model for text classification.
        - Sentence-level attention
        - Category-level attention
        - Fully connected layers for final classification
        """
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        shared_embedding_layer = DistilBERTEmbeddingLayer(bert_model=bert_model)

        # Inputs: (Batch, Categories=9, Sentences=10, MAX_LEN=256)
        input_ids = Input(shape=(PARAMS.NUM_STR_FEATURES, PARAMS.NUM_SENTENCES, PARAMS.MAX_LEN), dtype=tf.int32, name="input_ids")
        attention_masks = Input(shape=(PARAMS.NUM_STR_FEATURES, PARAMS.NUM_SENTENCES, PARAMS.MAX_LEN), dtype=tf.int32, name="attention_masks")
        int_features = Input(shape=(3, ), dtype=tf.int32, name="int_features")
        def flatten_inputs(inputs):
            input_ids, attention_masks = inputs
            shape = tf.shape(input_ids)  # Gets dynamic shape
            reshaped_input_ids = tf.reshape(input_ids, (-1, PARAMS.MAX_LEN))  # Flatten all sentences
            reshaped_attention_masks = tf.reshape(attention_masks, (-1, PARAMS.MAX_LEN))  # Flatten masks
            return reshaped_input_ids, reshaped_attention_masks, shape[0]  # Return batch_size dynamically
        
            

        # **🔹 Step 1: Flatten the category & sentence dimensions**
        reshaped_inputs = Lambda(flatten_inputs)([input_ids, attention_masks])

        # **🔹 Step 2: Pass through DistilBERT**
        sentence_embeddings = shared_embedding_layer([reshaped_inputs[0], reshaped_inputs[1]])  # (Batch * 90, 768)
        sentence_embeddings = Dense(PARAMS.REPRESENTATION_DIM, activation='tanh')(sentence_embeddings)
        # **🔹 Step 3: Restore Original Shape** `(Batch, 9, 10, 768)`
        def restore_shape(embedding):
            return tf.reshape(embedding, (-1, PARAMS.NUM_STR_FEATURES, PARAMS.NUM_SENTENCES, PARAMS.REPRESENTATION_DIM))

        sentence_embeddings = Lambda(restore_shape)(sentence_embeddings)

        # **🔹 Step 5: Apply Sentence-Level Attention Within Each Category**
        sentence_attention_layer = MultiHeadAttention(
            num_heads=PARAMS.NUM_HEADS, key_dim=PARAMS.PROJ_DIM // PARAMS.NUM_HEADS
        )
        attended_sentences = sentence_attention_layer(sentence_embeddings,
                                                      sentence_embeddings)  # (Batch, 9, 10, Hidden_Size)
        attended_sentences = LayerNormalization()(attended_sentences + sentence_embeddings)

        # **🔹 Step 6: Aggregate Sentences → Category Representation (Batch, Categories=9, Proj_Dim)**
        # sentence_reduction_layer = Dense(PARAMS.PROJ_DIM, activation='tanh')  # Projection layer
        category_embeddings = Lambda(lambda x: tf.reduce_mean(x, axis=2))(attended_sentences)  # Mean Pooling Across Sentences
        # category_embeddings = sentence_reduction_layer(category_embeddings)  # Apply projection
        stat_features = Input(shape=(PARAMS.NUM_STR_FEATURES, 5), dtype=tf.float32, name="stat_features") ## 여기에서 특징 개수 바꿔야 함.
        stat_embeddings = Dense(1, activation='sigmoid')(stat_features)
        category_embeddings = Lambda(lambda x: tf.concat([x[0], x[1]], axis=2))([category_embeddings, stat_embeddings])

        int_embeddings = Dense(PARAMS.REPRESENTATION_DIM + 1, activation='tanh')(int_features)
        int_embeddings = Lambda(lambda x: tf.expand_dims(x, axis=1))(int_embeddings)


        category_embeddings = Lambda(lambda x: tf.concat([x[0], x[1]], axis=1))([category_embeddings, int_embeddings])
        # **🔹 Step 7: Apply Category-Level Attention (Batch, Categories=9, Proj_Dim)**
        category_attention_layer = MultiHeadAttention(
            num_heads=PARAMS.NUM_HEADS, key_dim=PARAMS.PROJ_DIM // PARAMS.NUM_HEADS
        )
        attended_categories = category_attention_layer(category_embeddings, category_embeddings)
        attended_categories = LayerNormalization()(attended_categories + category_embeddings)

        
        # **🔹 Step 8: Flatten and Dense Layers**
        fl_output = Flatten()(attended_categories)
        output = Dense(1024, activation='relu')(fl_output)
        output = Dense(512, activation='relu')(output)
        output = Dense(256, activation='relu')(output)
        output = Dense(128, activation='relu')(output)

        # **🔹 Step 7: Compute Initial Bias**
        train_labels = to_categorical(self.train_df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)
        initial_bias = compute_class_biases(train_labels)

        outputs = Dense(PARAMS.NUM_CLASSES, activation='softmax', bias_initializer=Constant(initial_bias))(output)

        # **Define Model**
        self.model = Model(inputs=[input_ids, attention_masks, int_features, stat_features], outputs=outputs)

    def test(self):
        self.model = load_model('model.keras', custom_objects={'DistilBERTEmbeddingLayer': DistilBERTEmbeddingLayer})

        super().test()