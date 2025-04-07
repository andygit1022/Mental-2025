# model/proposed/proposed.py
from ..base_model import BaseModel
import tensorflow as tf
from keras.src.layers import Input, Dense, Concatenate, Layer, MultiHeadAttention, LayerNormalization, Flatten, TimeDistributed, Lambda, Reshape
from transformers import DistilBertTokenizer, TFDistilBertModel
import params as PARAMS
from keras.src.models import Model
import numpy as np
from keras.src.utils import to_categorical
from keras.src.initializers import Constant
from transformers import AutoTokenizer, TFAutoModel
from keras.src.initializers import HeNormal
from keras.src.ops import max as kmax
from keras.src.ops import mean as kmean
from sklearn.metrics import confusion_matrix
import sklearn.metrics as skm

from ..drawing import plot_attention_scores, plot_confusion_matrix
from keras.src.saving import load_model, register_keras_serializable
from keras.src.ops import Concatenate, concatenate
from keras.src.layers import concatenate
from keras.src.layers import LeakyReLU
from keras.src.layers import Dropout


@register_keras_serializable(package="Custom")
def my_concat_fn(inputs):
    """
    inputs: list/tuple of 2 tensors
    예) [(batch, 9, 255), (batch, 9, 16)]
    """
    return tf.concat(inputs, axis=2)

@register_keras_serializable(package="Custom")
def my_concat_fixed_out_shape(input_shapes):
    """
    input_shapes: [(None, 9, 255), (None, 9, 16)] 같이 들어올 것으로 가정
    여기서는 하드코딩으로 (None, 9, 271)로 반환

    ※ 실제 상황에 맞춰 조정 가능
    """
    return (None, 9, PARAMS.REPRESENTATION_DIM + PARAMS.STAT_DIM)





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
        self.bert_model.trainable = False

    def call(self, inputs):
        input_ids, attention_mask = inputs
        # ✅ Convert to int32 to match DistilBERT's expected dtype
        input_ids = tf.cast(input_ids, dtype=tf.int32)
        attention_mask = tf.cast(attention_mask, dtype=tf.int32)
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
        bert_model.trainable = False
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
            batch_input_ids = []
            batch_attention_masks = []
            batch_int_features = []
            batch_stat_features = []

            for index, row in df.iterrows():
                category_input_ids = []
                category_attention_masks = []
                category_int_features_list = []
                category_stat_features_list = []

                for category in PARAMS.FEATURES:
                    if category == "Patient_ID":
                        continue

                    if PARAMS.FULL_FEATURES[category] == 'str':
                        sentences = row[category] if isinstance(row[category], list) else []

                        # Prepare arrays for category-level input
                        cat_sent_input_ids = []
                        cat_sent_attn_masks = []
                        for i in range(PARAMS.NUM_SENTENCES):
                            if i >= len(sentences):
                                cat_sent_input_ids.append(np.zeros((PARAMS.MAX_LEN,), dtype=np.int32))
                                cat_sent_attn_masks.append(np.zeros((PARAMS.MAX_LEN,), dtype=np.int32))
                            else:
                                sent = sentences[i]
                                enc = self.tokenize_feature([sent])
                                cat_sent_input_ids.append(enc[0].numpy()[0])
                                cat_sent_attn_masks.append(enc[1].numpy()[0])

                        category_input_ids.append(np.stack(cat_sent_input_ids))
                        category_attention_masks.append(np.stack(cat_sent_attn_masks))

                        # → SS 옵션에 따라 stat features를 가져올지 말지
                        cat_stat_features = []
                        if PARAMS.SS:
                            # 문장 통계 feature 10개 (예시)
                            feats_list = [
                                # "_avg_words",
                                # "_total_sentences",
                                # "_avg_chars",
                                # "_total_numbers",
                                # "_avg_idf_weight",
                                "_polarity",
                                "_mrc_conc",
                                "_mrc_fam",
                                "_local_idf",
                                #"_tf_score"
                            ]
                            PARAMS.SENTENCE_FEATURE = len(feats_list)  # 예: 10

                            for f in feats_list:
                                c_name = category + f
                                val = tf.convert_to_tensor(row[c_name], dtype=tf.float32)
                                cat_stat_features.append(val)
                        else:
                            # SS=False → 문장 피처 없음
                            PARAMS.SENTENCE_FEATURE = 0

                        if cat_stat_features:
                            category_stat_features_list.append(np.stack(cat_stat_features))
                        else:
                            # 빈 list → shape (0,)
                            category_stat_features_list.append(np.zeros((0,), dtype=np.float32))

                    elif PARAMS.FULL_FEATURES[category] == 'int32':
                        category_int_features_list.append(tf.convert_to_tensor(row[category], dtype=tf.int32))

                batch_input_ids.append(np.stack(category_input_ids))         # shape: (9, 10, 128)
                batch_attention_masks.append(np.stack(category_attention_masks))
                batch_int_features.append(np.stack(category_int_features_list))  # (9,) int
                batch_stat_features.append(np.stack(category_stat_features_list))# (9, LEN or 0)

            # Convert entire batch to tensors
            input_ids_tensor = tf.convert_to_tensor(np.stack(batch_input_ids), dtype=tf.int32)
            attention_mask_tensor = tf.convert_to_tensor(np.stack(batch_attention_masks), dtype=tf.int32)
            int_feature_tensor = tf.convert_to_tensor(np.stack(batch_int_features), dtype=tf.int32)
            stat_feature_tensor = tf.convert_to_tensor(np.stack(batch_stat_features), dtype=tf.float32)

            # Labels → One-hot
            labels = to_categorical(df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)
            return input_ids_tensor, attention_mask_tensor, int_feature_tensor, stat_feature_tensor, labels

        # Call superclass
        super().make_dataset()

        self.train_input_ids, self.train_attention_masks, self.train_int_feature, self.train_stat_feature, self.train_labels = make_inputs(self.train_df)
        self.val_input_ids, self.val_attention_masks, self.val_int_feature, self.val_stat_feature, self.val_labels = make_inputs(self.val_df)

        self.train_inputs = [
            self.train_input_ids,
            self.train_attention_masks,
            self.train_int_feature,
            self.train_stat_feature
        ]
        self.val_inputs = [
            self.val_input_ids,
            self.val_attention_masks,
            self.val_int_feature,
            self.val_stat_feature
        ]

        self.data_loaded = True


    def build(self):
            ## Step 1: DistilBERT
        bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
        shared_embedding_layer = DistilBERTEmbeddingLayer(bert_model=bert_model)

        ## Step 2: Define Inputs
        input_ids = Input(
            shape=(PARAMS.NUM_STR_FEATURES, PARAMS.NUM_SENTENCES, PARAMS.MAX_LEN),
            dtype=tf.int32,
            name="input_ids"
        )
        attention_masks = Input(
            shape=(PARAMS.NUM_STR_FEATURES, PARAMS.NUM_SENTENCES, PARAMS.MAX_LEN),
            dtype=tf.int32,
            name="attention_masks"
        )
        int_features = Input(shape=(3,), dtype=tf.int32, name="int_features")

        # sentence_features 길이 = SS==True ? 10 : 0
        stat_features = Input(
            shape=(PARAMS.NUM_STR_FEATURES, PARAMS.SENTENCE_FEATURE),
            dtype=tf.float32,
            name="stat_features"
        )

        ## Step 3: Flatten
        @register_keras_serializable(package="Custom")
        def flatten_inputs(inputs):
            input_ids, attention_masks = inputs
            shape = tf.shape(input_ids)
            reshaped_input_ids = tf.reshape(input_ids, (-1, PARAMS.MAX_LEN))
            reshaped_attention_masks = tf.reshape(attention_masks, (-1, PARAMS.MAX_LEN))
            return reshaped_input_ids, reshaped_attention_masks, shape[0]

        reshaped_inputs = Lambda(flatten_inputs)([input_ids, attention_masks])

        ## Step 4: Sentence Embeddings
        sentence_embeddings = shared_embedding_layer(
            [reshaped_inputs[0], reshaped_inputs[1]]
        )  # (batch*90, 768)
        #sentence_embeddings = Dense(PARAMS.REPRESENTATION_DIM, activation='tanh')(sentence_embeddings)
        
        sentence_embeddings = Dense(PARAMS.REPRESENTATION_DIM)(sentence_embeddings)
        sentence_embeddings = LeakyReLU(alpha=0.01)(sentence_embeddings)

        ## Step 5: Restore shape
        @register_keras_serializable(package="Custom")
        def restore_shape(embedding):
            return tf.reshape(embedding, (-1, PARAMS.NUM_STR_FEATURES, PARAMS.NUM_SENTENCES, PARAMS.REPRESENTATION_DIM))

        sentence_embeddings = Lambda(restore_shape)(sentence_embeddings)

        ## Step 6: Self-Attention (Sentence)
        sentence_attention_layer = MultiHeadAttention(
            num_heads=PARAMS.NUM_HEADS,
            key_dim=PARAMS.PROJ_DIM // PARAMS.NUM_HEADS,
        )
        attended_sentences = sentence_attention_layer(sentence_embeddings, sentence_embeddings)
        attended_sentences = LayerNormalization()(attended_sentences + sentence_embeddings)
        #attended_sentences = Dropout(0.2)(attended_sentences)

        ## Step 7: Max Pool + Dense
        sentence_pooling = Dense(PARAMS.REPRESENTATION_DIM, activation='tanh')(attended_sentences)

        @register_keras_serializable(package="Custom")
        class ReduceMaxLayer(tf.keras.layers.Layer):
            def call(self, inputs):
                return kmax(inputs, axis=2)  # (batch,9,255)

            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[1], input_shape[3])
        
        
        class ReduceMeanLayer(tf.keras.layers.Layer):
            def call(self, inputs):
                return kmean(inputs, axis=2)
            
            def compute_output_shape(self, input_shape):
                return (input_shape[0], input_shape[1], input_shape[3])
        
        category_embeddings = ReduceMeanLayer()(sentence_pooling)
        #category_embeddings = ReduceMaxLayer()(sentence_pooling)

        ## Step 8: Feature Fusion
        if PARAMS.SS:
            # (batch,9, 10) -> Dense(16) => (batch,9,16)
            stat_embeddings = Dense(PARAMS.STAT_DIM, activation='relu', kernel_initializer=HeNormal())(stat_features)

            # concat => (batch,9,255+16=271)
            @register_keras_serializable(package="Custom")
            def concat_271_out_shape(input_shapes):
                # ex: [(None,9,255),(None,9,16)] => (None,9,271)
                return (None, 9, PARAMS.REPRESENTATION_DIM + PARAMS.STAT_DIM)

            cat_with_stat = Lambda(
                my_concat_fn,
                output_shape=concat_271_out_shape,
                name="concat_category_stat"
            )([category_embeddings, stat_embeddings])

            # int_features => Dense(255+16=271)
            int_embeddings = Dense(PARAMS.REPRESENTATION_DIM + PARAMS.STAT_DIM, activation='tanh', kernel_initializer=HeNormal())(int_features)
            final_feat_dim = PARAMS.REPRESENTATION_DIM + PARAMS.STAT_DIM

            category_embeddings = cat_with_stat

        else:
            # SS==False => skip stat_features entirely
            # keep category_embeddings = (batch,9,255)
            # int_features => 255
            int_embeddings = Dense(PARAMS.REPRESENTATION_DIM, activation='tanh', kernel_initializer=HeNormal())(int_features)
            final_feat_dim = PARAMS.REPRESENTATION_DIM

        # expand dim => (batch,1, final_feat_dim)
        @register_keras_serializable(package="Custom")
        def expand_dim_axis_1_fn(x):
            return tf.expand_dims(x, axis=1)

        @register_keras_serializable(package="Custom")
        def expand_dim_shape_1(input_shape):
            # ex: (None, 271) => (None,1,271)
            return (None, 1, final_feat_dim)

        int_embeddings = Lambda(
            expand_dim_axis_1_fn,
            output_shape=expand_dim_shape_1,
            name="expand_dim_int"
        )(int_embeddings)

        # concat => (batch, 10, final_feat_dim)
        @register_keras_serializable(package="Custom")
        def concat_cat_int_fn(inputs):
            return tf.concat(inputs, axis=1)
        @register_keras_serializable(package="Custom")
        def concat_cat_int_shape(input_shapes):
            # ex: [(None,9, X),(None,1,X)] => (None,10,X)
            return (None, 10, final_feat_dim)

        category_embeddings = Lambda(
            concat_cat_int_fn,
            output_shape=concat_cat_int_shape,
            name="concat_cat_int"
        )([category_embeddings, int_embeddings])

        ## Step 9: Category-level Attention
        category_attention_layer = MultiHeadAttention(
            num_heads=PARAMS.NUM_HEADS,
            key_dim=PARAMS.PROJ_DIM // PARAMS.NUM_HEADS,
        )
        attended_categories, cat_att_scores = category_attention_layer(
            category_embeddings,
            category_embeddings,
            return_attention_scores=True
        )
        attended_categories = LayerNormalization()(attended_categories + category_embeddings)

        ## Step 11: Flatten & Final Classifier
        fl_output = Flatten()(attended_categories)
        output = Dense(1024, activation='gelu', kernel_initializer=HeNormal())(fl_output)
        output = Dropout(0.1)(output)
        output = Dense(256, activation='gelu', kernel_initializer=HeNormal())(output)
        #output = Dropout(0.1)(output)
        output = Dense(64, activation='gelu', kernel_initializer=HeNormal())(output)

        # Step 12: Adaptive bias
        train_labels = to_categorical(self.train_df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)
        initial_bias = compute_class_biases(train_labels)

        outputs = Dense(PARAMS.NUM_CLASSES, activation='softmax',
                        bias_initializer=Constant(initial_bias))(output)

        self.model = Model(inputs=[input_ids, attention_masks, int_features, stat_features],
                        outputs=outputs)



    # def build(self):
    #     """
    #     Improved Hierarchical Attention Model:
    #     - Sentence-Level Self-Attention (within each category)
    #     - Category-Level Self-Attention (across categories)
    #     - Enhanced feature fusion for integer & statistical features
    #     - Transformer-based classification instead of dense layers
    #     """

    #     ## 🟢 **Step 1: DistilBERT for Sentence Embeddings**
    #     bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')
    #     shared_embedding_layer = DistilBERTEmbeddingLayer(bert_model=bert_model)

    #     ## 🟢 **Step 2: Define Inputs**
    #     input_ids = Input(shape=(PARAMS.NUM_STR_FEATURES, PARAMS.NUM_SENTENCES, PARAMS.MAX_LEN), dtype=tf.int32,
    #                       name="input_ids")
    #     attention_masks = Input(shape=(PARAMS.NUM_STR_FEATURES, PARAMS.NUM_SENTENCES, PARAMS.MAX_LEN), dtype=tf.int32,
    #                             name="attention_masks")
    #     int_features = Input(shape=(3,), dtype=tf.int32, name="int_features")  # Integer Features
    #     stat_features = Input(shape=(PARAMS.NUM_STR_FEATURES, PARAMS.SENTENCE_FEATURE), dtype=tf.float32,
    #                           name="stat_features")  # Statistical Features

    #     ## 🟢 **Step 3: Flatten Inputs for Efficient BERT Processing**
    #     @register_keras_serializable(package="Custom")
    #     def flatten_inputs(inputs):
    #         input_ids, attention_masks = inputs
    #         shape = tf.shape(input_ids)
    #         reshaped_input_ids = tf.reshape(input_ids, (-1, PARAMS.MAX_LEN))
    #         reshaped_attention_masks = tf.reshape(attention_masks, (-1, PARAMS.MAX_LEN))
    #         return reshaped_input_ids, reshaped_attention_masks, shape[0]

    #     reshaped_inputs = Lambda(flatten_inputs)([input_ids, attention_masks])

    #     ## 🟢 **Step 4: Get Sentence Embeddings**
    #     sentence_embeddings = shared_embedding_layer(
    #         [reshaped_inputs[0], reshaped_inputs[1]])  # Shape: (Batch * 90, 768)
    #     #sentence_embeddings = Dense(PARAMS.REPRESENTATION_DIM, activation='tanh')(sentence_embeddings) #tanh
    #     sentence_embeddings = Dense(PARAMS.REPRESENTATION_DIM)(sentence_embeddings)
    #     sentence_embeddings = LeakyReLU(alpha=0.01)(sentence_embeddings)

    #     ## 🟢 **Step 5: Restore Original Shape** `(Batch, 9, 10, Hidden_Size)`
    #     @register_keras_serializable(package="Custom")
    #     def restore_shape(embedding):
    #         return tf.reshape(embedding, (-1, PARAMS.NUM_STR_FEATURES, PARAMS.NUM_SENTENCES, PARAMS.REPRESENTATION_DIM))

    #     sentence_embeddings = Lambda(restore_shape)(sentence_embeddings)

    #     ## 🔵 **Step 6: Apply Self-Attention to Sentences Within Each Category**
    #     sentence_attention_layer = MultiHeadAttention(
    #         num_heads=PARAMS.NUM_HEADS, key_dim=PARAMS.PROJ_DIM // PARAMS.NUM_HEADS
    #     )
    #     attended_sentences = sentence_attention_layer(sentence_embeddings, sentence_embeddings)
    #     attended_sentences = LayerNormalization()(attended_sentences + sentence_embeddings)

    #     ## 🔵 **Step 7: Improved Sentence Aggregation (Max Pooling + Dense)**
    #     sentence_pooling = Dense(PARAMS.REPRESENTATION_DIM, activation='tanh')(attended_sentences)
        
    #     @register_keras_serializable(package="Custom")
    #     class ReduceMaxLayer(tf.keras.layers.Layer):
    #         def call(self, inputs):
    #             # Keras Backend의 max 함수를 써서 KerasTensor를 다룸
    #             return kmax(inputs, axis=2)  

    #         def compute_output_shape(self, input_shape):
    #             # batch dim: input_shape[0] -> None
    #             # axis=2 제거 -> (None, 9, 255)
    #             return (input_shape[0], input_shape[1], input_shape[3])

    #     category_embeddings = ReduceMaxLayer()(sentence_pooling)
        
        
    #     #category_embeddings = Lambda(lambda x: tf.reduce_max(x, axis=2))(sentence_pooling)
    #     # category_embeddings = tf.reduce_max(sentence_pooling, axis=2)  # Max Pooling
        
        

    #     ## 🟡 **Step 8: Feature Fusion (Integer & Statistical Features)**
    #     if PARAMS.SS:
    #         stat_embeddings = Dense(16, activation='relu', kernel_initializer=HeNormal())(stat_features)
            
    #         category_embeddings = Lambda(
    #             my_concat_fn,  # 위에서 정의한 함수
    #             output_shape=my_concat_fixed_out_shape,  # 하드코딩 (None, 9, 271)
    #             name="concat_category_stat"
    #         )([category_embeddings, stat_embeddings])
    #         #category_embeddings = Lambda(lambda x: tf.concat(x, axis=2),output_shape=(PARAMS.NUM_STR_FEATURES, 271))([category_embeddings, stat_embeddings]) ####
    #         int_embeddings = Dense(PARAMS.REPRESENTATION_DIM + 16, activation='tanh', kernel_initializer=HeNormal())(
    #             int_features)
            
    #     else:
    #         int_embeddings = Dense(PARAMS.REPRESENTATION_DIM, activation='tanh', kernel_initializer=HeNormal())(
    #             int_features)

    #     # int_embeddings = tf.expand_dims(int_embeddings, axis=1)
        
        
        
    #     @register_keras_serializable(package="Custom")
    #     def expand_dim_axis_1_fn(x):
    #         return tf.expand_dims(x, axis=1)

    #     @register_keras_serializable(package="Custom")
    #     def expand_dim_axis_1_out_shape(input_shape):
    #         # 예: input_shape = (None, 271) -> (None, 1, 271)
    #         return (None, 1, 271)
        
        
    #     int_embeddings = Lambda(
    #         expand_dim_axis_1_fn,
    #         output_shape=expand_dim_axis_1_out_shape,
    #         name="expand_dim_int"
    #     )(int_embeddings)
    #     # category_embeddings = tf.concat([category_embeddings, int_embeddings], axis=1)
    #     #category_embeddings = Lambda(lambda x: tf.concat(x, axis=1),output_shape=(PARAMS.NUM_STR_FEATURES + 1, 271))([category_embeddings, int_embeddings])

    #     # @register_keras_serializable(package="Custom")
    #     # def concat_cat_int_fn(inputs):
    #     #     return tf.concat(inputs, axis=1)

    #     # @register_keras_serializable(package="Custom")
    #     # def concat_cat_int_shape(input_shapes):
    #     #     # 예: [(None, 9, 271), (None, 1, 271)] -> (None, 10, 271)
    #     #     return (None, 10, 271)
        
        
    #     @register_keras_serializable(package="Custom")
    #     def concat_cat_int_fn(inputs):
    #         return tf.concat(inputs, axis=1)

    #     @register_keras_serializable(package="Custom")
    #     def concat_cat_int_shape(input_shapes):
    #         # 예: [(None, 9, 271), (None, 1, 271)] -> (None, 10, 271)
    #         return (None, 10, 271)
    #     category_embeddings = Lambda(
    #         concat_cat_int_fn,
    #         output_shape=concat_cat_int_shape,
    #         name="concat_cat_int"
    #     )([category_embeddings, int_embeddings])
        
        
    #     ## 🟠 **Step 9: Apply Category-Level Attention (Across All Categories)**
    #     category_attention_layer = MultiHeadAttention(
    #         num_heads=PARAMS.NUM_HEADS, key_dim=PARAMS.PROJ_DIM // PARAMS.NUM_HEADS
    #     )
        
    #     #category_attention_layer.call(return_attention_scores=)
    #     attended_categories, cat_att_scores = category_attention_layer(
    #         category_embeddings, 
    #         category_embeddings,
    #         return_attention_scores=True
    #     )
    #     attended_categories = LayerNormalization()(attended_categories + category_embeddings)

    #     ## 🔴 **Step 11: Flatten and Final Classifier**
    #     fl_output = Flatten()(attended_categories)
    #     output = Dense(1024, activation='gelu', kernel_initializer=HeNormal())(fl_output)
    #     output = Dense(256, activation='gelu', kernel_initializer=HeNormal())(output)
    #     output = Dense(64, activation='gelu', kernel_initializer=HeNormal())(output)
        
        
    #     # output = Dense(1024, activation='relu', kernel_initializer=HeNormal())(fl_output)
    #     # output = Dense(512, activation='relu', kernel_initializer=HeNormal())(output)
    #     # output = Dense(256, activation='relu', kernel_initializer=HeNormal())(output)
    #     # output = Dense(128, activation='relu', kernel_initializer=HeNormal())(output)

    #     ## 🔴 **Step 12: Adaptive Bias for Imbalanced Classification**
    #     train_labels = to_categorical(self.train_df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)
    #     initial_bias = compute_class_biases(train_labels)

    #     outputs = Dense(PARAMS.NUM_CLASSES, activation='softmax',
    #                     bias_initializer=Constant(initial_bias))(output)

    #     ## **Define Model**
    #     self.model = Model(inputs=[input_ids, attention_masks, int_features, stat_features], outputs=outputs)

    def test(self):
        """
        1) 저장된 모델(my_model.keras)을 로드
        2) 데이터셋 준비 (미리 안됐으면 make_dataset() 호출)
        3) 'multi_head_attention_1' 레이어를 submodel로 추출해 카테고리 어텐션 맵 획득
        4) 최종 분류 결과 계산 & confusion matrix + classification report
        5) (선택) 오분류 케이스 확인
        """
        # 1) 모델 로드
        self.model = load_model(
            PARAMS.MODEL_PATH,
            safe_mode=False,
            custom_objects={'DistilBERTEmbeddingLayer': DistilBERTEmbeddingLayer,
                            "my_concat_fn": my_concat_fn,
                            "my_concat_fixed_out_shape": my_concat_fixed_out_shape,
            }
        )
        
        # 2) 데이터셋 준비 (필요 시)
        if not self.data_loaded:
            self.make_dataset()

        # 3) Submodel을 이용해 카테고리 레벨 Attention map 얻기
        #    build()에서 'category_attention_layer' = MultiHeadAttention(return_attention_scores=True)
        #    Keras가 자동으로 이름을 "multi_head_attention_1"처럼 붙였을 것이므로, 해당 레이어를 get_layer로 찾습니다.
        category_attn_layer = self.model.get_layer("multi_head_attention_1")

        attention_out_model = Model(
            inputs=self.model.inputs,
            outputs=category_attn_layer.output  # (attended_categories, cat_att_scores)
        )

        # (3-1) Validation 세트에 대해 Attention 추론
        val_attn_out = attention_out_model.predict(self.val_inputs, verbose=1)
        # val_attn_out = [attended_categories, cat_att_scores]
        attended_tensor = val_attn_out[0]      # shape: (batch, seq_len, dim)
        cat_att_scores  = val_attn_out[1]      # shape: (batch, num_heads, seq_len, seq_len)

        print(f"[DEBUG] attended_tensor.shape = {attended_tensor.shape}")
        print(f"[DEBUG] cat_att_scores.shape  = {cat_att_scores.shape}")
        # 예: (batch_size, 10, 271), (batch_size, 4, 10, 10) 등

        # (3-2) Attention Map 평균: (batch, heads) → 단일 (seq_len, seq_len)
        # seq_len = 텍스트 카테고리 9개 + int_feature 1개 = 10 (혹은 12+1=13 등)
        att_map = cat_att_scores.mean(axis=(0,1))  # shape: (seq_len, seq_len)

        # (3-3) 시각화 (option)
        plot_attention_scores(att_map, feature_labels=PARAMS.FEATURES[1:])
        # NOTE: feature_labels 개수와 seq_len이 맞도록 조정!
        # PARAMS.FEATURES[1:] 를 쓰면 'Patient_ID'를 뺀 12개를 쓴다는 의미이므로,
        # 실제로는 9개 + 1 = 10개인지 12개 + 1 = 13개인지 확인 필요.

        # 4) 최종 분류 결과 & confusion matrix
        val_preds = self.model.predict(self.val_inputs, verbose=1)  # (batch, NUM_CLASSES)
        y_pred = tf.argmax(val_preds, axis=1).numpy()
        y_true = tf.argmax(self.val_labels, axis=1).numpy()

        cm = confusion_matrix(y_true, y_pred)
        print(skm.classification_report(y_true, y_pred))
        plot_confusion_matrix(cm, classes=PARAMS.CLASSES, title='Confusion Matrix')

        # 5) 오분류(Misclassification) 사례 확인
        diff_idx = np.where(y_pred != y_true)[0]
        if len(diff_idx) == 0:
            print("No misclassifications found.")
        else:
            print("\nMisclassifications:")
            
            mis_df = self.val_df.iloc[diff_idx].copy()
            mis_df["Pred_Label"] = [PARAMS.CLASSES[i] for i in y_pred[diff_idx]]
            
            print(mis_df[["Patient_ID","Label","Pred_Label","Age"]].to_string(index=False))
            #print(self.val_df.iloc[diff_idx][["Patient_ID", "Label", "Age"]].to_string(index=False))
