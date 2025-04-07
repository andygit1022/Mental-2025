########### model/llama/llama_classifier.py ##############
# from ..base_model import BaseModel
import params as PARAMS
import numpy as np
from transformers import LlamaTokenizer, LlamaForSequenceClassification, LlamaConfig, LlamaForCausalLM, LlamaModel, \
    LlamaPreTrainedModel
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModel, LlamaForCausalLM
# from tensorflow.python.keras.utils.np_utils import to_categorical
from peft import LoraConfig, get_peft_model, TaskType
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm  # For progress bar (optional)
import numpy as np
import torchmetrics
from torch.utils.data import TensorDataset, DataLoader
import matplotlib
import matplotlib.pyplot as plt
import os
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
import torch.nn.functional as F


# # Check if CUDA is available
# print("CUDA available:", torch.cuda.is_available())
# # Check if CUDA is available
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#
# # Print the name of the CUDA device
# if torch.cuda.is_available():
#     print("CUDA device name:", torch.cuda.get_device_name(0))
#     print("CUDA device count:", torch.cuda.device_count())
#     print("Current CUDA device:", torch.cuda.current_device())
# else:
#     print("No CUDA devices found.")


def compute_class_biases(labels):
    # Assuming labels are one-hot encoded, calculate the class distribution
    class_totals = np.sum(labels, axis=0)
    class_probs = class_totals / np.sum(class_totals)

    # Calculate logit bias: log(p / (1 - p)) for each class
    initial_bias = np.log(class_probs / (1 - class_probs))
    return initial_bias


tokenizer_path = "meta-llama/Llama-3.1-8B"
llama_model_path = "meta-llama/Llama-3.1-8B"
#tokenizer_path = "model/llama/Llama-3.1-8B-hf"
#llama_model_path = "model/llama/Llama-3.1-8B-hf"

class LlamaEmbeddingLayer(nn.Module):
    def __init__(self, llama_model, device, hidden_dim=4096, pooling_mode="attention", **kwargs):
        """
        pooling_mode 로 어떤 방식을 쓸지 결정:
          - "attention": Attention-weighted Pooling (Self-Attentive)
          - "learnable": MLP를 통해 점수를 구하는 Learnable Pooling
          - "proj"     : Downstream Projection Layer (mean pooling 후 MLP)
          - "mlp_after_pooling": (Max Pooling) → MLP(4096→1024→256→64→1)
          - "query_attn": Self-Attention Pooling (학습 Query 사용)

        그 외 필요한 kwargs는 프로젝트 상황에 맞춰 추가할 수 있습니다.
        """
        super(LlamaEmbeddingLayer, self).__init__(**kwargs)
        self.llama_model = llama_model
        self.device = device
        self.hidden_dim = hidden_dim
        self.pooling_mode = pooling_mode
        print("[Pooling method] : " + pooling_mode)

        # -----------------------------------------------------
        # (A) Attention-weighted Pooling에 쓰일 모듈 (1번) : BERT의 CLS와 유사
        # -----------------------------------------------------
        self.attn_score_proj = nn.Linear(hidden_dim, 1)
        self.attn_score_proj = self.attn_score_proj
        # self.attn_score_proj = nn.Linear(hidden_dim, 1, dtype=torch.float16)
        
        # -----------------------------------------------------
        # (B) Learnable Pooling (2번) - MLP로 score 계산
        # -----------------------------------------------------
        self.learnable_score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),  # 중간 채널 임의(512)
            nn.Tanh(),
            nn.Linear(512, 1)
        )

        # -----------------------------------------------------
        # (D) Downstream Projection Layer (4번)
        #     (Mean Pooling) → (4096→1024→256)
        # -----------------------------------------------------
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )

        # -----------------------------------------------------
        # (E) Pooling 이후 MLP (5번): (Max Pool) → (4096→1024→256→64→1)
        # -----------------------------------------------------
        self.pooling_mlp = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 최종 1차원(예: 로짓)
        )

        # -----------------------------------------------------
        # (F) Self-Attention으로 Pooling (6번) - Learnable Query
        # -----------------------------------------------------
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # (1,1,4096)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, input_ids, attention_mask=None):
        # GPU/Device 매핑
        # input_ids = input_ids.to(dtype=torch.long, device=self.device)
        # if attention_mask is not None:
        #     attention_mask = attention_mask.to(self.device)

        # LLaMA Forward
        outputs = self.llama_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # hidden_states: [batch_size, seq_len, hidden_dim]
        hidden_states = outputs.hidden_states[-1]

        # -----------------------
        # 0) CLS 토큰 임베딩 예시
        # -----------------------
        # if self.pooling_mode == "cls":
        #     # LLaMA는 공식적으로 [CLS] 토큰이 없지만,
        #     # 예시로 첫 토큰(:,0,:)이나 끝 토큰(:,-1,:)을 쓰는 경우가 있음
        #     embedding = hidden_states[:, 0, :]  # [CLS]라고 가정할 수도...
        #     return embedding

        # -----------------------
        # 1) Mean Pooling
        # -----------------------
        if self.pooling_mode == "mean":
            if attention_mask is None:
                # 단순 Mean Pooling
                embedding = hidden_states.mean(dim=1)
            else:
                mask_f = attention_mask.float()  # (B, L)
                masked_hidden = hidden_states * mask_f.unsqueeze(-1)  # (B, L, D)
                sum_hidden = masked_hidden.sum(dim=1)                 # (B, D)
                denom = mask_f.sum(dim=1, keepdim=True) + 1e-9
                embedding = sum_hidden / denom                        # (B, D)
            return embedding

        # -----------------------
        # 2) Max Pooling
        # -----------------------
        if self.pooling_mode == "max":
            if attention_mask is None:
                embedding, _ = hidden_states.max(dim=1)
            else:
                inf_mask = (1 - attention_mask) * (-1e9)
                inf_mask = inf_mask.unsqueeze(-1)     # (B, L, 1)
                masked_hidden = hidden_states + inf_mask
                embedding, _ = masked_hidden.max(dim=1)
            return embedding

        # ----------------------------------------------------------------
        # 3) Linear Attention Pooling [기존 코드와 유사]
        # ----------------------------------------------------------------
        if self.pooling_mode == "attention":
            if attention_mask is not None:
                # (B, L, 1)
                scores = self.attn_score_proj(hidden_states)          # -> (B, L, 1)
                scores = scores.squeeze(-1)                           # (B, L)
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))
                attn_weights = F.softmax(scores, dim=1)               # (B, L)
                
                # (B, L, 1)
                attn_weights = attn_weights.unsqueeze(-1)
            
                weighted_hidden = hidden_states * attn_weights        # (B, L, D)
                embedding = weighted_hidden.sum(dim=1)                # (B, D)
            
            else:
                scores = self.attn_score_proj(hidden_states).squeeze(-1)  # (B, L)
                attn_weights = F.softmax(scores, dim=1)                    # (B, L)
                attn_weights = attn_weights.unsqueeze(-1)                  # (B, L, 1)
                weighted_hidden = hidden_states * attn_weights             # (B, L, D)
                embedding = weighted_hidden.sum(dim=1)                     # (B, D)
                
            return embedding

        # ---------------------------------------
        # MLP-based attention pooling
        # ---------------------------------------
        if self.pooling_mode == "learnable":
            # MLP로 score 계산
            scores = self.learnable_score_mlp(hidden_states).squeeze(-1)  # (B, L)
            if attention_mask is not None:
                scores = scores.masked_fill(attention_mask == 0, float('-inf'))

            attn_weights = F.softmax(scores, dim=1)  # (B, L)
            attn_weights = attn_weights.unsqueeze(-1)
            weighted_hidden = hidden_states * attn_weights
            embedding = weighted_hidden.sum(dim=1)
            return embedding


        # ---------------------------------------
        # (4) Downstream Projection Layer
        #     (Mean Pooling -> MLP)
        # ---------------------------------------
        if self.pooling_mode == "proj":
            # 먼저 Mean Pooling
            if attention_mask is not None:
                mask_f = attention_mask.float()
                masked_hidden = hidden_states * mask_f.unsqueeze(-1)
                sum_hidden = masked_hidden.sum(dim=1)
                denom = mask_f.sum(dim=1, keepdim=True) + 1e-9
                pooled = sum_hidden / denom
            else:
                pooled = hidden_states.mean(dim=1)
            # (hidden_dim->1024->256)
            embedding = self.proj(pooled)  # (B, 256)
            return embedding

        # ---------------------------------------
        # (5) Pooling 이후 MLP (Max Pool -> 4096->1024->256->64->1)
        # ---------------------------------------
        if self.pooling_mode == "mlp_after_pooling":
            # Max Pool 먼저
            if attention_mask is None:
                pooled, _ = hidden_states.max(dim=1)  # (B, D=4096)
            else:
                inf_mask = (1 - attention_mask) * (-1e9)
                inf_mask = inf_mask.unsqueeze(-1)  # (B, L, 1)
                masked_hidden = hidden_states + inf_mask
                pooled, _ = masked_hidden.max(dim=1)
            # MLP
            logits = self.pooling_mlp(pooled)  # (B, 1)
            return logits  # 필요에 따라 (B, 1)을 embedding or output으로 사용

        # ---------------------------------------
        # (6) Self-Attention Pooling (Learnable Query)
        # ---------------------------------------
        if self.pooling_mode == "query_attn":
            B = hidden_states.size(0)
            # query: (1, 1, D) -> (B, 1, D)
            query_batch = self.query.expand(B, -1, -1)

            Q = self.W_q(query_batch)      # (B,1,D)
            K = self.W_k(hidden_states)    # (B,L,D)
            V = self.W_v(hidden_states)    # (B,L,D)

            # (B,1,D) x (B,D,L) = (B,1,L)
            attn_scores = torch.bmm(Q, K.transpose(1,2)) / (self.hidden_dim ** 0.5)
            if attention_mask is not None:
                attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1)==0, float('-inf'))

            attn_weights = F.softmax(attn_scores, dim=-1)  # (B,1,L)
            # (B,1,D) = (B,1,L) x (B,L,D)
            pooled = torch.bmm(attn_weights, V)
            embedding = pooled.squeeze(1)  # (B, D)
            return embedding

        # ---------------------------------------------------
        # 디폴트: attention-weighted pooling 과 동일 처리
        # ---------------------------------------------------
        scores = self.attn_score_proj(hidden_states).squeeze(-1)
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask == 0, float('-inf'))
        attn_weights = F.softmax(scores, dim=1)
        attn_weights = attn_weights.unsqueeze(-1)
        weighted_hidden = hidden_states * attn_weights
        embedding = weighted_hidden.sum(dim=1)
        return embedding


class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        #self.dropout = nn.Dropout(p=0.2)  # 원하는 확률로 설정 (예: 0.1, 0.2 등)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x, need_weights=False):
        x = x.to(self.attention.in_proj_weight.device)
        attn_output, attn_weights = self.attention(x, x, x, need_weights=need_weights)
        out = self.layer_norm(attn_output + x)
        # attn_weights shape = [batch_size, num_heads, seq_len, seq_len] (batch_first=True)
        return out, attn_weights
    
    # def forward(self, x):
    #     x = x.to(self.attention.in_proj_weight.device)
    #     attn_output, _ = self.attention(x, x, x)
    #     return self.layer_norm(attn_output + x)

class LlamaClassifier(nn.Module):

    def get_model(self):
        return self.model    
    
    def __init__(self, df, device, world_size, rank, hidden_dim=4096, pooling_mode="attention", *args, **kwargs):
        super().__init__(*args, **kwargs)
        (self.train_df, self.val_df) = df

        self.device = device
        self.world_size = world_size
        self.rank = rank
        self.pooling_mode = pooling_mode
        self.hidden_dim = hidden_dim
        
        
        #self.dropout = nn.Dropout(p=0.2)

        self.train_labels = torch.nn.functional.one_hot(
            torch.tensor(self.train_df['label_encoded'].values, dtype=torch.long),
            num_classes=PARAMS.NUM_CLASSES  # Adjust as needed
        ).float()

        self.val_labels = torch.nn.functional.one_hot(
            torch.tensor(self.val_df['label_encoded'].values, dtype=torch.long),
            num_classes=PARAMS.NUM_CLASSES  # Adjust as needed
        ).float()

        self.train_labels = torch.tensor(self.train_labels, dtype=torch.float16).unsqueeze(1)
        self.val_labels = torch.tensor(self.val_labels, dtype=torch.float16).unsqueeze(1)
        self.train_labels = self.train_labels.squeeze(1)
        self.val_labels = self.val_labels.squeeze(1)

        # self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path, legacy=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        # Load LLaMA model for embeddings
        base_model = AutoModelForCausalLM.from_pretrained(
            llama_model_path,
            torch_dtype=torch.float16,
            device_map={"": self.device}
        )

        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            #r=2,  # Low-rank dimension
            r=4,
            lora_alpha=64,
            lora_dropout=0.1,
            target_modules=["o_proj"],  # Apply LoRA to specific layers
            # target_modules=["k_proj"],  # Apply LoRA to specific layers
        )

        # Add LoRA to the model
        self.model = get_peft_model(base_model, lora_config)
        # # Distributed DataParallel (DDP)
        # self.model = DDP(self.model, device_ids=[device], output_device=device)

        # for name, param in self.model.named_parameters():
        #     if "lora" in name or "v_proj" in name:
        #         param.requires_grad = True
        #         print(f"{name}: Trainable")
        #     else:
        #         param.requires_grad = False  # Freeze all other parameters

        # Debug trainable parameters
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        print(f"Trainable params: {trainable_params} | Total params: {total_params}")
        # self.model.load_state_dict(torch.load(llama_model_path, weights_only=True))

        # weights = torch.load(llama_model_path, mmap=True, map_location="cuda")
        # self.model.load_state_dict(checkpoint)
        self.embedding_layer = LlamaEmbeddingLayer(llama_model=self.model, device=device,pooling_mode=PARAMS.POOLING_MODE)

                # -----------------------------------------------------
        # (A) Attention-weighted Pooling에 쓰일 모듈 (1번) : BERT의 CLS와 유사
        # -----------------------------------------------------
        self.attn_score_proj = nn.Linear(hidden_dim, 1)
        self.attn_score_proj = self.attn_score_proj
        # self.attn_score_proj = nn.Linear(hidden_dim, 1, dtype=torch.float16)
        
        # -----------------------------------------------------
        # (B) Learnable Pooling (2번) - MLP로 score 계산
        # -----------------------------------------------------
        self.learnable_score_mlp = nn.Sequential(
            nn.Linear(hidden_dim, 512),  # 중간 채널 임의(512)
            nn.Tanh(),
            nn.Linear(512, 1)
        )

        # -----------------------------------------------------
        # (D) Downstream Projection Layer (4번)
        #     (Mean Pooling) → (4096→1024→256)
        # -----------------------------------------------------
        self.proj = nn.Sequential(
            nn.Linear(hidden_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU()
        )
        
        
        
        
        

        # -----------------------------------------------------
        # (E) Pooling 이후 MLP (5번): (Max Pool) → (4096→1024→256→64→1)
        # -----------------------------------------------------
        self.pooling_mlp = nn.Sequential(
            nn.Linear(4096, 1024),
            nn.ReLU(),
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)  # 최종 1차원(예: 로짓)
        )

        # -----------------------------------------------------
        # (F) Self-Attention으로 Pooling (6번) - Learnable Query
        # -----------------------------------------------------
        self.query = nn.Parameter(torch.randn(1, 1, hidden_dim))  # (1,1,4096)
        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Define projection layers for string and integer features
        self.feature_projections = nn.ModuleDict()
        for feature, feature_type in PARAMS.FULL_FEATURES.items():
            feature_key = feature.replace(" ", "_")
            if feature_type == 'str' and feature != "Patient_ID":
                self.feature_projections[feature_key] = nn.Linear(4096, PARAMS.MAX_LEN)
            elif feature_type == 'int16':
                self.feature_projections[feature_key] = nn.Linear(1, PARAMS.MAX_LEN)

        # Attention layer
        self.attention_layer = MultiHeadAttentionLayer(embed_dim=PARAMS.MAX_LEN, num_heads=1)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(PARAMS.MAX_LEN*(len(PARAMS.FEATURES)-1), PARAMS.MAX_LEN).to(device)
        self.output = nn.Linear(PARAMS.MAX_LEN, PARAMS.NUM_CLASSES).to(device)
        # self.fc2 = nn.Linear(128, 64).to(device)
        #self.fc3 = nn.Linear(64, 32).to(device)
        #self.output = nn.Linear(32, 2).to(device)

    def tokenize_feature(self, texts, max_length=PARAMS.MAX_LEN):
        encoding = self.tokenizer(
            list(texts),
            max_length=128,
            padding="max_length",  # Ensures fixed length of max_length
            truncation=True,
            return_tensors="pt"
        )        
        # encoding["input_ids"] = torch.clamp(encoding["input_ids"], min=0, max=self.tokenizer.vocab_size - 1)

        # Return tensors directly
        # return encoding["input_ids"], encoding["attention_mask"]
        return {
            'input_ids': encoding["input_ids"].to(self.device),  # Ensure torch.int64
            'attention_mask': encoding["attention_mask"].to(self.device)
        }

    def make_dataset(self):
        train_encodings = {
            feature: self.tokenize_feature(self.train_df[feature])
            for feature in PARAMS.FEATURES if PARAMS.FULL_FEATURES[feature] == 'str'
        }

        val_encodings = {
            feature: self.tokenize_feature(self.val_df[feature])
            for feature in PARAMS.FEATURES if PARAMS.FULL_FEATURES[feature] == 'str'
        }

        train_inputs = []
        val_inputs = []

        # string features
        for feature in PARAMS.FEATURES:
            if feature == "Patient_ID":
                continue
            # Remove spaces in feature names for compatibility with input layer names
            feature_key = feature.replace(" ", "_")

            if PARAMS.FULL_FEATURES[feature] == 'str':
                # Add input IDs and attention masks for train and val to their respective lists
                train_inputs.append(train_encodings[feature]["input_ids"]) # input IDs for train
                train_inputs.append(train_encodings[feature]["attention_mask"])  # attention mask for train
                val_inputs.append(val_encodings[feature]["input_ids"])  # input IDs for val
                val_inputs.append(val_encodings[feature]["attention_mask"])  #

            elif PARAMS.FULL_FEATURES[feature] == 'int16':
                # Add integer features to train and val lists
                
                train_inputs.append(torch.tensor(self.train_df[feature].values, dtype=torch.float16).unsqueeze(-1))
                val_inputs.append(torch.tensor(self.val_df[feature].values, dtype=torch.float16).unsqueeze(-1))

        return train_inputs, self.train_labels, val_inputs, self.val_labels

        self.train_labels = self.train_labels.to(self.device, dtype=torch.float16)
        self.val_labels = self.val_labels.to(self.device, dtype=torch.float16)

        # for data in train_inputs:
        #     print(f"Input Data Stats: min={data.data.min()}, max={data.data.max()}")
        # for data in val_inputs:
        #     print(f"Input Data Stats: min={data.data.min()}, max={data.data.max()}")
        #
        # print(
        #     f"Labels Stats: min={self.train_labels.min()}, max={self.train_labels.max()}, unique={self.train_labels.unique()}")
        # print(
        #     f"Labels Stats: min={self.val_labels.min()}, max={self.val_labels.max()}, unique={self.val_labels.unique()}")

        self.train_dataset = TensorDataset(*train_inputs, self.train_labels)
        self.val_dataset = TensorDataset(*val_inputs, self.val_labels)

        # self.train_dataset = TensorDataset(*train_inputs, torch.tensor(self.train_labels, dtype=torch.float16))
        # self.val_dataset = TensorDataset(*val_inputs, torch.tensor(self.val_labels, dtype=torch.float16))

        self.data_loaded = True

    def forward(self, **inputs):
        feature_embeddings = []

        # print(f"Model inputs keys: {inputs.keys()}")

        for feature_name, feature_data in inputs.items():
            if feature_name.endswith("_attention_mask"):
                continue
            if feature_name.endswith("_input_ids"):
                input_ids = feature_data.to(self.device)

                attention_mask = inputs.get(f"{feature_name.replace('_input_ids', '_attention_mask')}", None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(self.device)
                    
                   

                # embedding = self.embedding_layer(input_ids=input_ids, attention_mask=attention_mask)
                # embedding = self.model.model.embed_tokens(input_ids=input_ids, attention_mask=attention_mask)
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
                hidden_states = outputs.hidden_states[-1]        # Shape: [batch_size, seq_len, hidden_dim]
                
                
                if self.pooling_mode == "cls":
                # LLaMA는 공식적으로 [CLS] 토큰이 없지만,
                # 예시로 첫 토큰(:,0,:)이나 끝 토큰(:,-1,:)을 쓰는 경우가 있음
                    embedding = hidden_states[:, 0, :]  # [CLS]라고 가정할 수도...
                # -----------------------
                # 1) Mean Pooling
                # -----------------------
                if self.pooling_mode == "mean":
                    if attention_mask is None:
                        # 단순 Mean Pooling
                        embedding = hidden_states.mean(dim=1)
                    else:
                        mask_f = attention_mask.float()  # (B, L)
                        masked_hidden = hidden_states * mask_f.unsqueeze(-1)  # (B, L, D)
                        sum_hidden = masked_hidden.sum(dim=1)                 # (B, D)
                        denom = mask_f.sum(dim=1, keepdim=True) + 1e-9
                        embedding = sum_hidden / denom                        # (B, D)

                # -----------------------
                # 2) Max Pooling
                # -----------------------
                if self.pooling_mode == "max":
                    if attention_mask is None:
                        embedding, _ = hidden_states.max(dim=1)
                    else:
                        inf_mask = (1 - attention_mask) * (-1e9)
                        inf_mask = inf_mask.unsqueeze(-1)     # (B, L, 1)
                        masked_hidden = hidden_states + inf_mask
                        embedding, _ = masked_hidden.max(dim=1)

                # ----------------------------------------------------------------
                # 3) Linear Attention Pooling [기존 코드와 유사]
                # ----------------------------------------------------------------
                if self.pooling_mode == "attention":
                    if attention_mask is not None:
                        # (B, L, 1)
                        scores = self.attn_score_proj(hidden_states)          # -> (B, L, 1)
                        scores = scores.squeeze(-1)                           # (B, L)
                        scores = scores.masked_fill(attention_mask == 0, float('-inf'))
                        attn_weights = F.softmax(scores, dim=1)               # (B, L)
                        
                        # (B, L, 1)
                        attn_weights = attn_weights.unsqueeze(-1)
                    
                        weighted_hidden = hidden_states * attn_weights        # (B, L, D)
                        embedding = weighted_hidden.sum(dim=1)                # (B, D)
                    
                    else:
                        scores = self.attn_score_proj(hidden_states).squeeze(-1)  # (B, L)
                        attn_weights = F.softmax(scores, dim=1)                    # (B, L)
                        attn_weights = attn_weights.unsqueeze(-1)                  # (B, L, 1)
                        weighted_hidden = hidden_states * attn_weights             # (B, L, D)
                        embedding = weighted_hidden.sum(dim=1)                     # (B, D)


                # ---------------------------------------
                # MLP-based attention pooling
                # ---------------------------------------
                if self.pooling_mode == "learnable":
                    # MLP로 score 계산
                    scores = self.learnable_score_mlp(hidden_states).squeeze(-1)  # (B, L)
                    if attention_mask is not None:
                        scores = scores.masked_fill(attention_mask == 0, float('-inf'))

                    attn_weights = F.softmax(scores, dim=1)  # (B, L)
                    attn_weights = attn_weights.unsqueeze(-1)
                    weighted_hidden = hidden_states * attn_weights
                    embedding = weighted_hidden.sum(dim=1)


                if self.pooling_mode == "mlp_multi_layers":
                    
                    all_layers = outputs.hidden_states
                    # 1) bidirectional attention이면, [B, seq_len, d] 평균 등을 사용할 수도 있음
                    #    causal이면 EOS 토큰 hidden_states[b, -1, :]
                    #    여기서는 일단 'bidirectional + mean' 가정
                    # => shape: (L, B, d)
                    stack_of_means = []
                    for layer_i in range(len(all_layers)):
                        # mean pooling => (B, d)
                        h_i_mean = all_layers[layer_i].mean(dim=1)
                        stack_of_means.append(h_i_mean)
                    # => (L, B, d)
                    stack_of_means = torch.stack(stack_of_means, dim=0)
                    
                    x_layers = stack_of_means.permute(1,0,2)
                    embedding = self.cross_attn_block(x_layers)

                # ---------------------------------------
                # (4) Downstream Projection Layer
                #     (Mean Pooling -> MLP)
                # ---------------------------------------
                
                
                if self.pooling_mode == "proj":
                    # 1) 혹시 마스크가 있다면, 토큰별 마스킹
                    if attention_mask is not None:
                        mask_f = attention_mask.float()          # (B, L)
                        mask_3d = mask_f.unsqueeze(-1)           # (B, L, 1)
                        hidden_states = hidden_states * mask_3d  # 0인 token 은 강제로 0
                    
                    # shape = (B, 128, 4096) 가정
                    # 1차 청크: 128개를 8개씩 묶어 평균 => (B, 16, 4096)
                    #   => 128=16*8
                    print(hidden_states.shape)
                    B, L, D = hidden_states.shape
                    if L != 128:
                        raise ValueError("예시 가정: L=128 이어야 함.")

                    # (B, 16, 8, 4096)
                    hidden_states = hidden_states.view(B, 16, 8, D)
                    # (B,16,4096)
                    hidden_states = hidden_states.mean(dim=2)

                    # 2차 청크: (16→4) => (B,4,4096)
                    hidden_states = hidden_states.view(B, 4, 4, D).mean(dim=2)

                    # 3차 청크: (4→1) => (B,1,4096)
                    hidden_states = hidden_states.view(B, 1, 4, D).mean(dim=2)

                    # shape=(B,1,4096) => (B,4096)
                    embedding = hidden_states.squeeze(1)

                    # 여기서 optional로 4096->256 proj
                    embedding = self.proj(embedding)  # shape=(B,256)
                # if self.pooling_mode == "proj":
                #     # 먼저 Mean Pooling
                #     if attention_mask is not None:
                #         mask_f = attention_mask.float()
                #         masked_hidden = hidden_states * mask_f.unsqueeze(-1)
                #         sum_hidden = masked_hidden.sum(dim=1)
                #         denom = mask_f.sum(dim=1, keepdim=True) + 1e-9
                #         pooled = sum_hidden / denom
                #     else:
                #         pooled = hidden_states.mean(dim=1)
                #     # (hidden_dim->1024->256)
                #     embedding = self.proj(pooled)  # (B, 256)

                # ---------------------------------------
                # (5) Pooling 이후 MLP (Max Pool -> 4096->1024->256->64->1)
                # ---------------------------------------
                if self.pooling_mode == "mlp_after_pooling":
                    # Max Pool 먼저
                    if attention_mask is None:
                        pooled, _ = hidden_states.max(dim=1)  # (B, D=4096)
                    else:
                        inf_mask = (1 - attention_mask) * (-1e9)
                        inf_mask = inf_mask.unsqueeze(-1)  # (B, L, 1)
                        masked_hidden = hidden_states + inf_mask
                        pooled, _ = masked_hidden.max(dim=1)
                    # MLP
                    logits = self.pooling_mlp(pooled)  # (B, 1)

                # ---------------------------------------
                # (6) Self-Attention Pooling (Learnable Query)
                # ---------------------------------------
                if self.pooling_mode == "query_attn":
                    B = hidden_states.size(0)
                    # query: (1, 1, D) -> (B, 1, D)
                    query_batch = self.query.expand(B, -1, -1)

                    Q = self.W_q(query_batch)      # (B,1,D)
                    K = self.W_k(hidden_states)    # (B,L,D)
                    V = self.W_v(hidden_states)    # (B,L,D)

                    # (B,1,D) x (B,D,L) = (B,1,L)
                    attn_scores = torch.bmm(Q, K.transpose(1,2)) / (self.hidden_dim ** 0.5)
                    if attention_mask is not None:
                        attn_scores = attn_scores.masked_fill(attention_mask.unsqueeze(1)==0, float('-inf'))

                    attn_weights = F.softmax(attn_scores, dim=-1)  # (B,1,L)
                    # (B,1,D) = (B,1,L) x (B,L,D)
                    pooled = torch.bmm(attn_weights, V)
                    embedding = pooled.squeeze(1)  # (B, D)

                
                
                # hidden_states = outputs.hidden_states[-1]        # Shape: [batch_size, seq_len, hidden_dim]
                # embedding = hidden_states.mean(dim=1)            # Shape: [batch_size, hidden_dim]

                # embedding = outputs.hidden_states[-1][:, 0, :]  # 첫 토큰만 임베딩시 대표성 상실-> mean pooling으로 진행.
                # hidden_states = outputs.hidden_states[-1]

                projection_key = feature_name.replace('_input_ids', '')

                if projection_key not in self.feature_projections:
                    raise KeyError(f"Projection for {projection_key} not found in feature_projections")

                # Ensure embedding matches projection layer's dtype and device
                projection_layer = self.feature_projections[projection_key]
                projection_layer = projection_layer.to(embedding.device)
                embedding = embedding.to(projection_layer.weight.dtype)

                projection = projection_layer(embedding)

            else:
                # Handle numeric features
                embedding = feature_data.float().unsqueeze(-1).to(self.device)
                projection_key = feature_name
                projection_layer = self.feature_projections[projection_key]

                # Ensure numeric embedding matches projection layer's dtype and device
                projection_layer = projection_layer.to(embedding.device)
                embedding = embedding.to(projection_layer.weight.dtype)

                projection = projection_layer(embedding)

            if projection.dim() == 2:  # Shape is [batch_size, 128]
                projection = projection.unsqueeze(1)
            feature_embeddings.append(projection)

            # Stack the embeddings along a new dimension for multi-head attention
        concatenated_features = torch.cat(feature_embeddings,
                                          dim=1)  # Shape: [batch_size, sequence_length, embedding_dim]

        # Apply multihead attention and layer normalization
        # attention_output = self.attention_layer(concatenated_features).to(self.device)  # attention map 구현 이전 코드
        attention_output, attn_weights = self.attention_layer(
           concatenated_features, 
           need_weights=True  # 추가
        )
        
        # Flatten the output and pass it through dense layers
        x = attention_output.view(attention_output.size(0), -1)
        x = torch.relu(self.fc1(x))
        
        # Dropout 추가
        #x = self.dropout(x)
        
        # x = torch.relu(self.fc2(x))
        #x = torch.relu(self.fc3(x))
        
        logits = self.output(x)

        return logits, attn_weights