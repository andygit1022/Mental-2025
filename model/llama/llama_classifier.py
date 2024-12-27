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
    def __init__(self, llama_model, device, **kwargs):
        super(LlamaEmbeddingLayer, self).__init__(**kwargs)
        self.llama_model = llama_model
        self.device = device

    # def forward(self, input_ids, attention_mask=None):
    #     input_ids = input_ids.to(dtype=torch.long, device=self.device)
    #     attention_mask = attention_mask.to(self.device) if attention_mask is not None else None

    #     outputs = self.llama_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #     # Use the last hidden state from the `hidden_states` tuple
    #     return outputs.hidden_states[-1][:, 0, :]  # Extract the [CLS] token embedding
    
    # avg pooling
    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(dtype=torch.long, device=self.device)
        attention_mask = attention_mask.to(self.device) if attention_mask is not None else None

        outputs = self.llama_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_dim]

        # Mean Pooling over all tokens
        embedding = hidden_states.mean(dim=1)  # Shape: [batch_size, hidden_dim]

        return embedding

    # def forward(self, input_ids, attention_mask=None):
    #     input_ids = input_ids.to(dtype=torch.long, device=self.device)
    #     attention_mask = attention_mask.to(self.device) if attention_mask is not None else None

    #     outputs = self.llama_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    #     hidden_states = outputs.hidden_states[-1]  # Shape: [batch_size, seq_len, hidden_dim]

    #     # Attention-based weighted pooling
    #     attn_scores = torch.softmax(hidden_states.mean(dim=-1), dim=1)  # Shape: [batch_size, seq_len]
    #     embedding = torch.sum(hidden_states * attn_scores.unsqueeze(-1), dim=1)  # Weighted sum

    #     return embedding



class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x.to(self.attention.in_proj_weight.device)
        attn_output, _ = self.attention(x, x, x)
        return self.layer_norm(attn_output + x)

class LlamaClassifier(nn.Module):

    def get_model(self):
        return self.model    
    def __init__(self, df, device, world_size, rank, *args, **kwargs):
        super().__init__(*args, **kwargs)
        (self.train_df, self.val_df) = df

        self.device = device
        self.world_size = world_size
        self.rank = rank

        self.train_labels = torch.nn.functional.one_hot(
            torch.tensor(self.train_df['label_encoded'].values, dtype=torch.long),
            num_classes=2  # Adjust as needed
        ).float()

        self.val_labels = torch.nn.functional.one_hot(
            torch.tensor(self.val_df['label_encoded'].values, dtype=torch.long),
            num_classes=2  # Adjust as needed
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
            r=2,  # Low-rank dimension
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
        self.embedding_layer = LlamaEmbeddingLayer(llama_model=self.model, device=device)

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
        self.output = nn.Linear(PARAMS.MAX_LEN, 2).to(device)
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
                embedding = hidden_states.mean(dim=1)            # Shape: [batch_size, hidden_dim]

                #embedding = outputs.hidden_states[-1][:, 0, :]  # 첫 토큰만 임베딩시 대표성 상실-> mean pooling으로 진행.

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
        attention_output = self.attention_layer(concatenated_features).to(self.device)

        # Flatten the output and pass it through dense layers
        x = attention_output.view(attention_output.size(0), -1)
        x = torch.relu(self.fc1(x))
        # x = torch.relu(self.fc2(x))
        #x = torch.relu(self.fc3(x))

        return self.output(x)

    # def train(self, rank, world_size, df):
    #     device = torch.device(f"cuda:{rank}")
    #
    #     model = LlamaClassifier(df).to(device)
    #     model = DDP(model, device_ids=[rank], output_device=rank)
    #
    #     # Determine total steps for the cosine decay restarts
    #     steps_per_epoch = len(self.train_df) // PARAMS.BATCH_SIZE
    #     total_steps = PARAMS.EPOCHS_PER_CYCLE * steps_per_epoch
    #
    #     # Initialize the optimizer and scheduler
    #     # optimizer = optim.SGD(self.model.parameters(), lr=PARAMS.LEARNING_RATE)
    #     optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.model.parameters()), lr=PARAMS.LEARNING_RATE)
    #     # lora_params = [p for n, p in self.model.named_parameters() if "lora" in n]
    #     # other_params = [p for n, p in self.model.named_parameters() if "lora" not in n]
    #     # print(lora_params)
    #     # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps, T_mult=1, eta_min=0)
    #
    #     # Loss function and evaluation metrics
    #     loss_fn = nn.CrossEntropyLoss()
    #     accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
    #     precision_metric_class0 = torchmetrics.Precision(task="multiclass", num_classes=2, average="none").to(device)[0]
    #     precision_metric_class1 = torchmetrics.Precision(task="multiclass", num_classes=2, average="none").to(device)[1]
    #     recall_metric_class0 = torchmetrics.Recall(task="multiclass", num_classes=2, average="none").to(device)[0]
    #     recall_metric_class1 = torchmetrics.Recall(task="multiclass", num_classes=2, average="none").to(device)[1]
    #
    #     # Load datasets
    #     self.make_dataset()
    #     # train_loader = DataLoader(self.train_dataset, batch_size=PARAMS.BATCH_SIZE, shuffle=True)
    #     # val_loader = DataLoader(self.val_dataset, batch_size=PARAMS.BATCH_SIZE, shuffle=False)
    #     self.model.gradient_checkpointing_disable()
    #
    #     # Data setup
    #     sampler = DistributedSampler(self.train_dataset, num_replicas=world_size, rank=rank)
    #     train_loader = DataLoader(self.train_dataset, batch_size=PARAMS.BATCH_SIZE, sampler=sampler)
    #
    #     # Initialize lists to store loss and accuracy values
    #     epoch_losses = []
    #     epoch_accuracies = []
    #
    #     # Initialize real-time plotting
    #     plt.ion()
    #     fig, ax = plt.subplots(2, 1, figsize=(4, 5))
    #
    #     # Configure loss plot
    #     ax[0].set_xlabel("Epoch")
    #     ax[0].set_ylabel("Loss")
    #     ax[0].set_title("Training Loss")
    #     loss_line, = ax[0].plot([], [], marker="o", label="Training Loss")
    #     ax[0].legend()
    #     ax[0].grid()
    #
    #     # Configure accuracy plot
    #     ax[1].set_xlabel("Epoch")
    #     ax[1].set_ylabel("Accuracy")
    #     ax[1].set_title("Training Accuracy")
    #     acc_line, = ax[1].plot([], [], marker="o", color="orange", label="Training Accuracy")
    #     ax[1].legend()
    #     ax[1].grid()
    #
    #     x_data = []  # Epochs
    #     y_loss_data = []  # Losses
    #     y_acc_data = []  # Accuracies
    #     torch.cuda.empty_cache()
    #     scaler = torch.cuda.amp.GradScaler()
    #     for epoch in range(PARAMS.EPOCHS):
    #         # Training loop
    #         self.model.train()
    #         train_loss = 0
    #         # Reset metrics after each epoch
    #         accuracy_metric.reset()
    #         precision_metric_class0.reset()
    #         precision_metric_class1.reset()
    #         recall_metric_class0.reset()
    #         recall_metric_class1.reset()
    #
    #         for batch in tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch}"):
    #             *input_data, labels = batch
    #             input_data = [feature.to(device) for feature in input_data]  # Move each input to device
    #             labels = labels.to(device)
    #
    #             # Initialize dictionary for model inputs
    #             model_inputs = {}
    #
    #             # Populate model_inputs with text and integer features
    #             idx = 0
    #             for feature in PARAMS.FEATURES:
    #                 if feature == "Patient_ID":
    #                     continue
    #
    #                 feature_key = feature.replace(" ", "_")
    #
    #                 if PARAMS.FULL_FEATURES[feature] == 'str':
    #                     # Assign input IDs and attention masks for text features
    #                     model_inputs[f"{feature_key}_input_ids"] = input_data[idx].to(device)
    #                     model_inputs[f"{feature_key}_attention_mask"] = input_data[idx + 1].to(device)
    #                     idx += 2
    #                 elif PARAMS.FULL_FEATURES[feature] == 'int16':
    #                     # Assign integer feature tensors directly
    #                     model_inputs[feature_key] = input_data[idx].to(device, dtype=torch.float16)
    #                     idx += 1
    #
    #             optimizer.zero_grad()
    #
    #             # with torch.cuda.amp.autocast():
    #             outputs = self.forward(**model_inputs)
    #             loss = loss_fn(outputs.float(), labels.float())
    #
    #             if torch.isnan(loss).any():
    #                 print("NaN detected in loss. Stopping training.")
    #                 break
    #
    #             # loss.backward()
    #             scaler.scale(loss).backward()
    #             scaler.step(optimizer)
    #             scaler.update()
    #
    #             # for name, param in self.model.named_parameters():
    #             #     assert param.device == device, f"Parameter {name} is not on {device}."
    #             #     if param.requires_grad:
    #             #         # print(f"{name}: Grad = {param.grad}")
    #             #         if param.grad is None:
    #             #             pass
    #             #             # print(f"[WARNING] Gradient is None for {name}. This parameter is not being updated.")
    #             #         else:
    #             #             print(
    #             #                 f"{name}: Parameter Mean = {param.data.mean():.6f}, Gradient Mean = {param.grad.mean():.6f}")
    #             #
    #             # for fc_layer_name in ['fc1', 'fc3', 'output']:
    #             #     fc_layer = getattr(self, fc_layer_name)
    #             #     print(f"\nLayer: {fc_layer_name}")
    #             #     for name, param in fc_layer.named_parameters():
    #             #         print(f"    Param: {fc_layer_name}.{name} - Mean: {param.data.mean().item()}")
    #             #         if param.grad is not None:
    #             #             print(f"    Grad: {fc_layer_name}.{name} - Grad Mean: {param.grad.mean().item()}")
    #             #         else:
    #             #             print(f"    Grad: {fc_layer_name}.{name} - None (no gradient)")
    #
    #             # Update metrics
    #             train_loss += loss.item()
    #             preds = torch.argmax(outputs, dim=-1)  # For multi-class classification
    #             labels = torch.argmax(labels, dim=-1)
    #
    #             with torch.no_grad():
    #                 accuracy_metric(preds, labels)
    #                 precision_metric_class0(preds, labels)
    #                 precision_metric_class1(preds, labels)
    #                 recall_metric_class0(preds, labels)
    #                 recall_metric_class1(preds, labels)
    #
    #         print(f"Rank {rank}, Epoch {epoch}, Loss: {train_loss / len(train_loader)}")
    #
    #         # Calculate average metrics for the epoch
    #         avg_train_loss = train_loss / len(train_loader)
    #         accuracy = accuracy_metric.compute().item()
    #         precision_class0 = precision_metric_class0.compute().item()
    #         precision_class1 = precision_metric_class1.compute().item()
    #         recall_class0 = recall_metric_class0.compute().item()
    #         recall_class1 = recall_metric_class1.compute().item()
    #
    #         print(f"Epoch {epoch + 1}/{PARAMS.EPOCHS}, Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.4f}")
    #         print(f"Precision (Class 0): {precision_class0:.4f}, Precision (Class 1): {precision_class1:.4f}")
    #         print(f"Recall (Class 0): {recall_class0:.4f}, Recall (Class 1): {recall_class1:.4f}")
    #
    #
    #
    #         epoch_losses.append(avg_train_loss)
    #
    #         # Calculate average metrics for the epoch
    #         avg_train_loss = train_loss / len(train_loader)
    #         train_accuracy = accuracy
    #         epoch_losses.append(avg_train_loss)
    #         epoch_accuracies.append(train_accuracy)
    #
    #         # Update real-time plots
    #         x_data.append(epoch + 1)
    #         y_loss_data.append(avg_train_loss)
    #         y_acc_data.append(train_accuracy)
    #
    #         loss_line.set_data(x_data, y_loss_data)
    #         acc_line.set_data(x_data, y_acc_data)
    #
    #         ax[0].set_xlim(0, max(x_data) + 1)
    #         ax[0].set_ylim(min(y_loss_data)-0.1, max(y_loss_data) + 0.1)
    #         ax[1].set_xlim(0, max(x_data) + 1)
    #         ax[1].set_ylim(0, max(y_acc_data) + 0.1)
    #
    #         plt.draw()
    #         plt.pause(0.1)
    #
    #     dist.destroy_process_group()
    #     # Turn off interactive mode and finalize the plot
    #     plt.ioff()
    #     plt.show()
    #
    #         # # Validation loop
    #         # self.model.eval()
    #         # with torch.no_grad():
    #         #     val_loss = 0
    #         #     for batch in val_loader:
    #         #         *input_data, labels = batch
    #         #         input_data = [feature.to(device) for feature in input_data]  # Move each input to device
    #         #         labels = labels.to(device)
    #         #
    #         #         # Initialize dictionary for model inputs
    #         #         model_inputs = {}
    #         #
    #         #         # Populate model_inputs with text and integer features
    #         #         idx = 0
    #         #         for feature in PARAMS.FEATURES:
    #         #             if feature == "Patient_ID":
    #         #                 continue
    #         #
    #         #             feature_key = feature.replace(" ", "_")
    #         #
    #         #             if PARAMS.FULL_FEATURES[feature] == 'str':
    #         #                 # Assign input IDs and attention masks for text features
    #         #                 model_inputs[f"{feature_key}_input_ids"] = input_data[idx].to(device)
    #         #                 model_inputs[f"{feature_key}_attention_mask"] = input_data[idx + 1].to(device)
    #         #                 idx += 2
    #         #             elif PARAMS.FULL_FEATURES[feature] == 'int16':
    #         #                 # Assign integer feature tensors directly
    #         #                 model_inputs[feature_key] = input_data[idx].to(device, dtype=torch.float16)
    #         #                 idx += 1
    #         #
    #         #         outputs = self.forward(**model_inputs)
    #         #         loss = loss_fn(outputs, labels)
    #         #         val_loss += loss.item()
    #         #
    #         #     avg_val_loss = val_loss / len(val_loader)
    #         #     print(f"Validation Loss: {avg_val_loss:.4f}")
    #
    #         # Optionally add any callbacks or logging functions here

    # def test(self):
    #     self.model = load_model('model.keras', custom_objects={'LlamaEmbeddingLayer': LlamaEmbeddingLayer})
    #
    #     super().test()