from ..base_model import BaseModel
import params as PARAMS
import numpy as np
from transformers import LlamaTokenizer, LlamaForSequenceClassification, LlamaConfig, LlamaForCausalLM, LlamaModel, LlamaPreTrainedModel
# from transformers import AutoTokenizer, AutoModelForCausalLM
from tensorflow.python.keras.utils.np_utils import to_categorical
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm  # For progress bar (optional)
import numpy as np
import torchmetrics
import os
import sentencepiece as spm

# Check if CUDA is available
print("CUDA available:", torch.cuda.is_available())
# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Print the name of the CUDA device
if torch.cuda.is_available():
    print("CUDA device name:", torch.cuda.get_device_name(0))
    print("CUDA device count:", torch.cuda.device_count())
    print("Current CUDA device:", torch.cuda.current_device())
else:
    print("No CUDA devices found.")

def compute_class_biases(labels):
    # Assuming labels are one-hot encoded, calculate the class distribution
    class_totals = np.sum(labels, axis=0)
    class_probs = class_totals / np.sum(class_totals)

    # Calculate logit bias: log(p / (1 - p)) for each class
    initial_bias = np.log(class_probs / (1 - class_probs))
    return initial_bias

tokenizer_path="model/llama/Llama3.1-8B"

llama_model_path = "model/llama/Llama3.1-8B"

config_path = "model/llama/Llama3.1-8B/config.json"

config = LlamaConfig.from_json_file(config_path)

class LlamaEmbeddingLayer(nn.Module):
    def __init__(self, llama_model, **kwargs):
        super(LlamaEmbeddingLayer, self).__init__(**kwargs)
        self.llama_model = llama_model

    def forward(self, input_ids, attention_mask=None):
        outputs = self.llama_model(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token embedding

class MultiHeadAttentionLayer(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(MultiHeadAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        return self.layer_norm(attn_output + x)

def assign(left, right):
    if left.shape != right.shape:
        raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")

    if isinstance(right, torch.Tensor):
        return torch.nn.Parameter(right.clone().detach())
    else:
        return torch.nn.Parameter(torch.tensor(right))


def load_weights_into_llama(model, param_config, params):
    model.tok_emb.weight = assign(model.tok_emb.weight, params["tok_embeddings.weight"])

    for l in range(param_config["n_layers"]):

        # Load attention weights
        model.trf_blocks[l].att.W_query.weight = assign(
            model.trf_blocks[l].att.W_query.weight,
            params[f"layers.{l}.attention.wq.weight"]
        )
        model.trf_blocks[l].att.W_key.weight = assign(
            model.trf_blocks[l].att.W_key.weight,
            params[f"layers.{l}.attention.wk.weight"]
        )
        model.trf_blocks[l].att.W_value.weight = assign(
            model.trf_blocks[l].att.W_value.weight,
            params[f"layers.{l}.attention.wv.weight"]
        )
        model.trf_blocks[l].att.out_proj.weight = assign(
            model.trf_blocks[l].att.out_proj.weight,
            params[f"layers.{l}.attention.wo.weight"]
        )
        model.trf_blocks[l].norm1.weight = assign(
            model.trf_blocks[l].norm1.weight,
            params[f"layers.{l}.attention_norm.weight"]
        )

        # Load FeedForward weights
        model.trf_blocks[l].ff.fc1.weight = assign(
            model.trf_blocks[l].ff.fc1.weight,
            params[f"layers.{l}.feed_forward.w1.weight"]
        )
        # For some reason w2 and w3 are provided in the wrong order in the weights file
        model.trf_blocks[l].ff.fc2.weight = assign(
            model.trf_blocks[l].ff.fc2.weight,
            params[f"layers.{l}.feed_forward.w3.weight"]
        )
        model.trf_blocks[l].ff.fc3.weight = assign(
            model.trf_blocks[l].ff.fc3.weight,
            params[f"layers.{l}.feed_forward.w2.weight"]
        )
        model.trf_blocks[l].norm2.weight = assign(
            model.trf_blocks[l].norm2.weight,
            params[f"layers.{l}.ffn_norm.weight"]
        )

    # Load output layer weights
    model.final_norm.weight = assign(model.final_norm.weight, params["norm.weight"])
    model.out_head.weight = assign(model.out_head.weight, params["output.weight"])

PPP = {
    "dim": 4096,
    "ffn_dim_multiplier": 1.3,
    "multiple_of": 1024,
    "n_heads": 32,
    "n_kv_heads": 8,
    "n_layers": 32,
    "norm_eps": 1e-05, "rope_theta": 500000.0, "use_scaled_rope": True, "vocab_size": 128256}

class LlamaClassifier(nn.Module):

    def __init__(self, df, *args, **kwargs):
        # super(LlamaClassifier, self).__init__(df)

        super().__init__(*args, **kwargs)
        (self.train_df, self.val_df) = df

        self.train_labels = to_categorical(self.train_df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)
        self.val_labels = to_categorical(self.val_df['label_encoded'], num_classes=PARAMS.NUM_CLASSES)

        self.train_labels = torch.tensor(self.train_labels, dtype=torch.float32).unsqueeze(1)
        self.val_labels = torch.tensor(self.val_labels, dtype=torch.float32).unsqueeze(1)


        self.tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
        # self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load LLaMA model for embeddings
        self.model = LlamaModel(config=config)
        # self.model = AutoModelForCausalLM.from_pretrained(
        #     llama_model_path
        #     )
        # self.model.load_state_dict(torch.load(llama_model_path, weights_only=True))

        # weights = torch.load(llama_model_path, mmap=True, map_location="cuda")
        # load_weights_into_llama(self.model, PPP, weights)
        # self.model.load_state_dict(checkpoint)
        self.embedding_layer = LlamaEmbeddingLayer(self.model)

        # Define projection layers for string and integer features
        self.feature_projections = nn.ModuleDict()
        for feature, feature_type in PARAMS.FULL_FEATURES.items():
            if feature_type == 'str' and feature != "Patient_ID":
                self.feature_projections[feature] = nn.Linear(config.hidden_size, 128)
            elif feature_type == 'int32':
                self.feature_projections[feature] = nn.Linear(1, 128)

        # Attention layer
        self.attention_layer = MultiHeadAttentionLayer(embed_dim=128, num_heads=4)

        # Fully connected layers for classification
        self.fc1 = nn.Linear(128 * len(PARAMS.FEATURES), 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 2)

    def tokenize_feature(self, texts, max_length=PARAMS.MAX_LEN):
        encoding = self.tokenizer(
            list(texts),
            max_length=max_length,
            padding="max_length",  # Ensures fixed length of max_length
            truncation=True,
            return_tensors="pt"
        )
        # Return tensors directly
        # return encoding["input_ids"], encoding["attention_mask"]
        return {'input_ids': encoding["input_ids"], 'attention_mask': encoding["attention_mask"]}

    def make_dataset(self):
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

            elif PARAMS.FULL_FEATURES[feature] == 'int32':
                # Add integer features to train and val lists
                self.train_inputs.append(torch.tensor(self.train_df[feature], dtype=torch.float32).unsqueeze(1))
                self.val_inputs.append(torch.tensor(self.val_df[feature], dtype=torch.float32).unsqueeze(1))

        self.data_loaded = True

    def forward(self, inputs):
        feature_embeddings = []

        for feature_name, feature_data in inputs.items():
            if feature_name.endswith("_input_ids"):
                input_ids = feature_data["input_ids"]
                attention_mask = feature_data["attention_mask"]
                embedding = self.embedding_layer(input_ids, attention_mask)
                projection = self.feature_projections[feature_name[:-10]](embedding)  # Project string feature
            else:
                embedding = feature_data.float().unsqueeze(-1)
                projection = self.feature_projections[feature_name](embedding)  # Project integer feature

            feature_embeddings.append(projection)

        concatenated_features = torch.stack(feature_embeddings, dim=1)

        # Apply multihead attention and layer normalization
        attention_output = self.attention_layer(concatenated_features)

        # Flatten and pass through dense layers
        x = attention_output.view(attention_output.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))

        return self.output(x)

    def train(self):
        # Determine total steps for the cosine decay restarts
        steps_per_epoch = len(self.train_df) // PARAMS.BATCH_SIZE
        total_steps = PARAMS.EPOCHS_PER_CYCLE * steps_per_epoch

        # Initialize the optimizer and scheduler
        optimizer = optim.SGD(self.model.parameters(), lr=PARAMS.LEARNING_RATE)
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps, T_mult=1, eta_min=0)

        # Loss function and evaluation metrics
        loss_fn = nn.CrossEntropyLoss()
        accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=2).to(device)
        precision_metric_class0 = torchmetrics.Precision(task="multiclass", num_classes=2, average="none")[0]
        precision_metric_class1 = torchmetrics.Precision(task="multiclass", num_classes=2, average="none")[1]
        recall_metric_class0 = torchmetrics.Recall(task="multiclass", num_classes=2, average="none")[0]
        recall_metric_class1 = torchmetrics.Recall(task="multiclass", num_classes=2, average="none")[1]

        # Load datasets
        self.make_dataset()
        train_loader = torch.utils.data.DataLoader(self.train_dataset, batch_size=PARAMS.BATCH_SIZE, shuffle=True)
        val_loader = torch.utils.data.DataLoader(self.val_dataset, batch_size=PARAMS.BATCH_SIZE, shuffle=False)

        for epoch in range(PARAMS.EPOCHS):
            # Training loop
            self.model.train()
            train_loss = 0
            for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{PARAMS.EPOCHS}"):
                input_data, labels = batch
                input_data, labels = input_data.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = self.model(input_data)
                loss = loss_fn(outputs, labels)
                loss.backward()
                optimizer.step()

                # Update metrics
                train_loss += loss.item()
                accuracy_metric(outputs, labels)
                precision_metric_class0(outputs, labels)
                precision_metric_class1(outputs, labels)
                recall_metric_class0(outputs, labels)
                recall_metric_class1(outputs, labels)

            # Step the learning rate scheduler
            scheduler.step(epoch + total_steps)

            # Calculate average metrics for the epoch
            avg_train_loss = train_loss / len(train_loader)
            accuracy = accuracy_metric.compute().item()
            precision_class0 = precision_metric_class0.compute().item()
            precision_class1 = precision_metric_class1.compute().item()
            recall_class0 = recall_metric_class0.compute().item()
            recall_class1 = recall_metric_class1.compute().item()

            print(f"Epoch {epoch + 1}/{PARAMS.EPOCHS}, Loss: {avg_train_loss:.4f}, Accuracy: {accuracy:.4f}")
            print(f"Precision (Class 0): {precision_class0:.4f}, Precision (Class 1): {precision_class1:.4f}")
            print(f"Recall (Class 0): {recall_class0:.4f}, Recall (Class 1): {recall_class1:.4f}")

            # Reset metrics after each epoch
            accuracy_metric.reset()
            precision_metric_class0.reset()
            precision_metric_class1.reset()
            recall_metric_class0.reset()
            recall_metric_class1.reset()

            # Validation loop
            self.model.eval()
            with torch.no_grad():
                val_loss = 0
                for batch in val_loader:
                    input_data, labels = batch
                    input_data, labels = input_data.to(device), labels.to(device)

                    outputs = self.model(input_data)
                    loss = loss_fn(outputs, labels)
                    val_loss += loss.item()

                avg_val_loss = val_loss / len(val_loader)
                print(f"Validation Loss: {avg_val_loss:.4f}")

            # Optionally add any callbacks or logging functions here

    # def test(self):
    #     self.model = load_model('model.keras', custom_objects={'LlamaEmbeddingLayer': LlamaEmbeddingLayer})
    #
    #     super().test()
