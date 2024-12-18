# from model.bert.bert import Bert
from model.llama.llama_classifier import LlamaClassifier
from util import *
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
from torch.cuda.amp import GradScaler, autocast
import torch.nn as nn
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
import torch.optim as optim
# import tensorflow as tf
# from tensorflow import keras

# print("TensorFlow version:", tf.__version__)
# print("Keras version:", keras.__version__)
#
# gpus = tf.config.experimental.list_physical_devices('GPU')
# if gpus:
#     try:
#         gpu_number = 0  # Change this to the desired GPU number (e.g., 1, 2, etc.)
#         tf.config.experimental.set_visible_devices(gpus[gpu_number], 'GPU')
#
#         # Optional: Set memory growth to avoid memory allocation issues
#         tf.config.experimental.set_memory_growth(gpus[gpu_number], True)
#
#         print(f"Using GPU: {gpu_number}")
#     except RuntimeError as e:
#         # Visible devices must be set at program startup
#         print(e)


# def main():
#     train_df, val_df = read_data()
#     df = (train_df, val_df)
#     # model = Bert(df)
#     model = LlamaClassifier(df)
#     model.train()
#     # model.test()

def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="file:///tmp/ddp_init",  # Use a file-based initialization
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def reduce_tensor(tensor, world_size):
    """
    Reduces a tensor from all ranks by summing it and divides by world size.
    Ensures global metric computation in DDP.
    """
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size

def compute_metrics(preds, labels):
    """
    Compute TP, FP, FN, and total for precision, recall, and accuracy.
    """
    tp = ((preds == 1) & (labels == 1)).sum().item()
    fp = ((preds == 1) & (labels == 0)).sum().item()
    fn = ((preds == 0) & (labels == 1)).sum().item()
    correct = (preds == labels).sum().item()
    total = labels.size(0)

    return tp, fp, fn, correct, total

def train(rank, world_size, df):
    print(f"Rank {rank} starting training...")
    setup_ddp(rank, world_size)
    try:
        device = torch.device(f"cuda:{rank}")
        print(f"Rank {rank} using device: {device}")
        classifier = LlamaClassifier(df=df, device=device, world_size=world_size, rank=rank)
        model = classifier.to(device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=PARAMS.LEARNING_RATE)
        # lora_params = [p for n, p in self.model.named_parameters() if "lora" in n]
        # other_params = [p for n, p in self.model.named_parameters() if "lora" not in n]
        # print(lora_params)
        # scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=total_steps, T_mult=1, eta_min=0)

        # Loss function and evaluation metrics
        loss_fn = nn.CrossEntropyLoss()
        accuracy_metric = torchmetrics.Accuracy(task="binary").to(device)
        precision_metric_class0 = torchmetrics.Precision(task="multiclass", num_classes=2, average="none").to(device)[0]
        precision_metric_class1 = torchmetrics.Precision(task="multiclass", num_classes=2, average="none").to(device)[1]
        recall_metric_class0 = torchmetrics.Recall(task="multiclass", num_classes=2, average="none").to(device)[0]
        recall_metric_class1 = torchmetrics.Recall(task="multiclass", num_classes=2, average="none").to(device)[1]

        # Load datasets
        train_inputs, train_labels, val_inputs, val_labels = classifier.make_dataset()
        train_dataset = TensorDataset(*train_inputs, train_labels)

        # train_loader = DataLoader(self.train_dataset, batch_size=PARAMS.BATCH_SIZE, shuffle=True)
        # val_loader = DataLoader(self.val_dataset, batch_size=PARAMS.BATCH_SIZE, shuffle=False)
        # model.gradient_checkpointing_disable()

        # Data setup
        sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        train_loader = DataLoader(train_dataset, batch_size=PARAMS.BATCH_SIZE, sampler=sampler)

        # Initialize lists to store loss and accuracy values
        epoch_losses = []
        epoch_accuracies = []

        # Initialize real-time plotting
        if rank == 0:
            plt.ion()
            fig, ax = plt.subplots(2, 1, figsize=(4, 5))

            # Configure loss plot
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            ax[0].set_title("Training Loss")
            loss_line, = ax[0].plot([], [], marker="o", label="Training Loss")
            ax[0].legend()
            ax[0].grid()

            # Configure accuracy plot
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")
            ax[1].set_title("Training Accuracy")
            acc_line, = ax[1].plot([], [], marker="o", color="orange", label="Training Accuracy")
            ax[1].legend()
            ax[1].grid()

            x_data = []  # Epochs
            y_loss_data = []  # Losses
            y_acc_data = []  # Accuracies
        torch.cuda.empty_cache()
        scaler = torch.amp.GradScaler(device="cuda")
        for epoch in range(PARAMS.EPOCHS):
            # Training loop
            dist.barrier()
            model.train()
            train_loss = 0
            # Reset metrics after each epoch
            # Local metric accumulators
            tp_sum, fp_sum, fn_sum = 0, 0, 0
            correct_sum, total_sum = 0, 0
            accuracy_metric.reset()
            precision_metric_class0.reset()
            precision_metric_class1.reset()
            recall_metric_class0.reset()
            recall_metric_class1.reset()

            for batch in tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch}"):
                *input_data, labels = batch
                input_data = [feature.to(device) for feature in input_data]  # Move each input to device
                labels = labels.to(device)

                # Initialize dictionary for model inputs
                model_inputs = {}

                # Populate model_inputs with text and integer features
                idx = 0
                for feature in PARAMS.FEATURES:
                    if feature == "Patient_ID":
                        continue

                    feature_key = feature.replace(" ", "_")

                    if PARAMS.FULL_FEATURES[feature] == 'str':
                        # Assign input IDs and attention masks for text features
                        model_inputs[f"{feature_key}_input_ids"] = input_data[idx].to(device)
                        model_inputs[f"{feature_key}_attention_mask"] = input_data[idx + 1].to(device)
                        idx += 2
                    elif PARAMS.FULL_FEATURES[feature] == 'int16':
                        # Assign integer feature tensors directly
                        model_inputs[feature_key] = input_data[idx].to(device, dtype=torch.float16)
                        idx += 1

                optimizer.zero_grad()

                # with torch.cuda.amp.autocast():


                outputs = model(**model_inputs)
                loss = loss_fn(outputs.float(), labels.float())

                if torch.isnan(loss).any():
                    print("NaN detected in loss. Stopping training.")
                    break

                # loss.backward()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # for name, param in self.model.named_parameters():
                #     assert param.device == device, f"Parameter {name} is not on {device}."
                #     if param.requires_grad:
                #         # print(f"{name}: Grad = {param.grad}")
                #         if param.grad is None:
                #             pass
                #             # print(f"[WARNING] Gradient is None for {name}. This parameter is not being updated.")
                #         else:
                #             print(
                #                 f"{name}: Parameter Mean = {param.data.mean():.6f}, Gradient Mean = {param.grad.mean():.6f}")
                #
                # for fc_layer_name in ['fc1', 'fc3', 'output']:
                #     fc_layer = getattr(self, fc_layer_name)
                #     print(f"\nLayer: {fc_layer_name}")
                #     for name, param in fc_layer.named_parameters():
                #         print(f"    Param: {fc_layer_name}.{name} - Mean: {param.data.mean().item()}")
                #         if param.grad is not None:
                #             print(f"    Grad: {fc_layer_name}.{name} - Grad Mean: {param.grad.mean().item()}")
                #         else:
                #             print(f"    Grad: {fc_layer_name}.{name} - None (no gradient)")

                # Update metrics
                train_loss += loss.item()
                preds = torch.argmax(outputs, dim=-1)  # For multi-class classification
                labels = torch.argmax(labels, dim=-1)

                # Compute local metrics
                tp, fp, fn, correct, total = compute_metrics(preds, labels)
                tp_sum += tp
                fp_sum += fp
                fn_sum += fn
                correct_sum += correct
                total_sum += total

                with torch.no_grad():
                    accuracy_metric(preds, labels)
                    precision_metric_class0(preds, labels)
                    precision_metric_class1(preds, labels)
                    recall_metric_class0(preds, labels)
                    recall_metric_class1(preds, labels)

            # Reduce and synchronize metrics across all ranks
            train_loss_tensor = torch.tensor(train_loss, device=device)
            tp_tensor = torch.tensor(tp_sum, device=device)
            fp_tensor = torch.tensor(fp_sum, device=device)
            fn_tensor = torch.tensor(fn_sum, device=device)
            correct_tensor = torch.tensor(correct_sum, device=device)
            total_tensor = torch.tensor(total_sum, device=device)

            dist.barrier()
            train_loss = reduce_tensor(train_loss_tensor, world_size).item()
            tp_global = reduce_tensor(tp_tensor, world_size).item()
            fp_global = reduce_tensor(fp_tensor, world_size).item()
            fn_global = reduce_tensor(fn_tensor, world_size).item()
            correct_global = reduce_tensor(correct_tensor, world_size).item()
            total_global = reduce_tensor(total_tensor, world_size).item()

            # Compute global metrics
            global_accuracy = correct_global / total_global
            global_precision = tp_global / (tp_global + fp_global + 1e-8)  # Avoid division by zero
            global_recall = tp_global / (tp_global + fn_global + 1e-8)
            global_f1 = 2 * (global_precision * global_recall) / (global_precision + global_recall + 1e-8)

            if rank == 0:
                print(f"Epoch {epoch + 1}/{PARAMS.EPOCHS}")
                print(f"Loss: {train_loss:.4f}, Accuracy: {global_accuracy:.4f}")
                print(f"Precision: {global_precision:.4f}, Recall: {global_recall:.4f}, F1 Score: {global_f1:.4f}")

                # Update real-time plots
                x_data.append(epoch + 1)
                y_loss_data.append(train_loss)
                y_acc_data.append(global_accuracy)

                loss_line.set_data(x_data, y_loss_data)
                acc_line.set_data(x_data, y_acc_data)

                ax[0].set_xlim(0, max(x_data) + 1)
                ax[0].set_ylim(min(y_loss_data) - 0.1, max(y_loss_data) + 0.1)
                ax[1].set_xlim(0, max(x_data) + 1)
                ax[1].set_ylim(0, max(y_acc_data) + 0.1)

                plt.draw()
            plt.pause(0.1)
                # dist.barrier()  # Ensure ranks sync before moving forward


        if rank == 0:
            plt.ioff()
            plt.show()

    finally:
        print(f"Rank {rank} cleaning up...")
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    # Main DDP Entry Point
    world_size = torch.cuda.device_count()
    train_df, val_df = read_data()
    df = (train_df, val_df)
    try:
        torch.multiprocessing.spawn(train, args=(world_size, df), nprocs=world_size, join=True)
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
    except Exception as e:
        print(e)
    finally:
        # Optional: Clean up any distributed processes manually
        os.system("pkill -9 python")