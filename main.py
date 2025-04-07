########### main.py ##############
import os
import csv
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader, TensorDataset
import torch.optim as optim
import torch.nn.functional as F
from torch.amp import GradScaler
from torch.amp import autocast
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt

import params as PARAMS
from model.llama.llama_classifier import LlamaClassifier
from util import read_data

#############################################################################
def compute_local_cm_counts(preds: torch.Tensor, labels: torch.Tensor, num_classes: int):
    """
    Builds a local confusion-matrix count of shape (num_classes, num_classes).
    preds, labels: shape=(batch,) with integer class labels in [0..num_classes-1].
    """
    cm = torch.zeros((num_classes, num_classes), device=preds.device)
    for i in range(num_classes):
        for j in range(num_classes):
            cm[i, j] = ((labels == i) & (preds == j)).sum()
    return cm

def compute_precision_recall_f1(tp, fp, fn):
    """
    Compute precision, recall, f1 for a single class (binary style).
    """
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1        = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1

def compute_per_class_metrics(confusion_matrix):
    """
    NxN confusion_matrix -> per-class precision, recall, f1, plus overall_acc (total diagonal / total).
    We'll compute per-class accuracy separately if needed.
    """
    num_classes = confusion_matrix.size(0)
    eps = 1e-9

    total_correct = torch.diagonal(confusion_matrix).sum().item()
    total_items   = confusion_matrix.sum().item()
    overall_acc   = total_correct / (total_items + eps)

    precision_list = []
    recall_list    = []
    f1_list        = []

    for i in range(num_classes):
        tp_i = confusion_matrix[i, i].item()
        fp_i = (confusion_matrix[:, i].sum() - tp_i).item()
        fn_i = (confusion_matrix[i, :].sum() - tp_i).item()
        prec_i, rec_i, f1_i = compute_precision_recall_f1(tp_i, fp_i, fn_i)
        precision_list.append(prec_i)
        recall_list.append(rec_i)
        f1_list.append(f1_i)

    return precision_list, recall_list, f1_list, overall_acc

#############################################################################
def compute_extra_metrics(confusion_matrix, precision_list, recall_list, f1_list):
    """
    Return:
      per_class_accuracy[i],
      macro_precision, macro_recall, macro_f1, macro_accuracy
    For class i, accuracy[i] = cm[i,i]/(sum of row i).
    """
    num_classes = confusion_matrix.size(0)
    acc_list = []
    for i in range(num_classes):
        row_sum = confusion_matrix[i, :].sum().item()
        if row_sum > 0:
            acc_i = confusion_matrix[i, i].item() / row_sum
        else:
            acc_i = 0.0
        acc_list.append(acc_i)

    macro_prec = sum(precision_list) / num_classes
    macro_rec  = sum(recall_list)    / num_classes
    macro_f1   = sum(f1_list)        / num_classes
    macro_acc  = sum(acc_list)       / num_classes
    return acc_list, macro_prec, macro_rec, macro_f1, macro_acc

#############################################################################
def visualize_attention_map(model, val_loader, device, filename="results/attention_plot.jpg"):
    """
    1) One batch from val_loader
    2) Forward -> get attn_weights
    3) Plot NxN portion in "Blues" (with its colorbar),
       and the sums row/column in "Oranges" (with a separate colorbar),
       so that the sums appear in a different color scale.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    import csv
    import os

    model.eval()
    batch = next(iter(val_loader))
    *input_data, labels = batch
    input_data = [t.to(device) for t in input_data]

    model_inputs = {}
    idx = 0
    for feature in PARAMS.FEATURES:
        if feature == "Patient_ID":
            continue
        fkey = feature.replace(" ", "_")
        if PARAMS.FULL_FEATURES[feature] == 'str':
            model_inputs[f"{fkey}_input_ids"]      = input_data[idx]
            model_inputs[f"{fkey}_attention_mask"] = input_data[idx+1]
            idx += 2
        elif PARAMS.FULL_FEATURES[feature] == 'int16':
            model_inputs[fkey] = input_data[idx].to(device, dtype=torch.float16)
            idx += 1

    with torch.no_grad():
        with autocast(device_type="cuda"):
            logits, attn_weights = model(**model_inputs)

    # shape: [batch_size=1, num_heads=1, seq_len, seq_len].
    attn_map = attn_weights[0].cpu().numpy()  # shape (N, N) if single-head

    feature_labels = [f for f in PARAMS.FEATURES if f != "Patient_ID"]

    # === Build augmented matrix with row/col sums
    N = attn_map.shape[0]
    row_sums = attn_map.sum(axis=1)
    col_sums = attn_map.sum(axis=0)
    total_sum = attn_map.sum()

    aug_map = np.zeros((N+1, N+1), dtype=attn_map.dtype)
    aug_map[:N, :N] = attn_map
    aug_map[:N, -1] = row_sums
    aug_map[-1, :N] = col_sums
    aug_map[-1, -1] = total_sum

    ext_labels = feature_labels + ["Sum"]

    # === Write full (N+1)x(N+1) to CSV
    os.makedirs("results", exist_ok=True)
    csv_filename = os.path.join("results", "attention_score.csv")
    with open(csv_filename, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow([""] + ext_labels)
        for i, row_label in enumerate(ext_labels):
            row_data = aug_map[i]
            row_str = [f"{val:.4f}" for val in row_data]
            writer.writerow([row_label] + row_str)

    # === We'll plot NxN in "Blues" on one Axes, sums row/col in "Oranges" on another Axes,
    #     and combine them into a single figure. 
    #     This approach is simpler than overlaying partial heatmaps with mask.
    fig = plt.figure(figsize=(12, 9))

    # (A) Axes for NxN portion (top-left N x N)
    ax_main = fig.add_axes([0.1, 0.1, 0.65, 0.65])  # left, bottom, width, height
    # We'll keep extra row/col out of view for now by slicing
    nxn_data = aug_map[:N, :N]
    sns.heatmap(
        nxn_data,
        ax=ax_main,
        cmap="Blues",
        annot=True,
        fmt=".2f",
        xticklabels=feature_labels,
        yticklabels=feature_labels,
        cbar=True
    )
    ax_main.set_title("NxN Attention")
    ax_main.tick_params(axis='x', rotation=45)

    # (B) Axes for sums row (the bottom row) – we’ll plot as a 1xN heatmap
    # Let's place it directly below the NxN portion, same width
    ax_sum_row = fig.add_axes([0.1, 0.77, 0.65, 0.08])  # test these coords
    # note: we want the row to appear horizontally, so we can shape it as (1,N).
    sum_row_data = aug_map[-1:, :N]  # shape (1, N)
    sns.heatmap(
        sum_row_data,
        ax=ax_sum_row,
        cmap="Oranges",
        annot=True,
        fmt=".2f",
        xticklabels=feature_labels,
        yticklabels=["Sum"],
        cbar=True
    )
    
    ax_sum_row.set_title("Row Sums")
    ax_sum_row.set_xticklabels([])
    ax_sum_row.set_yticklabels([])
    ax_sum_row.set_xlabel('')
    ax_sum_row.set_ylabel('')
    # Hide spines, etc., or carefully adjust if they overlap

    # (C) Axes for sums column (the rightmost column)
    # We'll do (N x 1) data. 
    ax_sum_col = fig.add_axes([0.76, 0.1, 0.08, 0.65])
    sum_col_data = aug_map[:N, -1:]  # shape (N, 1)
    sns.heatmap(
        sum_col_data,
        ax=ax_sum_col,
        cmap="Oranges",
        annot=True,
        fmt=".2f",
        xticklabels=["Sum"],
        yticklabels=feature_labels,
        cbar=True
    )
    ax_sum_col.set_title("Column Sums")
    ax_sum_col.set_xticklabels([])
    ax_sum_col.set_yticklabels([])
    ax_sum_col.set_xlabel('')
    ax_sum_col.set_ylabel('')

    # (D) Axes for the bottom-right single cell (the total sum)
    ax_total = fig.add_axes([0.76, 0.77, 0.08, 0.08])
    total_data = aug_map[-1:, -1:]  # shape (1,1)
    sns.heatmap(
        total_data,
        ax=ax_total,
        cmap="Oranges",
        annot=True,
        fmt=".2f",
        xticklabels=["Sum"],
        yticklabels=["Sum"],
        cbar=True
    )
    ax_total.set_title("Total")

    fig.suptitle("Attention Map with Sums (Separate Colorbars)", fontsize=14)
    plt.savefig(filename, bbox_inches='tight')
    plt.close(fig)

    print(f"[INFO] Attention map (with sums) saved to {filename}")
    print(f"[INFO] CSV (with sums) saved to {csv_filename}")

#############################################################################
def setup_ddp(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="file:///tmp/ddp_init",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def reduce_tensor(tensor, world_size):
    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor / world_size

def train(rank, world_size, df):
    setup_ddp(rank, world_size)
    if rank == 0:
        os.makedirs("results", exist_ok=True)
    dist.barrier()

    print(f"Rank {rank} starting training...")

    try:
        device = torch.device(f"cuda:{rank}")
        classifier = LlamaClassifier(
            df=df,
            device=device,
            pooling_mode=PARAMS.POOLING_MODE,
            world_size=world_size,
            rank=rank
        )
        model = classifier.to(device)
        model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

        optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=PARAMS.LEARNING_RATE
        )
        loss_fn = nn.CrossEntropyLoss()

        train_inputs, train_labels, val_inputs, val_labels = classifier.make_dataset()
        train_dataset = TensorDataset(*train_inputs, train_labels)
        val_dataset   = TensorDataset(*val_inputs, val_labels)

        train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
        val_sampler   = DistributedSampler(val_dataset,   num_replicas=world_size, rank=rank, shuffle=False)

        train_loader  = DataLoader(train_dataset, batch_size=PARAMS.BATCH_SIZE, sampler=train_sampler)
        val_loader    = DataLoader(val_dataset,   batch_size=PARAMS.BATCH_SIZE, sampler=val_sampler)

        # Only rank=0 does real-time plotting
        if rank == 0:
            plt.ion()
            fig, ax = plt.subplots(2, 1, figsize=(8, 8))
            ax[0].set_title("Loss")
            ax[0].set_xlabel("Epoch")
            ax[0].set_ylabel("Loss")
            ax[0].grid(True)
            train_loss_line, = ax[0].plot([], [], marker="o", label="Train Loss", color='blue')
            val_loss_line,   = ax[0].plot([], [], marker="o", label="Val Loss",   color='red')
            ax[0].legend()

            ax[1].set_title("Accuracy (Overall)")
            ax[1].set_xlabel("Epoch")
            ax[1].set_ylabel("Accuracy")
            ax[1].grid(True)
            train_acc_line,  = ax[1].plot([], [], marker="o", label="Train Acc", color='blue')
            val_acc_line,    = ax[1].plot([], [], marker="o", label="Val Acc",   color='red')
            ax[1].legend()

            x_data = []
            train_loss_list, val_loss_list = [], []
            train_acc_list,  val_acc_list  = [], []

            # For multi-class: track metrics per class
            class_labels = PARAMS.CLASSES
            num_classes  = len(class_labels)

            # We'll also store per-class accuracy each epoch => so it can appear in evaluation_matrix_XXX.jpg
            val_precision_list = [[] for _ in range(num_classes)]
            val_recall_list    = [[] for _ in range(num_classes)]
            val_f1_list        = [[] for _ in range(num_classes)]
            val_acc_each_class = [[] for _ in range(num_classes)]  # <-- (ADDED) track per-class accuracy
        else:
            train_loss_line = val_loss_line = train_acc_line = val_acc_line = None
            x_data = None
            train_loss_list = None
            val_loss_list = None
            train_acc_list = None
            val_acc_list = None
            class_labels = PARAMS.CLASSES
            num_classes  = len(class_labels)
            val_precision_list = None
            val_recall_list    = None
            val_f1_list        = None
            val_acc_each_class = None

        scaler = GradScaler()

        for epoch in range(PARAMS.EPOCHS):
            dist.barrier()
            model.train()
            train_loss_sum = 0.0
            correct_train, total_train = 0, 0

            for batch in tqdm(train_loader, desc=f"Rank {rank} Epoch {epoch} (Train)"):
                *input_data, labels = batch
                input_data = [t.to(device) for t in input_data]
                labels     = labels.to(device)

                model_inputs = {}
                idx = 0
                for feature in PARAMS.FEATURES:
                    if feature == "Patient_ID":
                        continue
                    fkey = feature.replace(" ", "_")
                    if PARAMS.FULL_FEATURES[feature] == 'str':
                        model_inputs[f"{fkey}_input_ids"]      = input_data[idx]
                        model_inputs[f"{fkey}_attention_mask"] = input_data[idx+1]
                        idx += 2
                    elif PARAMS.FULL_FEATURES[feature] == 'int16':
                        model_inputs[fkey] = input_data[idx].to(device, dtype=torch.float16)
                        idx += 1

                optimizer.zero_grad()
                with autocast(device_type="cuda"):
                    logits, attn_weights = model(**model_inputs)
                    loss = loss_fn(logits.float(), labels.float())

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                train_loss_sum += loss.item()
                preds  = torch.argmax(logits, dim=-1)
                target = torch.argmax(labels, dim=-1)
                correct_train += (preds == target).sum().item()
                total_train   += target.size(0)

            train_loss_tensor = torch.tensor(train_loss_sum, device=device)
            correct_tensor    = torch.tensor(correct_train,  device=device)
            total_tensor      = torch.tensor(total_train,    device=device)

            dist.barrier()
            global_train_loss = reduce_tensor(train_loss_tensor, world_size).item()
            global_correct    = reduce_tensor(correct_tensor,    world_size).item()
            global_total      = reduce_tensor(total_tensor,      world_size).item()

            global_train_loss /= (len(train_loader)+1e-9)
            global_train_acc   = global_correct / (global_total+1e-9)

            # Validation
            model.eval()
            val_loss_sum = 0.0
            correct_val, total_val = 0, 0
            local_val_preds = []
            local_val_labels= []

            with torch.no_grad():
                for batch in tqdm(val_loader, desc=f"Rank {rank} Epoch {epoch} (Val)"):
                    *input_data, labels = batch
                    input_data = [t.to(device) for t in input_data]
                    labels     = labels.to(device)

                    model_inputs = {}
                    idx = 0
                    for feature in PARAMS.FEATURES:
                        if feature == "Patient_ID":
                            continue
                        fkey = feature.replace(" ", "_")
                        if PARAMS.FULL_FEATURES[feature] == 'str':
                            model_inputs[f"{fkey}_input_ids"]      = input_data[idx]
                            model_inputs[f"{fkey}_attention_mask"] = input_data[idx+1]
                            idx += 2
                        elif PARAMS.FULL_FEATURES[feature] == 'int16':
                            model_inputs[fkey] = input_data[idx].to(device, dtype=torch.float16)
                            idx += 1

                    with autocast(device_type="cuda"):
                        logits, attn_weights = model(**model_inputs)
                        loss = loss_fn(logits.float(), labels.float())

                    val_loss_sum += loss.item()
                    preds  = torch.argmax(logits, dim=-1)
                    target = torch.argmax(labels, dim=-1)
                    correct_val += (preds == target).sum().item()
                    total_val   += target.size(0)

                    local_val_preds.append(preds.clone())
                    local_val_labels.append(target.clone())

            val_loss_tensor = torch.tensor(val_loss_sum, device=device)
            correct_val_t   = torch.tensor(correct_val,  device=device)
            total_val_t     = torch.tensor(total_val,    device=device)

            dist.barrier()
            global_val_loss = reduce_tensor(val_loss_tensor, world_size).item()
            global_val_corr = reduce_tensor(correct_val_t,   world_size).item()
            global_val_totl = reduce_tensor(total_val_t,     world_size).item()

            global_val_loss /= (len(val_loader)+1e-9)
            global_val_acc   = global_val_corr / (global_val_totl+1e-9)

            # confusion matrix
            local_val_preds  = torch.cat(local_val_preds,  dim=0)
            local_val_labels = torch.cat(local_val_labels, dim=0)
            local_cm = compute_local_cm_counts(local_val_preds, local_val_labels, num_classes)
            dist.barrier()
            flat_local_cm = local_cm.view(-1)
            dist.all_reduce(flat_local_cm, op=dist.ReduceOp.SUM)
            global_cm = flat_local_cm.view(num_classes, num_classes)

            if rank == 0:
                precision_each, recall_each, f1_each, overall_val_acc = compute_per_class_metrics(global_cm)
                # We'll also get per-class accuracy to track each epoch
                acc_each, _, _, _, _ = compute_extra_metrics(global_cm, precision_each, recall_each, f1_each)

                epoch_index = epoch + 1
                print(f"\nEpoch {epoch_index}/{PARAMS.EPOCHS}")
                print(f"Train Loss: {global_train_loss:.4f}, Train Acc: {global_train_acc:.4f}")
                print(f"Val   Loss: {global_val_loss:.4f},   Val Acc: {global_val_acc:.4f}")

                x_data.append(epoch_index)
                train_loss_list.append(global_train_loss)
                val_loss_list.append(global_val_loss)
                train_acc_list.append(global_train_acc)
                val_acc_list.append(global_val_acc)

                # store per-class metrics for plotting
                for i in range(num_classes):
                    val_precision_list[i].append(precision_each[i])
                    val_recall_list[i].append(recall_each[i])
                    val_f1_list[i].append(f1_each[i])
                    val_acc_each_class[i].append(acc_each[i])  # new

                # update real-time plot lines
                train_loss_line.set_data(x_data, train_loss_list)
                val_loss_line.set_data(x_data,   val_loss_list)
                train_acc_line.set_data(x_data,  train_acc_list)
                val_acc_line.set_data(x_data,    val_acc_list)

                plt.tight_layout()
                for axis in ax:
                    axis.relim()
                    axis.autoscale_view()
                plt.pause(0.1)

        #================= After all epochs, final outputs ====================
        if rank == 0:
            plt.savefig("results/train_val_plot.png")
            plt.ioff()
            plt.show()

            # Confusion matrix figure
            final_cm = global_cm.cpu().numpy()
            fig_cm, ax_cm = plt.subplots(figsize=(12,8))
            sns.heatmap(final_cm, annot=True, fmt='.0f', cmap='Blues', ax=ax_cm,
                        xticklabels=class_labels, yticklabels=class_labels)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("True")
            ax_cm.set_title("Confusion Matrix")
            plt.tight_layout()
            plt.savefig("results/confusion_matrix.jpg")
            plt.close(fig_cm)

            # Evaluate final metrics
            precision_each, recall_each, f1_each, overall_val_acc = compute_per_class_metrics(
                torch.from_numpy(final_cm)
            )
            acc_each, macro_prec, macro_rec, macro_f1, macro_acc = compute_extra_metrics(
                torch.from_numpy(final_cm),
                precision_each, recall_each, f1_each
            )

            # Separate per-class metric plots (f1, recall, precision, accuracy)
            epochs_range = range(1, PARAMS.EPOCHS + 1)
            for i, cl_name in enumerate(class_labels):
                fig_cls, ax_cls = plt.subplots(figsize=(9,6))
                # Plot lines
                ax_cls.plot(epochs_range, val_f1_list[i],          label="F1",        marker='o')
                ax_cls.plot(epochs_range, val_recall_list[i],      label="Recall",    marker='x')
                ax_cls.plot(epochs_range, val_precision_list[i],   label="Precision", marker='^')
                ax_cls.plot(epochs_range, val_acc_each_class[i],   label="Accuracy",  marker='D')  # (ADDED)

                ax_cls.set_title(f"Validation Metrics for {cl_name}")
                ax_cls.set_xlabel("Epoch")
                ax_cls.set_ylabel("Score")
                ax_cls.legend()
                ax_cls.grid(True)
                plt.tight_layout()
                plt.savefig(f"results/evaluation_matrix_{cl_name.lower()}.jpg")
                plt.close(fig_cls)

            # Write final CSV
            csv_path = "results/evaluation.csv"
            with open(csv_path, "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["class","f1","recall","precision","accuracy"])
                for i, cl_name in enumerate(class_labels):
                    writer.writerow([
                        cl_name,
                        f"{f1_each[i]:.4f}",
                        f"{recall_each[i]:.4f}",
                        f"{precision_each[i]:.4f}",
                        f"{acc_each[i]:.4f}"
                    ])
                # overall => macro avg
                writer.writerow([
                    "Overall",
                    f"{macro_f1:.4f}",   # macro-F1
                    f"{macro_rec:.4f}",  # macro-recall
                    f"{macro_prec:.4f}", # macro-precision
                    f"{macro_acc:.4f}"   # macro-accuracy
                ])
            print(f"Final per-class metrics saved to {csv_path}")

            # Finally, attention map
            visualize_attention_map(model, val_loader, device,
                filename="results/attention_plot.jpg")

    finally:
        print(f"Rank {rank} cleaning up...")
        dist.destroy_process_group()

#############################################################################
if __name__ == "__main__":
    torch.cuda.empty_cache()
    try:
        world_size = torch.cuda.device_count()
        train_df, val_df = read_data()
        df = (train_df, val_df)
        torch.multiprocessing.spawn(
            train,
            args=(world_size, df),
            nprocs=world_size,
            join=True
        )
    except KeyboardInterrupt:
        print("Keyboard interrupt detected. Cleaning up...")
    except Exception as e:
        print(e)
    finally:
        os.system("pkill -9 python")
