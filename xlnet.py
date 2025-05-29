import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import (
    XLNetTokenizer,
    XLNetForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    # get_scheduler, # Not explicitly used with Trainer managing schedules
)
from datasets import Dataset
# import evaluate # Not used, compute_metrics uses sklearn
from tqdm.auto import tqdm  # Keep for potential use in data loading or elsewhere
# Keep for potential use, though Trainer handles its own
from torch.utils.data import DataLoader
from transformers.utils import logging
from captum.attr import LayerIntegratedGradients
import random  # Ensure random is imported
import copy
import os
import torch.nn.functional as F


# Set up logging
logging.set_verbosity_info()

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Configuration
MODEL_NAME = "xlnet-base-cased"
MAX_LENGTH = 512
BATCH_SIZE = 8  # Per device train batch size
EVAL_BATCH_SIZE = BATCH_SIZE * 2  # Per device eval batch size
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 2e-5
NUM_EPOCHS_PER_FOLD = 15  # Renamed for clarity
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
CROSS_VAL_FOLDS = 5  # Standardized to 5 folds

# Focal Loss implementation for handling class imbalance


class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss_val = self.alpha * (1 - pt)**self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss_val.mean()
        elif self.reduction == 'sum':
            return focal_loss_val.sum()
        else:  # 'none'
            return focal_loss_val

# Custom Trainer for Focal Loss


class CustomTrainer(Trainer):
    def __init__(self, *args, focal_loss_alpha=1, focal_loss_gamma=2, **kwargs):
        super().__init__(*args, **kwargs)
        # Pass alpha and gamma if you want to configure them per trainer instance
        self.focal_loss = FocalLoss(
            alpha=focal_loss_alpha, gamma=focal_loss_gamma)

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # Added **kwargs
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        loss = self.focal_loss(logits, labels)
        return (loss, outputs) if return_outputs else loss


def load_data(file_path):
    encodings = ['cp1252', 'latin-1', 'utf-8']
    df = None
    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded data with {encoding} encoding.")
            break
        except UnicodeDecodeError:
            print(f"Failed to decode with {encoding}.")
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")
            continue
    if df is None:
        print("All standard encodings failed. Trying utf-8 with error replacement.")
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')

    for col in ['Paper Title', 'Abstract', 'Review Text', 'Review Type']:
        if col in df.columns:
            df[col] = df[col].fillna('')
            if col != 'Review Type':  # Only apply ASCII conversion to text fields
                df[col] = df[col].apply(lambda x: str(x).encode(
                    'ascii', 'ignore').decode('ascii'))
        else:
            print(
                f"Warning: Column {col} not found. It will be treated as empty.")
            df[col] = ''

    # Create weighted text combination with section importance weights
    df['text'] = (
        # Ensure string type
        "[TITLE] " + df['Paper Title'].astype(str) + " [/TITLE] " +
        # Ensure string type
        "[ABSTRACT] " + df['Abstract'].astype(str) + " [/ABSTRACT] " +
        # Ensure string type
        "[REVIEW] " + df['Review Text'].astype(str) + " [/REVIEW]"
    )

    df = df.dropna(subset=['text', 'Review Type'])
    df = df[df['Review Type'].apply(lambda x: isinstance(x, str) and len(
        x.strip()) > 0)]  # Ensure Review Type is valid string

    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Review Type'])

    return df, label_encoder.classes_


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        padding="max_length",  # XLNet often uses 'longest' or explicit padding to max_length
        max_length=MAX_LENGTH,
        truncation=True,
        # return_tensors="pt" # Trainer handles this if datasets are not formatted to torch yet
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average='weighted'
    )
    accuracy = (predictions == labels).mean()

    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall
    }


def plot_training_history(history_logs, filename='xlnet_training_history_cv_best_fold.png'):
    train_metrics = [
        log for log in history_logs if 'loss' in log and 'eval_loss' not in log]
    eval_metrics = [log for log in history_logs if 'eval_loss' in log]

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    if train_metrics:
        # Aggregate training loss per epoch if multiple steps are logged
        epoch_train_loss = {}
        for m in train_metrics:
            epoch = round(m['epoch'])
            if epoch not in epoch_train_loss:
                epoch_train_loss[epoch] = []
            epoch_train_loss[epoch].append(m['loss'])

        sorted_epochs = sorted(epoch_train_loss.keys())
        avg_train_loss = [np.mean(epoch_train_loss[e]) for e in sorted_epochs]
        if sorted_epochs and avg_train_loss:
            plt.plot(sorted_epochs, avg_train_loss,
                     label='Training Loss', marker='o', linestyle='--')

    if eval_metrics:
        eval_epochs = [m['epoch'] for m in eval_metrics]
        plt.plot(eval_epochs, [m['eval_loss']
                 for m in eval_metrics], label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training & Validation Loss (Best Fold)')
    plt.grid(True)

    plt.subplot(1, 3, 2)
    if eval_metrics:
        eval_epochs = [m['epoch'] for m in eval_metrics]
        plt.plot(eval_epochs, [m['eval_accuracy']
                 for m in eval_metrics], label='Validation Accuracy', marker='o')
        plt.plot(eval_epochs, [m['eval_f1'] for m in eval_metrics],
                 label='Validation F1 Score', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics (Best Fold)')
    plt.grid(True)

    plt.subplot(1, 3, 3)
    if eval_metrics:
        eval_epochs = [m['epoch'] for m in eval_metrics]
        plt.plot(eval_epochs, [m['eval_precision']
                 for m in eval_metrics], label='Validation Precision', marker='o')
        plt.plot(eval_epochs, [m['eval_recall']
                 for m in eval_metrics], label='Validation Recall', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Precision & Recall (Best Fold)')
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"Training history plot saved to {filename}")
    plt.close()


def plot_confusion_matrix(y_true, y_pred, class_names, filename='xlnet_confusion_matrix_cv_test.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix (Test Set)')
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Confusion matrix saved to {filename}")
    plt.close()


def explain_prediction(text, model, tokenizer, class_names, device):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Ensure model is in eval mode for explanations
    model.eval()

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class = class_names[predicted_class_idx]

    # Setup for attribution - use model.transformer.word_emb for XLNet
    # Check if the model has 'transformer' and 'word_emb'
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'word_emb'):
        embedding_layer = model.transformer.word_emb
    else:
        # Fallback or error if the expected layer structure is not found
        print("Warning: XLNet embedding layer 'model.transformer.word_emb' not found. Using 'model.transformer.word_embedding' as fallback or check model structure.")
        # This was the original, might be specific to a version.
        # For robustness, one might need to inspect model.config or model structure.
        # For now, let's assume one of them will work or default to the original if word_emb is missing.
        if hasattr(model, 'transformer') and hasattr(model.transformer, 'word_embedding'):
            embedding_layer = model.transformer.word_embedding
        else:
            print("Error: Could not find a suitable embedding layer for LIG attribution.")
            return {
                "prediction": predicted_class,
                "confidence": float(probabilities[0][predicted_class_idx]),
                "top_words": [("Error in attribution", 0.0)]
            }

    lig = LayerIntegratedGradients(model, embedding_layer)

    # Ensure inputs['input_ids'] is on the same device as the model for LIG
    input_ids_on_device = inputs['input_ids'].to(model.device)
    # Baselines on the same device
    baselines = torch.zeros_like(input_ids_on_device)

    attributions, delta = lig.attribute(
        inputs=input_ids_on_device,
        baselines=baselines,
        target=predicted_class_idx,
        return_convergence_delta=True,
        n_steps=50,  # Number of steps for approximation
        # internal_batch_size=BATCH_SIZE # Optional: if inputs are large
    )

    tokens = tokenizer.convert_ids_to_tokens(
        # Ensure on CPU for numpy conversion
        inputs['input_ids'][0].cpu().numpy())
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)  # Normalize
    attributions = attributions.cpu().detach().numpy()

    word_importances = []
    for token, importance in zip(tokens, attributions):
        # Filter out special tokens
        if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
            word_importances.append((token, float(importance)))  # Fixed loop

    # Sort by absolute importance
    word_importances.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "prediction": predicted_class,
        "confidence": float(probabilities[0][predicted_class_idx]),
        "top_words": word_importances[:20]  # Top 20 words
    }


def train_model_with_cv():
    print("Loading data...")
    df_full, class_names = load_data('corpus.csv')
    if df_full.empty:
        print("Error: No data loaded. Exiting.")
        return
    print(
        f"Loaded {len(df_full)} samples with {len(class_names)} classes: {class_names}")

    df_train_cv, df_test = train_test_split(
        df_full, test_size=0.15, random_state=SEED, stratify=df_full['label'])
    print(
        f"Train/CV set size: {len(df_train_cv)}, Test set size: {len(df_test)}")
    df_train_cv = df_train_cv.reset_index(drop=True)  # Reset index for iloc
    df_test = df_test.reset_index(drop=True)

    tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME)

    skf = StratifiedKFold(n_splits=CROSS_VAL_FOLDS,
                          shuffle=True, random_state=SEED)

    best_fold_f1 = 0.0
    best_model_state_dict = None
    best_fold_log_history = None

    fold_metrics_summary = []

    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train_cv, df_train_cv['label'])):
        print(f"\\n--- Fold {fold + 1}/{CROSS_VAL_FOLDS} ---")

        train_df_fold = df_train_cv.iloc[train_idx]
        val_df_fold = df_train_cv.iloc[val_idx]

        train_dataset_fold = Dataset.from_pandas(train_df_fold)
        val_dataset_fold = Dataset.from_pandas(val_df_fold)

        # Ensure 'text' column exists before tokenization
        if 'text' not in train_dataset_fold.column_names:
            print("Error: 'text' column missing in train_dataset_fold.")
            continue
        if 'text' not in val_dataset_fold.column_names:
            print("Error: 'text' column missing in val_dataset_fold.")
            continue

        tokenized_train_fold = train_dataset_fold.map(lambda ex: tokenize_function(ex, tokenizer), batched=True, remove_columns=[
                                                      'text'] + [col for col in train_df_fold.columns if col not in ['label', 'input_ids', 'attention_mask', 'token_type_ids'] and col in train_dataset_fold.column_names])
        tokenized_val_fold = val_dataset_fold.map(lambda ex: tokenize_function(ex, tokenizer), batched=True, remove_columns=[
                                                  'text'] + [col for col in val_df_fold.columns if col not in ['label', 'input_ids', 'attention_mask', 'token_type_ids'] and col in val_dataset_fold.column_names])

        # Set format for PyTorch
        tokenized_train_fold.set_format("torch")
        tokenized_val_fold.set_format("torch")

        model = XLNetForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(class_names))
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Trainer handles device placement
        # model.to(device)

        training_args = TrainingArguments(
            output_dir=f'./xlnet_results/fold_{fold+1}',
            num_train_epochs=NUM_EPOCHS_PER_FOLD,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            learning_rate=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            logging_dir=f'./xlnet_logs/fold_{fold+1}',
            logging_strategy="epoch",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            report_to="tensorboard",
            fp16=torch.cuda.is_available(),
            seed=SEED,
            # dataloader_num_workers=2, # Optional: for faster data loading
        )

        trainer = CustomTrainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_train_fold,
            eval_dataset=tokenized_val_fold,
            compute_metrics=compute_metrics,
            tokenizer=tokenizer,
            callbacks=[EarlyStoppingCallback(
                early_stopping_patience=3, early_stopping_threshold=0.001)]
        )

        print(f"Starting training for fold {fold + 1}...")
        trainer.train()

        print(f"Evaluating fold {fold + 1} on its validation set...")
        eval_results = trainer.evaluate()
        current_fold_f1 = eval_results['eval_f1']
        print(
            f"Fold {fold + 1} - Validation F1: {current_fold_f1:.4f}, Accuracy: {eval_results['eval_accuracy']:.4f}")
        fold_metrics_summary.append(eval_results)

        if current_fold_f1 > best_fold_f1:
            best_fold_f1 = current_fold_f1
            # Get state_dict from trainer.model which is the best one if load_best_model_at_end=True
            best_model_state_dict = copy.deepcopy(model.state_dict())
            best_fold_log_history = copy.deepcopy(trainer.state.log_history)
            print(
                f"New best model found in fold {fold + 1} with F1: {best_fold_f1:.4f}")

    if fold_metrics_summary:
        avg_metrics = {
            "avg_eval_loss": np.mean([res["eval_loss"] for res in fold_metrics_summary if "eval_loss" in res]),
            "avg_eval_accuracy": np.mean([res["eval_accuracy"] for res in fold_metrics_summary if "eval_accuracy" in res]),
            "avg_eval_f1": np.mean([res["eval_f1"] for res in fold_metrics_summary if "eval_f1" in res]),
            "avg_eval_precision": np.mean([res["eval_precision"] for res in fold_metrics_summary if "eval_precision" in res]),
            "avg_eval_recall": np.mean([res["eval_recall"] for res in fold_metrics_summary if "eval_recall" in res]),
        }
        print("\\n--- Cross-Validation Summary (Average over folds) ---")
        for key, value in avg_metrics.items():
            print(f"{key}: {value:.4f}")

    if best_model_state_dict:
        print("\\nLoading the overall best model from cross-validation...")
        overall_best_model = XLNetForSequenceClassification.from_pretrained(
            MODEL_NAME, num_labels=len(class_names))
        overall_best_model.load_state_dict(best_model_state_dict)
        # overall_best_model.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu')) # Trainer will handle device for predict

        model_save_path = "./xlnet_classifier_cv"
        print(f"Saving the best model to {model_save_path}...")
        overall_best_model.save_pretrained(model_save_path)
        tokenizer.save_pretrained(model_save_path)

        if best_fold_log_history:
            print("Plotting training history of the best fold...")
            plot_training_history(
                best_fold_log_history, filename='xlnet_training_history_cv_best_fold.png')

        print("\\nEvaluating the best model on the hold-out test set...")
        test_dataset_pd = Dataset.from_pandas(df_test)
        if 'text' not in test_dataset_pd.column_names:
            print("Error: 'text' column missing in test_dataset_pd.")
        else:
            tokenized_test_dataset = test_dataset_pd.map(lambda ex: tokenize_function(ex, tokenizer), batched=True, remove_columns=[
                                                         'text'] + [col for col in df_test.columns if col not in ['label', 'input_ids', 'attention_mask', 'token_type_ids'] and col in test_dataset_pd.column_names])
            tokenized_test_dataset.set_format("torch")

            test_trainer_args = TrainingArguments(
                output_dir='./xlnet_results/test_eval_final',
                per_device_eval_batch_size=EVAL_BATCH_SIZE,
                report_to="none",
                fp16=torch.cuda.is_available(),
            )
            test_trainer = Trainer(
                model=overall_best_model,
                args=test_trainer_args,
                compute_metrics=compute_metrics,
                tokenizer=tokenizer
            )

            test_predictions_output = test_trainer.predict(
                tokenized_test_dataset)
            test_predictions = np.argmax(
                test_predictions_output.predictions, axis=-1)
            test_true_labels = test_predictions_output.label_ids

            print("\\nTest Set Evaluation Report:")
            print(classification_report(test_true_labels,
                  test_predictions, target_names=class_names, digits=4))

            print("Plotting confusion matrix for the test set...")
            plot_confusion_matrix(test_true_labels, test_predictions,
                                  class_names, filename='xlnet_confusion_matrix_cv_test.png')

            if not df_test.empty:
                sample_idx = random.randint(0, len(df_test) - 1)
                sample_text = df_test.iloc[sample_idx]['text']
                sample_true_label_idx = df_test.iloc[sample_idx]['label']
                sample_true_label = class_names[sample_true_label_idx]
                print(
                    f"\\nExplaining prediction for a random sample from test set (True Label: {sample_true_label}):")
                try:
                    # Ensure model is on the correct device for explain_prediction
                    device_for_explain = torch.device(
                        'cuda' if torch.cuda.is_available() else 'cpu')
                    overall_best_model.to(device_for_explain)
                    explanation = explain_prediction(
                        sample_text, overall_best_model, tokenizer, class_names, device_for_explain)
                    print(
                        f"  Predicted: {explanation['prediction']} (Confidence: {explanation['confidence']:.4f})")
                    print(
                        f"  Top words contributing: {explanation['top_words']}")
                except Exception as e:
                    print(f"  Could not generate explanation: {e}")
                    import traceback
                    traceback.print_exc()
    else:
        print("No model was successfully trained and selected across folds.")


if __name__ == "__main__":
    # import matplotlib # Not needed if plt is used directly
    # matplotlib.use('Agg') # Uncomment if running in a headless environment

    # Create directories if they don't exist
    os.makedirs("./xlnet_results", exist_ok=True)
    os.makedirs("./xlnet_logs", exist_ok=True)
    os.makedirs("./xlnet_classifier_cv", exist_ok=True)

    print(
        f"Starting XLNet model training with {CROSS_VAL_FOLDS}-fold cross-validation and Focal Loss")
    print(
        f"Model: {MODEL_NAME}, Max Length: {MAX_LENGTH}, Batch Size (per device): {BATCH_SIZE}")
    print(
        f"Effective Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}, Eval Batch Size: {EVAL_BATCH_SIZE}")
    print(
        f"Learning Rate: {LEARNING_RATE}, Epochs per fold: {NUM_EPOCHS_PER_FOLD}")

    train_model_with_cv()
    print("\\nXLNet training and evaluation with cross-validation finished.")


"""
Starting XLNet model training with 5-fold cross-validation and Focal Loss
Model: xlnet-base-cased, Max Length: 512, Batch Size (per device): 8
Effective Batch Size: 32, Eval Batch Size: 16
Learning Rate: 2e-05, Epochs per fold: 15
Loading data...
Failed to decode with cp1252.
Successfully loaded data with latin-1 encoding.
Loaded 346 samples with 3 classes: ['AI-Generated' 'Authentic' 'Generic']
Train/CV set size: 294, Test set size: 52
loading file spiece.model from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/spiece.model
loading file added_tokens.json from cache at None
loading file special_tokens_map.json from cache at None
loading file tokenizer_config.json from cache at None
loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/tokenizer.json
loading file chat_template.jinja from cache at None
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/config.json
Model config XLNetConfig {
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": false,
  "bos_token_id": 1,
  "clamp_len": -1,
  "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "gelu",
  "initializer_range": 0.02,
  "layer_norm_eps": 1e-12,
  "mem_len": null,
  "model_type": "xlnet",
  "n_head": 12,
  "n_layer": 12,
  "pad_token_id": 5,
  "reuse_len": null,
  "same_length": false,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 250
    }
  },
  "transformers_version": "4.52.2",
  "untie_r": true,
  "use_mems_eval": true,
  "use_mems_train": false,
  "vocab_size": 32000
}

\n--- Fold 1/5 ---
Map: 100%
 235/235 [00:00<00:00, 323.25 examples/s]
Map: 100%
 59/59 [00:00<00:00, 312.27 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/config.json
Model config XLNetConfig {
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": false,
  "bos_token_id": 1,
  "clamp_len": -1,
  "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "gelu",
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "initializer_range": 0.02,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  },
  "layer_norm_eps": 1e-12,
  "mem_len": null,
  "model_type": "xlnet",
  "n_head": 12,
  "n_layer": 12,
  "pad_token_id": 5,
  "reuse_len": null,
  "same_length": false,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 250
    }
  },
  "transformers_version": "4.52.2",
  "untie_r": true,
  "use_mems_eval": true,
  "use_mems_train": false,
  "vocab_size": 32000
}

loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/pytorch_model.bin
Attempting to create safetensors variant
Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.bias', 'lm_loss.weight']
- This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.bias', 'logits_proj.weight', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
<ipython-input-4-b79c724d525b>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
Starting training for fold 1...
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 235
  Num Epochs = 15
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 4
  Total optimization steps = 120
  Number of trainable parameters = 117,311,235
Safetensors PR exists
 [104/120 10:47 < 01:41, 0.16 it/s, Epoch 13/15]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.828500	0.399816	0.576271	0.474576	0.436158	0.576271
2	1.560000	0.334967	0.508475	0.478773	0.472864	0.508475
3	1.222100	0.276452	0.644068	0.611573	0.661608	0.644068
4	0.945000	0.210736	0.762712	0.748625	0.779200	0.762712
5	0.901500	0.242239	0.694915	0.697543	0.733757	0.694915
6	0.762900	0.201991	0.711864	0.672693	0.765537	0.711864
7	0.663300	0.185067	0.745763	0.751584	0.773403	0.745763
8	0.602000	0.206034	0.745763	0.735583	0.763038	0.745763
9	0.573200	0.190735	0.677966	0.681004	0.688559	0.677966
10	0.474600	0.147059	0.762712	0.755719	0.778494	0.762712
11	0.433900	0.149856	0.745763	0.744703	0.751745	0.745763
12	0.370500	0.160296	0.728814	0.725813	0.733959	0.728814
13	0.329000	0.164009	0.728814	0.729548	0.736090	0.728814
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-8
Configuration saved in ./xlnet_results/fold_1/checkpoint-8/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-8/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-8/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-8/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-16
Configuration saved in ./xlnet_results/fold_1/checkpoint-16/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-16/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-16/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-16/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-24
Configuration saved in ./xlnet_results/fold_1/checkpoint-24/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-24/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-24/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-24/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-32
Configuration saved in ./xlnet_results/fold_1/checkpoint-32/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-32/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-32/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-32/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-40
Configuration saved in ./xlnet_results/fold_1/checkpoint-40/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-40/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-40/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-40/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-48
Configuration saved in ./xlnet_results/fold_1/checkpoint-48/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-48/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-48/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-48/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-56
Configuration saved in ./xlnet_results/fold_1/checkpoint-56/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-56/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-56/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-56/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-64
Configuration saved in ./xlnet_results/fold_1/checkpoint-64/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-64/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-64/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-64/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-72
Configuration saved in ./xlnet_results/fold_1/checkpoint-72/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-72/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-72/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-72/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-80
Configuration saved in ./xlnet_results/fold_1/checkpoint-80/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-80/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-80/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-80/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-88
Configuration saved in ./xlnet_results/fold_1/checkpoint-88/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-88/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-88/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-88/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-96
Configuration saved in ./xlnet_results/fold_1/checkpoint-96/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-96/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-96/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-96/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-104
Configuration saved in ./xlnet_results/fold_1/checkpoint-104/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-104/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-104/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-104/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_1/checkpoint-80 (score: 0.755718954248366).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Evaluating fold 1 on its validation set...
 [4/4 00:03]
Fold 1 - Validation F1: 0.7557, Accuracy: 0.7627
New best model found in fold 1 with F1: 0.7557
\n--- Fold 2/5 ---
Map: 100%
 235/235 [00:00<00:00, 314.27 examples/s]
Map: 100%
 59/59 [00:00<00:00, 286.43 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/config.json
Model config XLNetConfig {
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": false,
  "bos_token_id": 1,
  "clamp_len": -1,
  "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "gelu",
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "initializer_range": 0.02,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  },
  "layer_norm_eps": 1e-12,
  "mem_len": null,
  "model_type": "xlnet",
  "n_head": 12,
  "n_layer": 12,
  "pad_token_id": 5,
  "reuse_len": null,
  "same_length": false,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 250
    }
  },
  "transformers_version": "4.52.2",
  "untie_r": true,
  "use_mems_eval": true,
  "use_mems_train": false,
  "vocab_size": 32000
}

loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/pytorch_model.bin
Attempting to create safetensors variant
Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.bias', 'lm_loss.weight']
- This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.bias', 'logits_proj.weight', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
<ipython-input-4-b79c724d525b>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 235
  Num Epochs = 15
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 4
  Total optimization steps = 120
  Number of trainable parameters = 117,311,235
Safetensors PR exists
Starting training for fold 2...
 [ 64/120 06:49 < 06:09, 0.15 it/s, Epoch 8/15]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	2.053000	0.473726	0.372881	0.320119	0.348356	0.372881
2	1.539900	0.285416	0.576271	0.497821	0.438257	0.576271
3	1.134200	0.223000	0.677966	0.665369	0.663612	0.677966
4	0.845200	0.233825	0.661017	0.615015	0.640766	0.661017
5	0.752000	0.190141	0.711864	0.714270	0.728371	0.711864
6	0.577300	0.205325	0.711864	0.653988	0.742938	0.711864
7	0.469400	0.191072	0.694915	0.696483	0.708218	0.694915
8	0.343400	0.183887	0.728814	0.713678	0.728489	0.728814
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-8
Configuration saved in ./xlnet_results/fold_2/checkpoint-8/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-8/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-8/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-8/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-16
Configuration saved in ./xlnet_results/fold_2/checkpoint-16/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-16/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-16/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-16/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-24
Configuration saved in ./xlnet_results/fold_2/checkpoint-24/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-24/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-24/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-24/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-32
Configuration saved in ./xlnet_results/fold_2/checkpoint-32/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-32/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-32/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-32/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-40
Configuration saved in ./xlnet_results/fold_2/checkpoint-40/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-40/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-40/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-40/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-48
Configuration saved in ./xlnet_results/fold_2/checkpoint-48/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-48/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-48/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-48/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-56
Configuration saved in ./xlnet_results/fold_2/checkpoint-56/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-56/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-56/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-56/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-64
Configuration saved in ./xlnet_results/fold_2/checkpoint-64/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-64/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-64/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-64/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_2/checkpoint-40 (score: 0.7142696080273592).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Evaluating fold 2 on its validation set...
 [4/4 00:03]
Fold 2 - Validation F1: 0.7143, Accuracy: 0.7119
\n--- Fold 3/5 ---
Map: 100%
 235/235 [00:00<00:00, 314.02 examples/s]
Map: 100%
 59/59 [00:00<00:00, 282.31 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/config.json
Model config XLNetConfig {
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": false,
  "bos_token_id": 1,
  "clamp_len": -1,
  "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "gelu",
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "initializer_range": 0.02,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  },
  "layer_norm_eps": 1e-12,
  "mem_len": null,
  "model_type": "xlnet",
  "n_head": 12,
  "n_layer": 12,
  "pad_token_id": 5,
  "reuse_len": null,
  "same_length": false,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 250
    }
  },
  "transformers_version": "4.52.2",
  "untie_r": true,
  "use_mems_eval": true,
  "use_mems_train": false,
  "vocab_size": 32000
}

loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/pytorch_model.bin
Attempting to create safetensors variant
Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.bias', 'lm_loss.weight']
- This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.bias', 'logits_proj.weight', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
<ipython-input-4-b79c724d525b>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
Starting training for fold 3...
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 235
  Num Epochs = 15
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 4
  Total optimization steps = 120
  Number of trainable parameters = 117,311,235
Safetensors PR exists
 [ 72/120 08:01 < 05:30, 0.15 it/s, Epoch 9/15]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	2.423800	0.539708	0.322034	0.329787	0.417551	0.322034
2	1.604900	0.322947	0.610169	0.590034	0.590961	0.610169
3	1.076300	0.279456	0.745763	0.745174	0.748716	0.745763
4	0.822700	0.248765	0.762712	0.750409	0.778852	0.762712
5	0.634100	0.208343	0.762712	0.758692	0.766102	0.762712
6	0.501300	0.224892	0.779661	0.779803	0.780956	0.779661
7	0.401700	0.251382	0.745763	0.748863	0.754049	0.745763
8	0.294800	0.243357	0.745763	0.741600	0.744364	0.745763
9	0.227100	0.370441	0.627119	0.632318	0.703248	0.627119
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-8
Configuration saved in ./xlnet_results/fold_3/checkpoint-8/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-8/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-8/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-8/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-16
Configuration saved in ./xlnet_results/fold_3/checkpoint-16/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-16/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-16/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-16/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-24
Configuration saved in ./xlnet_results/fold_3/checkpoint-24/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-24/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-24/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-24/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-32
Configuration saved in ./xlnet_results/fold_3/checkpoint-32/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-32/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-32/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-32/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-40
Configuration saved in ./xlnet_results/fold_3/checkpoint-40/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-40/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-40/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-40/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-48
Configuration saved in ./xlnet_results/fold_3/checkpoint-48/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-48/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-48/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-48/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-56
Configuration saved in ./xlnet_results/fold_3/checkpoint-56/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-56/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-56/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-56/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-64
Configuration saved in ./xlnet_results/fold_3/checkpoint-64/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-64/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-64/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-64/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-72
Configuration saved in ./xlnet_results/fold_3/checkpoint-72/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-72/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-72/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-72/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_3/checkpoint-48 (score: 0.779802613879331).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Evaluating fold 3 on its validation set...
 [4/4 00:03]
Fold 3 - Validation F1: 0.7798, Accuracy: 0.7797
New best model found in fold 3 with F1: 0.7798
\n--- Fold 4/5 ---
Map: 100%
 235/235 [00:00<00:00, 299.82 examples/s]
Map: 100%
 59/59 [00:00<00:00, 168.24 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/config.json
Model config XLNetConfig {
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": false,
  "bos_token_id": 1,
  "clamp_len": -1,
  "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "gelu",
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "initializer_range": 0.02,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  },
  "layer_norm_eps": 1e-12,
  "mem_len": null,
  "model_type": "xlnet",
  "n_head": 12,
  "n_layer": 12,
  "pad_token_id": 5,
  "reuse_len": null,
  "same_length": false,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 250
    }
  },
  "transformers_version": "4.52.2",
  "untie_r": true,
  "use_mems_eval": true,
  "use_mems_train": false,
  "vocab_size": 32000
}

loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/pytorch_model.bin
Attempting to create safetensors variant
Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.bias', 'lm_loss.weight']
- This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.bias', 'logits_proj.weight', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
<ipython-input-4-b79c724d525b>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Safetensors PR exists
Using auto half precision backend
Starting training for fold 4...
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 235
  Num Epochs = 15
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 4
  Total optimization steps = 120
  Number of trainable parameters = 117,311,235
 [ 56/120 05:35 < 06:37, 0.16 it/s, Epoch 7/15]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.991200	0.447224	0.372881	0.389268	0.448896	0.372881
2	1.632400	0.403508	0.593220	0.473646	0.437225	0.593220
3	1.351800	0.308759	0.644068	0.624256	0.641362	0.644068
4	1.108400	0.286513	0.677966	0.633624	0.675210	0.677966
5	0.933800	0.290119	0.661017	0.632232	0.641986	0.661017
6	0.816600	0.288644	0.661017	0.620512	0.641990	0.661017
7	0.681800	0.272370	0.559322	0.568061	0.585256	0.559322
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-8
Configuration saved in ./xlnet_results/fold_4/checkpoint-8/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-8/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-8/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-8/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-16
Configuration saved in ./xlnet_results/fold_4/checkpoint-16/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-16/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-16/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-16/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-24
Configuration saved in ./xlnet_results/fold_4/checkpoint-24/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-24/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-24/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-24/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-32
Configuration saved in ./xlnet_results/fold_4/checkpoint-32/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-32/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-32/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-32/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-40
Configuration saved in ./xlnet_results/fold_4/checkpoint-40/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-40/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-40/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-40/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-48
Configuration saved in ./xlnet_results/fold_4/checkpoint-48/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-48/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-48/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-48/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-56
Configuration saved in ./xlnet_results/fold_4/checkpoint-56/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-56/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-56/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-56/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_4/checkpoint-32 (score: 0.6336243793870912).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 59
  Batch size = 16
Evaluating fold 4 on its validation set...
 [4/4 00:03]
Fold 4 - Validation F1: 0.6336, Accuracy: 0.6780
\n--- Fold 5/5 ---
Map: 100%
 236/236 [00:00<00:00, 305.78 examples/s]
Map: 100%
 58/58 [00:00<00:00, 283.75 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/config.json
Model config XLNetConfig {
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": false,
  "bos_token_id": 1,
  "clamp_len": -1,
  "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "gelu",
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "initializer_range": 0.02,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  },
  "layer_norm_eps": 1e-12,
  "mem_len": null,
  "model_type": "xlnet",
  "n_head": 12,
  "n_layer": 12,
  "pad_token_id": 5,
  "reuse_len": null,
  "same_length": false,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 250
    }
  },
  "transformers_version": "4.52.2",
  "untie_r": true,
  "use_mems_eval": true,
  "use_mems_train": false,
  "vocab_size": 32000
}

loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/pytorch_model.bin
Attempting to create safetensors variant
Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.bias', 'lm_loss.weight']
- This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.bias', 'logits_proj.weight', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
<ipython-input-4-b79c724d525b>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 236
  Num Epochs = 15
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 4
  Total optimization steps = 120
  Number of trainable parameters = 117,311,235
Safetensors PR exists
Starting training for fold 5...
 [ 40/120 04:38 < 09:47, 0.14 it/s, Epoch 5/15]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.990400	0.485224	0.344828	0.312226	0.303879	0.344828
2	1.538700	0.316774	0.758621	0.717657	0.835423	0.758621
3	1.179700	0.246256	0.689655	0.662518	0.681034	0.689655
4	0.969100	0.264849	0.672414	0.648989	0.659151	0.672414
5	0.739500	0.223791	0.620690	0.627622	0.637931	0.620690
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 58
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-8
Configuration saved in ./xlnet_results/fold_5/checkpoint-8/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-8/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-8/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-8/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 58
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-16
Configuration saved in ./xlnet_results/fold_5/checkpoint-16/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-16/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-16/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-16/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 58
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-24
Configuration saved in ./xlnet_results/fold_5/checkpoint-24/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-24/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-24/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-24/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 58
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-32
Configuration saved in ./xlnet_results/fold_5/checkpoint-32/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-32/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-32/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-32/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 58
  Batch size = 16
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-40
Configuration saved in ./xlnet_results/fold_5/checkpoint-40/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-40/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-40/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-40/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_5/checkpoint-16 (score: 0.7176568188167651).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 58
  Batch size = 16
Evaluating fold 5 on its validation set...
 [4/4 00:03]
Fold 5 - Validation F1: 0.7177, Accuracy: 0.7586
\n--- Cross-Validation Summary (Average over folds) ---
avg_eval_loss: 0.2331
avg_eval_accuracy: 0.7382
avg_eval_f1: 0.7202
avg_eval_precision: 0.7597
avg_eval_recall: 0.7382
\nLoading the overall best model from cross-validation...
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/config.json
Model config XLNetConfig {
  "architectures": [
    "XLNetLMHeadModel"
  ],
  "attn_type": "bi",
  "bi_data": false,
  "bos_token_id": 1,
  "clamp_len": -1,
  "d_head": 64,
  "d_inner": 3072,
  "d_model": 768,
  "dropout": 0.1,
  "end_n_top": 5,
  "eos_token_id": 2,
  "ff_activation": "gelu",
  "id2label": {
    "0": "LABEL_0",
    "1": "LABEL_1",
    "2": "LABEL_2"
  },
  "initializer_range": 0.02,
  "label2id": {
    "LABEL_0": 0,
    "LABEL_1": 1,
    "LABEL_2": 2
  },
  "layer_norm_eps": 1e-12,
  "mem_len": null,
  "model_type": "xlnet",
  "n_head": 12,
  "n_layer": 12,
  "pad_token_id": 5,
  "reuse_len": null,
  "same_length": false,
  "start_n_top": 5,
  "summary_activation": "tanh",
  "summary_last_dropout": 0.1,
  "summary_type": "last",
  "summary_use_proj": true,
  "task_specific_params": {
    "text-generation": {
      "do_sample": true,
      "max_length": 250
    }
  },
  "transformers_version": "4.52.2",
  "untie_r": true,
  "use_mems_eval": true,
  "use_mems_train": false,
  "vocab_size": 32000
}

loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/pytorch_model.bin
Attempting to create safetensors variant
Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.bias', 'lm_loss.weight']
- This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.bias', 'logits_proj.weight', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Configuration saved in ./xlnet_classifier_cv/config.json
Safetensors PR exists
Saving the best model to ./xlnet_classifier_cv...
Model weights saved in ./xlnet_classifier_cv/model.safetensors
tokenizer config file saved in ./xlnet_classifier_cv/tokenizer_config.json
Special tokens file saved in ./xlnet_classifier_cv/special_tokens_map.json
Plotting training history of the best fold...
Training history plot saved to xlnet_training_history_cv_best_fold.png
\nEvaluating the best model on the hold-out test set...
Map: 100%
 52/52 [00:00<00:00, 280.91 examples/s]
PyTorch: setting up devices
<ipython-input-4-b79c724d525b>:477: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  test_trainer = Trainer(
Using auto half precision backend

***** Running Prediction *****
  Num examples = 52
  Batch size = 16
\nTest Set Evaluation Report:
              precision    recall  f1-score   support

AI-Generated     1.0000    0.8889    0.9412         9
   Authentic     0.7778    0.7778    0.7778        27
     Generic     0.5882    0.6250    0.6061        16

    accuracy                         0.7500        52
   macro avg     0.7887    0.7639    0.7750        52
weighted avg     0.7579    0.7500    0.7532        52

Plotting confusion matrix for the test set...
Confusion matrix saved to xlnet_confusion_matrix_cv_test.png
\nExplaining prediction for a random sample from test set (True Label: Authentic):
Warning: XLNet embedding layer 'model.transformer.word_emb' not found. Using 'model.transformer.word_embedding' as fallback or check model structure.
  Could not generate explanation: CUDA out of memory. Tried to allocate 14.65 GiB. GPU 0 has a total capacity of 14.74 GiB of which 10.83 GiB is free. Process 23509 has 3.91 GiB memory in use. Of the allocated memory 3.32 GiB is allocated by PyTorch, and 461.80 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
\nXLNet training and evaluation with cross-validation finished.
Traceback (most recent call last):
  File "<ipython-input-4-b79c724d525b>", line 510, in train_model_with_cv
    explanation = explain_prediction(
                  ^^^^^^^^^^^^^^^^^^^
  File "<ipython-input-4-b79c724d525b>", line 292, in explain_prediction
    attributions, delta = lig.attribute(
                          ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/captum/log/dummy_log.py", line 39, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/captum/attr/_core/layer/layer_integrated_gradients.py", line 563, in attribute
    attributions = self.ig.attribute.__wrapped__(  # type: ignore
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/captum/attr/_core/integrated_gradients.py", line 289, in attribute
    attributions = self._attribute(
                   ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/captum/attr/_core/integrated_gradients.py", line 368, in _attribute
    grads = self.gradient_func(
            ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/captum/attr/_core/layer/layer_integrated_gradients.py", line 212, in _gradient_func
    output = _run_forward(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/captum/_utils/common.py", line 588, in _run_forward
    output = forward_func(
             ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py", line 818, in forward
    return model_forward(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/accelerate/utils/operations.py", line 806, in __call__
    return convert_to_fp32(self.model_forward(*args, **kwargs))
                           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/amp/autocast_mode.py", line 44, in decorate_autocast
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/xlnet/modeling_xlnet.py", line 1795, in forward
    transformer_outputs = self.transformer(
                          ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/xlnet/modeling_xlnet.py", line 1431, in forward
    outputs = layer_module(
              ^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/xlnet/modeling_xlnet.py", line 494, in forward
    outputs = self.rel_attn(
              ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1739, in _wrapped_call_impl
    return self._call_impl(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py", line 1750, in _call_impl
    return forward_call(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/xlnet/modeling_xlnet.py", line 425, in forward
    attn_vec = self.rel_attn_core(
               ^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/xlnet/modeling_xlnet.py", line 263, in rel_attn_core
    ac = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_w_bias, k_head_h)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/functional.py", line 407, in einsum
    return _VF.einsum(equation, operands)  # type: ignore[attr-defined]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 14.65 GiB. GPU 0 has a total capacity of 14.74 GiB of which 10.83 GiB is free. Process 23509 has 3.91 GiB memory in use. Of the allocated memory 3.32 GiB is allocated by PyTorch, and 461.80 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
"""
