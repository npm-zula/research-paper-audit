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
MAX_LENGTH = 384
BATCH_SIZE = 16  # Per device train batch size
EVAL_BATCH_SIZE = BATCH_SIZE * 2  # Per device eval batch size
# Effective batch size = BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS_PER_FOLD = 6  # Renamed for clarity
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
    df_full, class_names = load_data('/content/corpus-thefinal.csv')
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
            eval_strategy="epoch",
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
Model: xlnet-base-cased, Max Length: 384, Batch Size (per device): 16
Effective Batch Size: 64, Eval Batch Size: 32
Learning Rate: 2e-05, Epochs per fold: 6
Loading data...
Failed to decode with cp1252.
Successfully loaded data with latin-1 encoding.
Loaded 248 samples with 3 classes: ['AI-Generated' 'Authentic' 'Generic']
Train/CV set size: 210, Test set size: 38
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
spiece.model: 100%
 798k/798k [00:00<00:00, 5.22MB/s]
tokenizer.json: 100%
 1.38M/1.38M [00:00<00:00, 7.46MB/s]
loading file spiece.model from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/spiece.model
loading file added_tokens.json from cache at None
loading file special_tokens_map.json from cache at None
loading file tokenizer_config.json from cache at None
loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/tokenizer.json
loading file chat_template.jinja from cache at None
config.json: 100%
 760/760 [00:00<00:00, 44.4kB/s]
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
 168/168 [00:01<00:00, 89.13 examples/s]
Map: 100%
 42/42 [00:00<00:00, 86.40 examples/s]
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

Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
WARNING:huggingface_hub.file_download:Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
pytorch_model.bin: 100%
 467M/467M [00:04<00:00, 109MB/s]
loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/pytorch_model.bin
Attempting to create safetensors variant
Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.bias', 'lm_loss.weight']
- This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.bias', 'logits_proj.weight', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
<ipython-input-1-e41307206d80>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Safetensors PR exists
Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
WARNING:huggingface_hub.file_download:Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
model.safetensors: 100%
 467M/467M [00:10<00:00, 69.0MB/s]
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 168
  Num Epochs = 6
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 64
  Gradient Accumulation steps = 4
  Total optimization steps = 18
  Number of trainable parameters = 117,311,235
Starting training for fold 1...
 [18/18 03:22, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.897300	0.445024	0.333333	0.295156	0.265714	0.333333
2	1.746800	0.424223	0.357143	0.310235	0.275458	0.357143
3	1.629200	0.303016	0.642857	0.551501	0.781295	0.642857
4	1.379900	0.240394	0.690476	0.615476	0.789446	0.690476
5	1.134100	0.240659	0.523810	0.501232	0.528499	0.523810
6	0.939800	0.211348	0.571429	0.564469	0.560091	0.571429
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-3
Configuration saved in ./xlnet_results/fold_1/checkpoint-3/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-3/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-3/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-3/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-6
Configuration saved in ./xlnet_results/fold_1/checkpoint-6/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-6/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-6/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-6/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-9
Configuration saved in ./xlnet_results/fold_1/checkpoint-9/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-9/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-9/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-9/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-12
Configuration saved in ./xlnet_results/fold_1/checkpoint-12/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-12/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-12/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-12/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-15
Configuration saved in ./xlnet_results/fold_1/checkpoint-15/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-15/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-15/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-15/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-18
Configuration saved in ./xlnet_results/fold_1/checkpoint-18/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-18/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-18/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-18/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_1/checkpoint-12 (score: 0.6154761904761905).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Evaluating fold 1 on its validation set...
 [2/2 00:00]
Fold 1 - Validation F1: 0.6155, Accuracy: 0.6905
New best model found in fold 1 with F1: 0.6155
\n--- Fold 2/5 ---
Map: 100%
 168/168 [00:00<00:00, 357.71 examples/s]
Map: 100%
 42/42 [00:00<00:00, 317.83 examples/s]
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
<ipython-input-1-e41307206d80>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
Starting training for fold 2...
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 168
  Num Epochs = 6
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 64
  Gradient Accumulation steps = 4
  Total optimization steps = 18
  Number of trainable parameters = 117,311,235
Safetensors PR exists
 [18/18 03:25, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.806900	0.509021	0.333333	0.285572	0.450000	0.333333
2	1.757400	0.477038	0.428571	0.406542	0.452325	0.428571
3	1.669700	0.432335	0.404762	0.320728	0.479022	0.404762
4	1.434900	0.302419	0.785714	0.781599	0.801389	0.785714
5	1.146400	0.305368	0.619048	0.559873	0.748873	0.619048
6	1.007100	0.253005	0.642857	0.600515	0.613235	0.642857
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-3
Configuration saved in ./xlnet_results/fold_2/checkpoint-3/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-3/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-3/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-3/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-6
Configuration saved in ./xlnet_results/fold_2/checkpoint-6/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-6/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-6/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-6/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-9
Configuration saved in ./xlnet_results/fold_2/checkpoint-9/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-9/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-9/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-9/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-12
Configuration saved in ./xlnet_results/fold_2/checkpoint-12/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-12/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-12/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-12/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-15
Configuration saved in ./xlnet_results/fold_2/checkpoint-15/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-15/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-15/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-15/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-18
Configuration saved in ./xlnet_results/fold_2/checkpoint-18/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-18/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-18/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-18/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_2/checkpoint-12 (score: 0.7815994614900225).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Evaluating fold 2 on its validation set...
 [2/2 00:00]
Fold 2 - Validation F1: 0.7816, Accuracy: 0.7857
New best model found in fold 2 with F1: 0.7816
\n--- Fold 3/5 ---
Map: 100%
 168/168 [00:00<00:00, 355.84 examples/s]
Map: 100%
 42/42 [00:00<00:00, 266.50 examples/s]
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
<ipython-input-1-e41307206d80>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 168
  Num Epochs = 6
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 64
  Gradient Accumulation steps = 4
  Total optimization steps = 18
  Number of trainable parameters = 117,311,235
Starting training for fold 3...
Safetensors PR exists
 [18/18 03:58, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.997700	0.492201	0.476190	0.390476	0.354654	0.476190
2	1.812100	0.455974	0.547619	0.459034	0.526190	0.547619
3	1.622800	0.320407	0.523810	0.441327	0.390678	0.523810
4	1.419200	0.229911	0.714286	0.664814	0.797258	0.714286
5	1.158300	0.264450	0.642857	0.572562	0.667702	0.642857
6	1.073300	0.209979	0.690476	0.626077	0.787546	0.690476
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-3
Configuration saved in ./xlnet_results/fold_3/checkpoint-3/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-3/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-3/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-3/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-6
Configuration saved in ./xlnet_results/fold_3/checkpoint-6/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-6/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-6/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-6/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-9
Configuration saved in ./xlnet_results/fold_3/checkpoint-9/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-9/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-9/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-9/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-12
Configuration saved in ./xlnet_results/fold_3/checkpoint-12/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-12/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-12/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-12/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-15
Configuration saved in ./xlnet_results/fold_3/checkpoint-15/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-15/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-15/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-15/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-18
Configuration saved in ./xlnet_results/fold_3/checkpoint-18/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-18/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-18/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-18/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_3/checkpoint-12 (score: 0.6648142094366352).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Evaluating fold 3 on its validation set...
 [2/2 00:00]
Fold 3 - Validation F1: 0.6648, Accuracy: 0.7143
\n--- Fold 4/5 ---
Map: 100%
 168/168 [00:00<00:00, 360.20 examples/s]
Map: 100%
 42/42 [00:00<00:00, 297.93 examples/s]
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
<ipython-input-1-e41307206d80>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
Starting training for fold 4...
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 168
  Num Epochs = 6
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 64
  Gradient Accumulation steps = 4
  Total optimization steps = 18
  Number of trainable parameters = 117,311,235
Safetensors PR exists
 [18/18 04:24, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.864700	0.536209	0.357143	0.279365	0.251082	0.357143
2	1.823400	0.458868	0.476190	0.370738	0.499839	0.476190
3	1.670300	0.424404	0.452381	0.301587	0.226190	0.452381
4	1.621300	0.421898	0.428571	0.365429	0.485390	0.428571
5	1.378000	0.274164	0.619048	0.532981	0.615801	0.619048
6	1.201800	0.307119	0.571429	0.458450	0.388476	0.571429
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-3
Configuration saved in ./xlnet_results/fold_4/checkpoint-3/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-3/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-3/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-3/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-6
Configuration saved in ./xlnet_results/fold_4/checkpoint-6/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-6/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-6/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-6/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-9
Configuration saved in ./xlnet_results/fold_4/checkpoint-9/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-9/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-9/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-9/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-12
Configuration saved in ./xlnet_results/fold_4/checkpoint-12/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-12/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-12/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-12/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-15
Configuration saved in ./xlnet_results/fold_4/checkpoint-15/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-15/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-15/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-15/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-18
Configuration saved in ./xlnet_results/fold_4/checkpoint-18/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-18/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-18/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-18/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_4/checkpoint-15 (score: 0.5329813976872801).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Evaluating fold 4 on its validation set...
 [2/2 00:00]
Fold 4 - Validation F1: 0.5330, Accuracy: 0.6190
\n--- Fold 5/5 ---
Map: 100%
 168/168 [00:00<00:00, 337.97 examples/s]
Map: 100%
 42/42 [00:00<00:00, 296.09 examples/s]
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
<ipython-input-1-e41307206d80>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
Starting training for fold 5...
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 168
  Num Epochs = 6
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 64
  Gradient Accumulation steps = 4
  Total optimization steps = 18
  Number of trainable parameters = 117,311,235
Safetensors PR exists
 [18/18 03:48, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.942900	0.519247	0.357143	0.281818	0.236975	0.357143
2	1.770900	0.465745	0.380952	0.347884	0.466954	0.380952
3	1.604400	0.407928	0.380952	0.361473	0.514063	0.380952
4	1.307200	0.318722	0.619048	0.597032	0.620370	0.619048
5	0.982200	0.279446	0.547619	0.514849	0.512302	0.547619
6	0.811600	0.290726	0.571429	0.566727	0.587251	0.571429
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-3
Configuration saved in ./xlnet_results/fold_5/checkpoint-3/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-3/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-3/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-3/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-6
Configuration saved in ./xlnet_results/fold_5/checkpoint-6/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-6/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-6/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-6/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-9
Configuration saved in ./xlnet_results/fold_5/checkpoint-9/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-9/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-9/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-9/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-12
Configuration saved in ./xlnet_results/fold_5/checkpoint-12/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-12/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-12/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-12/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-15
Configuration saved in ./xlnet_results/fold_5/checkpoint-15/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-15/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-15/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-15/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-18
Configuration saved in ./xlnet_results/fold_5/checkpoint-18/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-18/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-18/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-18/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_5/checkpoint-12 (score: 0.597032436162871).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 42
  Batch size = 32
Evaluating fold 5 on its validation set...
 [2/2 00:00]
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

Fold 5 - Validation F1: 0.5970, Accuracy: 0.6190
\n--- Cross-Validation Summary (Average over folds) ---
avg_eval_loss: 0.2731
avg_eval_accuracy: 0.6857
avg_eval_f1: 0.6384
avg_eval_precision: 0.7249
avg_eval_recall: 0.6857
\nLoading the overall best model from cross-validation...
loading weights file pytorch_model.bin from cache at /root/.cache/huggingface/hub/models--xlnet-base-cased/snapshots/ceaa69c7bc5e512b5007106a7ccbb7daf24b2c79/pytorch_model.bin
Attempting to create safetensors variant
Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.bias', 'lm_loss.weight']
- This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.bias', 'logits_proj.weight', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Configuration saved in ./xlnet_classifier_cv/config.json
Saving the best model to ./xlnet_classifier_cv...
Safetensors PR exists
Model weights saved in ./xlnet_classifier_cv/model.safetensors
tokenizer config file saved in ./xlnet_classifier_cv/tokenizer_config.json
Special tokens file saved in ./xlnet_classifier_cv/special_tokens_map.json
Plotting training history of the best fold...
Training history plot saved to xlnet_training_history_cv_best_fold.png
\nEvaluating the best model on the hold-out test set...
Map: 100%
 38/38 [00:00<00:00, 246.62 examples/s]
PyTorch: setting up devices
<ipython-input-1-e41307206d80>:477: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  test_trainer = Trainer(
Using auto half precision backend

***** Running Prediction *****
  Num examples = 38
  Batch size = 32
\nTest Set Evaluation Report:
              precision    recall  f1-score   support

AI-Generated     0.8571    0.7500    0.8000         8
   Authentic     0.6190    0.7647    0.6842        17
     Generic     0.5000    0.3846    0.4348        13

    accuracy                         0.6316        38
   macro avg     0.6587    0.6331    0.6397        38
weighted avg     0.6284    0.6316    0.6233        38

Plotting confusion matrix for the test set...
Confusion matrix saved to xlnet_confusion_matrix_cv_test.png

"""


"""
Starting XLNet model training with 5-fold cross-validation and Focal Loss
Model: xlnet-base-cased, Max Length: 384, Batch Size (per device): 16
Effective Batch Size: 32, Eval Batch Size: 32
Learning Rate: 5e-05, Epochs per fold: 6
Loading data...
Successfully loaded data with cp1252 encoding.
Loaded 399 samples with 3 classes: ['AI-Generated' 'Authentic' 'Generic']
Train/CV set size: 339, Test set size: 60
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
 271/271 [00:01<00:00, 186.59 examples/s]
Map: 100%
 68/68 [00:00<00:00, 174.42 examples/s]
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
<ipython-input-2-b03fa68760c4>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 271
  Num Epochs = 6
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 2
  Total optimization steps = 54
  Number of trainable parameters = 117,311,235
Starting training for fold 1...
Safetensors PR exists
 [54/54 06:22, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	0.898800	0.410657	0.426471	0.376614	0.338965	0.426471
2	0.700900	0.433762	0.647059	0.535006	0.472465	0.647059
3	0.555700	0.306249	0.661765	0.567351	0.760861	0.661765
4	0.400100	0.271019	0.676471	0.651852	0.671462	0.676471
5	0.272300	0.286453	0.691176	0.695482	0.704657	0.691176
6	0.197300	0.303150	0.676471	0.651852	0.671462	0.676471
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-9
Configuration saved in ./xlnet_results/fold_1/checkpoint-9/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-9/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-9/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-9/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-18
Configuration saved in ./xlnet_results/fold_1/checkpoint-18/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-18/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-18/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-18/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-27
Configuration saved in ./xlnet_results/fold_1/checkpoint-27/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-27/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-27/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-27/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-36
Configuration saved in ./xlnet_results/fold_1/checkpoint-36/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-36/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-36/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-36/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-45
Configuration saved in ./xlnet_results/fold_1/checkpoint-45/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-45/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-45/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-45/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_1/checkpoint-54
Configuration saved in ./xlnet_results/fold_1/checkpoint-54/config.json
Model weights saved in ./xlnet_results/fold_1/checkpoint-54/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_1/checkpoint-54/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_1/checkpoint-54/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_1/checkpoint-45 (score: 0.6954824806449038).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Evaluating fold 1 on its validation set...
 [3/3 00:01]
Fold 1 - Validation F1: 0.6955, Accuracy: 0.6912
New best model found in fold 1 with F1: 0.6955
\n--- Fold 2/5 ---
Map: 100%
 271/271 [00:00<00:00, 307.95 examples/s]
Map: 100%
 68/68 [00:00<00:00, 310.18 examples/s]
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
<ipython-input-2-b03fa68760c4>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
Starting training for fold 2...
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 271
  Num Epochs = 6
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 2
  Total optimization steps = 54
  Number of trainable parameters = 117,311,235
Safetensors PR exists
 [54/54 06:49, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	0.909000	0.459542	0.514706	0.349800	0.264922	0.514706
2	0.873600	0.403484	0.544118	0.409948	0.552362	0.544118
3	0.816900	0.425536	0.529412	0.455437	0.596069	0.529412
4	0.751000	0.397527	0.558824	0.475118	0.624036	0.558824
5	0.589600	0.379080	0.602941	0.610351	0.624231	0.602941
6	0.464700	0.378947	0.617647	0.611291	0.611520	0.617647
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-9
Configuration saved in ./xlnet_results/fold_2/checkpoint-9/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-9/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-9/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-9/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-18
Configuration saved in ./xlnet_results/fold_2/checkpoint-18/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-18/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-18/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-18/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-27
Configuration saved in ./xlnet_results/fold_2/checkpoint-27/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-27/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-27/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-27/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-36
Configuration saved in ./xlnet_results/fold_2/checkpoint-36/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-36/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-36/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-36/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-45
Configuration saved in ./xlnet_results/fold_2/checkpoint-45/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-45/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-45/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-45/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_2/checkpoint-54
Configuration saved in ./xlnet_results/fold_2/checkpoint-54/config.json
Model weights saved in ./xlnet_results/fold_2/checkpoint-54/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_2/checkpoint-54/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_2/checkpoint-54/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_2/checkpoint-54 (score: 0.6112906701141995).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Evaluating fold 2 on its validation set...
 [3/3 00:01]
Fold 2 - Validation F1: 0.6113, Accuracy: 0.6176
\n--- Fold 3/5 ---
Map: 100%
 271/271 [00:00<00:00, 304.98 examples/s]
Map: 100%
 68/68 [00:00<00:00, 296.17 examples/s]
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
<ipython-input-2-b03fa68760c4>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 271
  Num Epochs = 6
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 2
  Total optimization steps = 54
  Number of trainable parameters = 117,311,235
Starting training for fold 3...
Safetensors PR exists
 [54/54 06:17, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	0.939500	0.423745	0.411765	0.373239	0.346609	0.411765
2	0.689500	0.404963	0.573529	0.462256	0.416916	0.573529
3	0.578900	0.375732	0.647059	0.632897	0.697041	0.647059
4	0.530600	0.360699	0.588235	0.499224	0.603190	0.588235
5	0.439100	0.393389	0.485294	0.464095	0.686029	0.485294
6	0.382800	0.322979	0.632353	0.626763	0.648734	0.632353
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-9
Configuration saved in ./xlnet_results/fold_3/checkpoint-9/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-9/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-9/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-9/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-18
Configuration saved in ./xlnet_results/fold_3/checkpoint-18/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-18/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-18/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-18/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-27
Configuration saved in ./xlnet_results/fold_3/checkpoint-27/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-27/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-27/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-27/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-36
Configuration saved in ./xlnet_results/fold_3/checkpoint-36/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-36/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-36/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-36/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-45
Configuration saved in ./xlnet_results/fold_3/checkpoint-45/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-45/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-45/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-45/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_3/checkpoint-54
Configuration saved in ./xlnet_results/fold_3/checkpoint-54/config.json
Model weights saved in ./xlnet_results/fold_3/checkpoint-54/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_3/checkpoint-54/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_3/checkpoint-54/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_3/checkpoint-27 (score: 0.6328972909106253).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Evaluating fold 3 on its validation set...
 [3/3 00:01]
Fold 3 - Validation F1: 0.6329, Accuracy: 0.6471
\n--- Fold 4/5 ---
Map: 100%
 271/271 [00:00<00:00, 311.41 examples/s]
Map: 100%
 68/68 [00:00<00:00, 294.08 examples/s]
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
<ipython-input-2-b03fa68760c4>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
Starting training for fold 4...
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 271
  Num Epochs = 6
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 2
  Total optimization steps = 54
  Number of trainable parameters = 117,311,235
Safetensors PR exists
 [54/54 06:24, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	0.933900	0.426753	0.500000	0.333333	0.250000	0.500000
2	0.848300	0.449921	0.470588	0.405308	0.359069	0.470588
3	0.841400	0.400281	0.602941	0.495150	0.484571	0.602941
4	0.665600	0.315500	0.544118	0.549208	0.570519	0.544118
5	0.564000	0.303607	0.617647	0.558618	0.600218	0.617647
6	0.457600	0.254265	0.647059	0.659165	0.705206	0.647059
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-9
Configuration saved in ./xlnet_results/fold_4/checkpoint-9/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-9/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-9/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-9/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-18
Configuration saved in ./xlnet_results/fold_4/checkpoint-18/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-18/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-18/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-18/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-27
Configuration saved in ./xlnet_results/fold_4/checkpoint-27/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-27/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-27/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-27/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-36
Configuration saved in ./xlnet_results/fold_4/checkpoint-36/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-36/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-36/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-36/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-45
Configuration saved in ./xlnet_results/fold_4/checkpoint-45/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-45/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-45/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-45/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_4/checkpoint-54
Configuration saved in ./xlnet_results/fold_4/checkpoint-54/config.json
Model weights saved in ./xlnet_results/fold_4/checkpoint-54/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_4/checkpoint-54/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_4/checkpoint-54/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_4/checkpoint-54 (score: 0.6591645353793691).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 68
  Batch size = 32
Evaluating fold 4 on its validation set...
 [3/3 00:01]
Fold 4 - Validation F1: 0.6592, Accuracy: 0.6471
\n--- Fold 5/5 ---
Map: 100%
 272/272 [00:00<00:00, 313.45 examples/s]
Map: 100%
 67/67 [00:00<00:00, 285.86 examples/s]
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
Safetensors PR exists
Some weights of the model checkpoint at xlnet-base-cased were not used when initializing XLNetForSequenceClassification: ['lm_loss.bias', 'lm_loss.weight']
- This IS expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing XLNetForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of XLNetForSequenceClassification were not initialized from the model checkpoint at xlnet-base-cased and are newly initialized: ['logits_proj.bias', 'logits_proj.weight', 'sequence_summary.summary.bias', 'sequence_summary.summary.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
<ipython-input-2-b03fa68760c4>:81: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `CustomTrainer.__init__`. Use `processing_class` instead.
  super().__init__(*args, **kwargs)
Using auto half precision backend
Starting training for fold 5...
The following columns in the Training set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 272
  Num Epochs = 6
  Instantaneous batch size per device = 16
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 2
  Total optimization steps = 54
  Number of trainable parameters = 117,311,235
 [54/54 07:00, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	0.951700	0.436914	0.507463	0.391866	0.349861	0.507463
2	0.757700	0.372341	0.507463	0.498668	0.588825	0.507463
3	0.511800	0.367570	0.611940	0.562238	0.634605	0.611940
4	0.399700	0.396010	0.567164	0.570756	0.596042	0.567164
5	0.260800	0.344390	0.582090	0.594657	0.651427	0.582090
6	0.157900	0.297993	0.611940	0.618262	0.635732	0.611940
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 67
  Batch size = 32
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-9
Configuration saved in ./xlnet_results/fold_5/checkpoint-9/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-9/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-9/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-9/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 67
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-18
Configuration saved in ./xlnet_results/fold_5/checkpoint-18/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-18/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-18/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-18/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 67
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-27
Configuration saved in ./xlnet_results/fold_5/checkpoint-27/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-27/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-27/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-27/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 67
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-36
Configuration saved in ./xlnet_results/fold_5/checkpoint-36/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-36/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-36/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-36/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 67
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-45
Configuration saved in ./xlnet_results/fold_5/checkpoint-45/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-45/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-45/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-45/special_tokens_map.json
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 67
  Batch size = 32
Saving model checkpoint to ./xlnet_results/fold_5/checkpoint-54
Configuration saved in ./xlnet_results/fold_5/checkpoint-54/config.json
Model weights saved in ./xlnet_results/fold_5/checkpoint-54/model.safetensors
tokenizer config file saved in ./xlnet_results/fold_5/checkpoint-54/tokenizer_config.json
Special tokens file saved in ./xlnet_results/fold_5/checkpoint-54/special_tokens_map.json


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./xlnet_results/fold_5/checkpoint-54 (score: 0.6182617567553353).
The following columns in the Evaluation set don't have a corresponding argument in `XLNetForSequenceClassification.forward` and have been ignored: __index_level_0__. If __index_level_0__ are not expected by `XLNetForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 67
  Batch size = 32
Evaluating fold 5 on its validation set...
 [3/3 00:01]
Fold 5 - Validation F1: 0.6183, Accuracy: 0.6119
\n--- Cross-Validation Summary (Average over folds) ---
avg_eval_loss: 0.3187
avg_eval_accuracy: 0.6430
avg_eval_f1: 0.6434
avg_eval_precision: 0.6708
avg_eval_recall: 0.6430
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
Saving the best model to ./xlnet_classifier_cv...
Model weights saved in ./xlnet_classifier_cv/model.safetensors
Safetensors PR exists
tokenizer config file saved in ./xlnet_classifier_cv/tokenizer_config.json
Special tokens file saved in ./xlnet_classifier_cv/special_tokens_map.json
Plotting training history of the best fold...
Training history plot saved to xlnet_training_history_cv_best_fold.png
\nEvaluating the best model on the hold-out test set...
Map: 100%
 60/60 [00:00<00:00, 256.38 examples/s]
PyTorch: setting up devices
<ipython-input-2-b03fa68760c4>:477: FutureWarning: `tokenizer` is deprecated and will be removed in version 5.0.0 for `Trainer.__init__`. Use `processing_class` instead.
  test_trainer = Trainer(
Using auto half precision backend

***** Running Prediction *****
  Num examples = 60
  Batch size = 32
\nTest Set Evaluation Report:
              precision    recall  f1-score   support

AI-Generated     0.9000    0.7500    0.8182        12
   Authentic     0.6667    0.6667    0.6667        30
     Generic     0.4500    0.5000    0.4737        18

    accuracy                         0.6333        60
   macro avg     0.6722    0.6389    0.6528        60
weighted avg     0.6483    0.6333    0.6391        60

Plotting confusion matrix for the test set...
Confusion matrix saved to xlnet_confusion_matrix_cv_test.png
\nExplaining prediction for a random sample from test set (True Label: AI-Generated):
Warning: XLNet embedding layer 'model.transformer.word_emb' not found. Using 'model.transformer.word_embedding' as fallback or check model structure.
  Could not generate explanation: CUDA out of memory. Tried to allocate 16.48 GiB. GPU 0 has a total capacity of 14.74 GiB of which 740.12 MiB is free. Process 14957 has 14.02 GiB memory in use. Of the allocated memory 12.77 GiB is allocated by PyTorch, and 1.11 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
\nXLNet training and evaluation with cross-validation finished.
Traceback (most recent call last):
  File "<ipython-input-2-b03fa68760c4>", line 510, in train_model_with_cv
    explanation = explain_prediction(
                  ^^^^^^^^^^^^^^^^^^^
  File "<ipython-input-2-b03fa68760c4>", line 292, in explain_prediction
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
  File "/usr/local/lib/python3.11/dist-packages/transformers/models/xlnet/modeling_xlnet.py", line 266, in rel_attn_core
    bd = torch.einsum("ibnd,jbnd->bnij", q_head + self.r_r_bias, k_head_r)
         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/functional.py", line 407, in einsum
    return _VF.einsum(equation, operands)  # type: ignore[attr-defined]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.48 GiB. GPU 0 has a total capacity of 14.74 GiB of which 740.12 MiB is free. Process 14957 has 14.02 GiB memory in use. Of the allocated memory 12.77 GiB is allocated by PyTorch, and 1.11 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
"""
