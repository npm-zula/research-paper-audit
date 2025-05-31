import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from transformers import (
    RobertaTokenizer,
    RobertaForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
    # get_scheduler, # Trainer handles scheduler based on args
)
from datasets import Dataset
# import evaluate # Not strictly necessary if using sklearn metrics via compute_metrics
from tqdm.auto import tqdm
# Used for manual evaluation if needed, Trainer has get_eval_dataloader
from torch.utils.data import DataLoader
from transformers.utils import logging
from captum.attr import LayerIntegratedGradients  # For explainability
import random  # For seeding
import copy  # For deep copying model states
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
MODEL_NAME = "roberta-base"
MAX_LENGTH = 384
BATCH_SIZE = 8  # Per device train batch size for Trainer
EVAL_BATCH_SIZE = BATCH_SIZE * 2  # Per device eval batch size
GRADIENT_ACCUMULATION_STEPS = 2
LEARNING_RATE = 5e-5
NUM_EPOCHS_PER_FOLD = 8  # Renamed for clarity
WEIGHT_DECAY = 0.02
WARMUP_RATIO = 0.1
CROSS_VAL_FOLDS = 5

# Function to load and preprocess data with robust encoding handling


def load_data(file_path):
    # Try multiple encodings
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
            continue  # Continue to next encoding

    if df is None:
        print("All standard encodings failed. Trying utf-8 with error replacement.")
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')

    # Handle missing values more robustly
    for col in ['Paper Title', 'Abstract', 'Review Text', 'Review Type']:
        if col in df.columns:
            df[col] = df[col].fillna('')
        else:
            print(
                f"Warning: Column {col} not found in CSV. It will be treated as empty.")
            df[col] = ''  # Add empty column if missing to prevent key errors

    # Clean text data - remove non-ASCII characters
    for col in ['Paper Title', 'Abstract', 'Review Text']:
        if col in df.columns:  # Check again in case it was added as empty
            df[col] = df[col].apply(lambda x: str(x).encode(
                'ascii', 'ignore').decode('ascii'))

    # Create weighted text combination with special tokens to help model distinguish sections
    df['text'] = (
        "<title>" + df['Paper Title'] + "</title> " +
        "<abstract>" + df['Abstract'] + "</abstract> " +
        "<review>" + df['Review Text'] + "</review>"
    )

    # Drop rows with missing values in critical columns AFTER text combination
    df = df.dropna(subset=['text', 'Review Type'])
    if df.empty:
        raise ValueError(
            "DataFrame is empty after preprocessing and dropping NA values. Check input data and preprocessing steps.")

    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Review Type'])

    return df, label_encoder.classes_, label_encoder

# Tokenization function


def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt"
    )

# Compute metrics function with more comprehensive metrics


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    # Calculate multiple metrics
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

# Function to plot training history


def plot_training_history(history, filename='roberta_training_history.png'):  # Added filename
    plt.figure(figsize=(12, 4))

    # Check if keys exist before plotting
    if 'train_loss' in history and 'eval_loss' in history:
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Training Loss')
        plt.plot(history['eval_loss'], label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.title('Training and Validation Loss')
    else:
        print("Warning: Loss history not complete for plotting.")

    if 'eval_accuracy' in history and 'eval_f1' in history:
        plt.subplot(1, 2, 2)
        plt.plot(history['eval_accuracy'], label='Accuracy')
        plt.plot(history['eval_f1'], label='F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Score')
        plt.legend()
        plt.title('Validation Metrics')
    else:
        print("Warning: Metrics history not complete for plotting.")

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Training history plot saved to {filename}")

# Function to plot confusion matrix


def plot_confusion_matrix(y_true, y_pred, class_names, filename='roberta_confusion_matrix.png'):  # Added filename
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Confusion matrix plot saved to {filename}")

# Model explainability function using Integrated Gradients


def explain_prediction(text, model, tokenizer, class_names, device):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                       truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get prediction
    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class = class_names[predicted_class_idx]

    # Setup for attribution
    lig = LayerIntegratedGradients(model, model.roberta.embeddings)

    # Get attributions
    attributions, delta = lig.attribute(
        inputs=inputs['input_ids'],
        baselines=torch.zeros_like(inputs['input_ids']),
        target=predicted_class_idx,
        return_convergence_delta=True,
        n_steps=50
    )

    # Convert attributions to word importances
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()

    # Create visualization data
    word_importances = []
    for token, importance in zip(tokens, attributions):
        if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
            word_importances.append((token, float(importance)))

    # Sort by absolute importance
    word_importances.sort(key=lambda x: abs(x[1]), reverse=True)

    return {
        "prediction": predicted_class,
        "confidence": float(probabilities[0][predicted_class_idx]),
        "top_words": word_importances[:20]  # Top 20 most influential words
    }

# Main training function with cross-validation


def train_model_with_cv():
    print("Loading data...")
    df_full, class_names, label_encoder = load_data(
        '/content/corpus-final.csv')
    print(
        f"Loaded {len(df_full)} samples with {len(class_names)} classes: {class_names}")

    # Split data into Train/CV and Test sets
    train_cv_df, test_df = train_test_split(
        df_full, test_size=0.2, stratify=df_full['label'], random_state=SEED
    )
    print(
        f"Train/CV set size: {len(train_cv_df)}, Test set size: {len(test_df)}")

    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)

    # Setup cross-validation
    skf = StratifiedKFold(n_splits=CROSS_VAL_FOLDS,
                          shuffle=True, random_state=SEED)

    fold_results_summary = []
    best_overall_val_f1 = 0.0
    best_model_state_dict = None
    best_fold_log_history = None  # To store log_history of the best fold

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_cv_df, train_cv_df['label'])):
        print(f"\\n{'='*50}\\nTraining Fold {fold+1}/{CROSS_VAL_FOLDS}\\n{'='*50}")

        current_train_df = train_cv_df.iloc[train_idx].reset_index(drop=True)
        current_val_df = train_cv_df.iloc[val_idx].reset_index(drop=True)

        train_dataset_fold = Dataset.from_pandas(current_train_df)
        val_dataset_fold = Dataset.from_pandas(current_val_df)

        train_dataset_fold = train_dataset_fold.map(
            lambda ex: tokenize_function(ex, tokenizer), batched=True)
        val_dataset_fold = val_dataset_fold.map(
            lambda ex: tokenize_function(ex, tokenizer), batched=True)

        train_dataset_fold.set_format(
            type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset_fold.set_format(type='torch', columns=[
                                    'input_ids', 'attention_mask', 'label'])

        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(class_names),
            id2label={i: label for i, label in enumerate(class_names)},
            label2id={label: i for i, label in enumerate(class_names)}
        )
        # model.to(device) # Trainer handles device placement

        training_args_fold = TrainingArguments(
            output_dir=f"./roberta_results/fold_{fold+1}",
            eval_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=NUM_EPOCHS_PER_FOLD,
            weight_decay=WEIGHT_DECAY,
            warmup_ratio=WARMUP_RATIO,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            logging_dir=f"./roberta_logs/fold_{fold+1}",
            logging_strategy="epoch",
            report_to="tensorboard",  # or "all" or "none"
            save_total_limit=1  # Only keep the best checkpoint from this fold
        )

        trainer_fold = Trainer(
            model=model,
            args=training_args_fold,
            train_dataset=train_dataset_fold,
            eval_dataset=val_dataset_fold,
            compute_metrics=compute_metrics,
            # Patience of 3 epochs
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )

        print(f"Starting training for Fold {fold+1}...")
        trainer_fold.train()

        print(
            f"Evaluating best model of Fold {fold+1} on its validation set...")
        eval_results_fold = trainer_fold.evaluate(
            eval_dataset=val_dataset_fold)
        fold_val_f1 = eval_results_fold['eval_f1']

        fold_results_summary.append({
            'fold': fold + 1,
            'val_loss': eval_results_fold['eval_loss'],
            'val_accuracy': eval_results_fold['eval_accuracy'],
            'val_f1': fold_val_f1,
            'val_precision': eval_results_fold['eval_precision'],
            'val_recall': eval_results_fold['eval_recall']
        })
        print(
            f"Fold {fold+1} Validation: F1={fold_val_f1:.4f}, Accuracy={eval_results_fold['eval_accuracy']:.4f}")

        if fold_val_f1 > best_overall_val_f1:
            best_overall_val_f1 = fold_val_f1
            best_model_state_dict = copy.deepcopy(
                trainer_fold.model.state_dict())
            best_fold_log_history = copy.deepcopy(
                trainer_fold.state.log_history)
            print(
                f"New best model found in Fold {fold+1} with Val F1: {best_overall_val_f1:.4f}")

    print("\\nAverage Cross-Validation Results:")
    avg_cv_metrics = pd.DataFrame(fold_results_summary).mean().to_dict()
    for metric, value in avg_cv_metrics.items():
        if metric != 'fold':
            print(f"  Average {metric.replace('val_', 'eval_')}: {value:.4f}")

    if best_model_state_dict is None:
        print("No best model was found (e.g., all folds failed or produced NaN metrics). Exiting.")
        return None, None, None

    print("\\nLoading the overall best model for final evaluation...")
    overall_best_model = RobertaForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label={i: label for i, label in enumerate(class_names)},
        label2id={label: i for i, label in enumerate(class_names)}
    )
    overall_best_model.load_state_dict(best_model_state_dict)
    overall_best_model.to(device)

    if best_fold_log_history:
        history_to_plot = {
            'train_loss': [], 'eval_loss': [], 'eval_accuracy': [],
            'eval_f1': [], 'eval_precision': [], 'eval_recall': []
        }
        temp_train_loss = {}
        for log_entry in best_fold_log_history:
            epoch_num = log_entry.get('epoch')
            if epoch_num is not None:  # Ensure epoch number exists
                epoch_num = float(epoch_num)  # Trainer logs epoch as float
                if 'loss' in log_entry and 'eval_loss' not in log_entry:  # Training loss for the epoch
                    # For epoch-level logging, 'loss' is the training loss of that epoch
                    temp_train_loss[int(epoch_num)] = log_entry['loss']
                elif 'eval_loss' in log_entry:  # Evaluation metrics
                    current_epoch_int = int(epoch_num)
                    if current_epoch_int in temp_train_loss:  # Check if train loss for this epoch was logged
                        history_to_plot['train_loss'].append(
                            temp_train_loss[current_epoch_int])
                        history_to_plot['eval_loss'].append(
                            log_entry['eval_loss'])
                        history_to_plot['eval_accuracy'].append(
                            log_entry['eval_accuracy'])
                        history_to_plot['eval_f1'].append(log_entry['eval_f1'])
                        history_to_plot['eval_precision'].append(
                            log_entry['eval_precision'])
                        history_to_plot['eval_recall'].append(
                            log_entry['eval_recall'])

        if history_to_plot['eval_f1']:
            plot_training_history(
                history_to_plot, filename='roberta_training_history_cv_best_fold.png')
        else:
            print(
                "Could not extract sufficient history for plotting the best fold. Check log_history content.")
    else:
        print("No log history from the best fold to plot.")

    print("\\nPreparing test set for final evaluation...")
    test_dataset_final = Dataset.from_pandas(test_df)
    test_dataset_final = test_dataset_final.map(
        lambda ex: tokenize_function(ex, tokenizer), batched=True)
    test_dataset_final.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Manual evaluation loop for the test set
    overall_best_model.eval()
    all_predictions_test = []
    all_labels_test = []

    test_loader_final = DataLoader(
        test_dataset_final, batch_size=EVAL_BATCH_SIZE)

    for batch in tqdm(test_loader_final, desc="Final Test Set Evaluation"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        with torch.no_grad():
            outputs = overall_best_model(
                input_ids=input_ids, attention_mask=attention_mask)

        predictions = torch.argmax(outputs.logits, dim=1)
        all_predictions_test.extend(predictions.cpu().numpy())
        all_labels_test.extend(labels.cpu().numpy())

    test_accuracy = np.mean(np.array(all_predictions_test)
                            == np.array(all_labels_test))
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        all_labels_test, all_predictions_test, average='weighted', zero_division=0
    )
    print(
        f"  Test Set: Accuracy={test_accuracy:.4f}, F1={test_f1:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}")

    plot_confusion_matrix(all_labels_test, all_predictions_test,
                          class_names, filename='roberta_confusion_matrix_cv_test.png')

    report_dict_test = classification_report(all_labels_test, all_predictions_test,
                                             target_names=class_names, output_dict=True, zero_division=0)
    report_str_test = classification_report(all_labels_test, all_predictions_test,
                                            target_names=class_names, zero_division=0)

    print("\\nTest Set Performance Report (Overall Best Model):")
    print(report_str_test)

    print("\\nTest Set Performance Summary (Overall Best Model):")
    for cls_name in class_names:  # Use cls_name to avoid conflict
        if cls_name in report_dict_test:
            print(f"  {cls_name}: F1={report_dict_test[cls_name]['f1-score']:.4f}, "
                  f"Precision={report_dict_test[cls_name]['precision']:.4f}, "
                  f"Recall={report_dict_test[cls_name]['recall']:.4f}")

    if 'weighted avg' in report_dict_test:
        print(f"\\n  Overall Test: F1={report_dict_test['weighted avg']['f1-score']:.4f}, "
              f"Accuracy={report_dict_test['accuracy']:.4f}")

    print("\\nSaving overall best model and tokenizer...")
    overall_best_model.save_pretrained("./roberta_classifier_cv")
    tokenizer.save_pretrained("./roberta_classifier_cv")

    return overall_best_model, tokenizer, class_names

# Prediction function with confidence scores (can be used with the final model)


def predict(text, model, tokenizer, class_names, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.to(device)
    model.eval()

    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probabilities = F.softmax(outputs.logits, dim=-1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class = class_names[predicted_class_idx]

    # Get confidence scores for all classes
    confidence_scores = {class_names[i]: float(
        probabilities[0][i]) for i in range(len(class_names))}

    return {
        "prediction": predicted_class,
        "confidence": float(probabilities[0][predicted_class_idx]),
        "all_scores": confidence_scores
    }

# Function to evaluate model on test set and generate detailed report (This is now part of train_model_with_cv)
# def evaluate_model(model, tokenizer, test_df, class_names, device=None):
# ... (Can be removed or kept for standalone evaluation if needed, but train_model_with_cv handles the main test eval)


# Main execution
if __name__ == "__main__":
    print(
        f"Starting RoBERTa model training with {CROSS_VAL_FOLDS}-fold cross-validation")
    print(f"Model: {MODEL_NAME}, Max Length: {MAX_LENGTH}, Batch Size: {BATCH_SIZE} (Train), {EVAL_BATCH_SIZE} (Eval)")
    print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(
        f"Learning Rate: {LEARNING_RATE}, Epochs per fold: {NUM_EPOCHS_PER_FOLD}")

    final_model, final_tokenizer, final_class_names = train_model_with_cv()

    if final_model and final_tokenizer and final_class_names:
        print("\\nTraining, cross-validation, and final test evaluation complete.")
        print(f"Best model and tokenizer saved to ./roberta_classifier_cv")
        print("Visualization files saved: roberta_training_history_cv_best_fold.png, roberta_confusion_matrix_cv_test.png")

        # Example of model explainability using the final best model
        print("\\nModel Explainability Example (using the best CV model):")
        example_text = "<title>Advances in Natural Language Processing</title> <abstract>This paper presents a novel approach to sentiment analysis using transformer models.</abstract> <review>The methodology is sound but lacks comparison with recent work.</review>"

        current_device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        try:
            explanation = explain_prediction(
                example_text, final_model, final_tokenizer, final_class_names, current_device)
            print(f"Prediction: {explanation['prediction']}")
            print(f"Confidence: {explanation['confidence']:.4f}")
            print("Top influential words:")
            for word, importance in explanation['top_words'][:10]:
                print(f"  {word}: {importance:.4f}")
        except Exception as e:
            print(f"Could not generate explanation: {e}")
            print("Ensure the model passed to explain_prediction is the base RobertaModel if LayerIntegratedGradients targets model.roberta.embeddings, or adjust the layer target.")
            print("The 'explain_prediction' function might need 'final_model.roberta' if 'final_model' is 'RobertaForSequenceClassification'.")
            # For Captum with RobertaForSequenceClassification, the target layer for Integrated Gradients
            # is typically model.roberta.embeddings. The `explain_prediction` function seems to assume this.
            # If `final_model` is an instance of `RobertaForSequenceClassification`, then `final_model.roberta.embeddings` is correct.

    else:
        print(
            "\\nTraining process did not complete successfully or no best model was found.")


"""
Starting RoBERTa model training with 5-fold cross-validation
Model: roberta-base, Max Length: 384, Batch Size: 8 (Train), 16 (Eval)
Gradient Accumulation Steps: 2
Learning Rate: 5e-05, Epochs per fold: 8
Loading data...
Failed to decode with cp1252.
Successfully loaded data with latin-1 encoding.
Loaded 248 samples with 3 classes: ['AI-Generated' 'Authentic' 'Generic']
Train/CV set size: 198, Test set size: 50
loading file vocab.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/vocab.json
loading file merges.txt from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/merges.txt
loading file added_tokens.json from cache at None
loading file special_tokens_map.json from cache at None
loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/tokenizer_config.json
loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/tokenizer.json
loading file chat_template.jinja from cache at None
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.3",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

Using device: cuda
\n==================================================\nTraining Fold 1/5\n==================================================
Map: 100%
 158/158 [00:02<00:00, 62.38 examples/s]
Map: 100%
 40/40 [00:00<00:00, 40.48 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.3",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 158
  Num Epochs = 8
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 2
  Total optimization steps = 80
  Number of trainable parameters = 124,647,939
Starting training for Fold 1...
 [80/80 03:57, Epoch 8/8]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.099200	1.025159	0.450000	0.279310	0.202500	0.450000
2	0.998900	0.916949	0.450000	0.279310	0.202500	0.450000
3	0.859500	0.772675	0.650000	0.518848	0.439068	0.650000
4	0.659600	0.568665	0.650000	0.638333	0.640909	0.650000
5	0.474700	0.651655	0.625000	0.618791	0.614316	0.625000
6	0.309000	0.698972	0.725000	0.725806	0.735882	0.725000
7	0.218600	0.807765	0.700000	0.701176	0.706250	0.700000
8	0.217600	0.750636	0.750000	0.750000	0.767460	0.750000
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-10
Configuration saved in ./roberta_results/fold_1/checkpoint-10/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-10/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-20
Configuration saved in ./roberta_results/fold_1/checkpoint-20/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-20/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-30
Configuration saved in ./roberta_results/fold_1/checkpoint-30/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-30/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-10] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-40
Configuration saved in ./roberta_results/fold_1/checkpoint-40/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-40/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-20] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-50
Configuration saved in ./roberta_results/fold_1/checkpoint-50/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-50/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-30] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-60
Configuration saved in ./roberta_results/fold_1/checkpoint-60/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-60/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-40] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-70
Configuration saved in ./roberta_results/fold_1/checkpoint-70/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-70/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-50] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-80
Configuration saved in ./roberta_results/fold_1/checkpoint-80/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-80/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-60] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_1/checkpoint-80 (score: 0.75).
Deleting older checkpoint [roberta_results/fold_1/checkpoint-70] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Evaluating best model of Fold 1 on its validation set...
 [3/3 00:00]
Fold 1 Validation: F1=0.7500, Accuracy=0.7500
New best model found in Fold 1 with Val F1: 0.7500
\n==================================================\nTraining Fold 2/5\n==================================================
Map: 100%
 158/158 [00:00<00:00, 184.37 examples/s]
Map: 100%
 40/40 [00:00<00:00, 131.40 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.3",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
Starting training for Fold 2...
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 158
  Num Epochs = 8
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 2
  Total optimization steps = 80
  Number of trainable parameters = 124,647,939
 [80/80 03:27, Epoch 8/8]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.082600	1.044531	0.450000	0.279310	0.202500	0.450000
2	1.051400	0.993372	0.450000	0.329231	0.283333	0.450000
3	0.875700	0.778552	0.700000	0.690040	0.739755	0.700000
4	0.674600	0.579173	0.725000	0.725653	0.730385	0.725000
5	0.439200	0.557792	0.725000	0.719836	0.719006	0.725000
6	0.296400	0.614709	0.775000	0.778046	0.783333	0.775000
7	0.195000	0.762640	0.750000	0.752207	0.756579	0.750000
8	0.104200	0.766261	0.750000	0.748874	0.748887	0.750000
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-10
Configuration saved in ./roberta_results/fold_2/checkpoint-10/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-10/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-20
Configuration saved in ./roberta_results/fold_2/checkpoint-20/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-20/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-30
Configuration saved in ./roberta_results/fold_2/checkpoint-30/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-30/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-10] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-40
Configuration saved in ./roberta_results/fold_2/checkpoint-40/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-40/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-20] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-50
Configuration saved in ./roberta_results/fold_2/checkpoint-50/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-50/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-30] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-60
Configuration saved in ./roberta_results/fold_2/checkpoint-60/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-60/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-40] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-70
Configuration saved in ./roberta_results/fold_2/checkpoint-70/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-70/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-50] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-80
Configuration saved in ./roberta_results/fold_2/checkpoint-80/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-80/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-70] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_2/checkpoint-60 (score: 0.7780459770114942).
Deleting older checkpoint [roberta_results/fold_2/checkpoint-80] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Evaluating best model of Fold 2 on its validation set...
 [3/3 00:00]
Fold 2 Validation: F1=0.7780, Accuracy=0.7750
New best model found in Fold 2 with Val F1: 0.7780
\n==================================================\nTraining Fold 3/5\n==================================================
Map: 100%
 158/158 [00:00<00:00, 188.93 examples/s]
Map: 100%
 40/40 [00:00<00:00, 90.04 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.3",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 158
  Num Epochs = 8
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 2
  Total optimization steps = 80
  Number of trainable parameters = 124,647,939
Starting training for Fold 3...
 [80/80 03:58, Epoch 8/8]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.080800	1.046082	0.450000	0.279310	0.202500	0.450000
2	1.026800	0.914520	0.375000	0.288820	0.390000	0.375000
3	0.634100	0.656210	0.625000	0.632719	0.678205	0.625000
4	0.467200	0.674491	0.675000	0.660360	0.732609	0.675000
5	0.365100	0.729646	0.675000	0.680748	0.694561	0.675000
6	0.252500	0.814563	0.650000	0.648831	0.702273	0.650000
7	0.126700	1.047662	0.625000	0.619775	0.682391	0.625000
8	0.076600	1.122068	0.625000	0.627813	0.694664	0.625000
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-10
Configuration saved in ./roberta_results/fold_3/checkpoint-10/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-10/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-20
Configuration saved in ./roberta_results/fold_3/checkpoint-20/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-20/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-30
Configuration saved in ./roberta_results/fold_3/checkpoint-30/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-30/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-10] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-40
Configuration saved in ./roberta_results/fold_3/checkpoint-40/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-40/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-20] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-50
Configuration saved in ./roberta_results/fold_3/checkpoint-50/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-50/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-30] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-60
Configuration saved in ./roberta_results/fold_3/checkpoint-60/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-60/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-40] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-70
Configuration saved in ./roberta_results/fold_3/checkpoint-70/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-70/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-60] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-80
Configuration saved in ./roberta_results/fold_3/checkpoint-80/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-80/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-70] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_3/checkpoint-50 (score: 0.6807482359206498).
Deleting older checkpoint [roberta_results/fold_3/checkpoint-80] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 40
  Batch size = 16
Evaluating best model of Fold 3 on its validation set...
 [3/3 00:00]
Fold 3 Validation: F1=0.6807, Accuracy=0.6750
\n==================================================\nTraining Fold 4/5\n==================================================
Map: 100%
 159/159 [00:00<00:00, 184.44 examples/s]
Map: 100%
 39/39 [00:00<00:00, 90.56 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.3",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 159
  Num Epochs = 8
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 2
  Total optimization steps = 80
  Number of trainable parameters = 124,647,939
Starting training for Fold 4...
 [80/80 03:37, Epoch 8/8]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.095400	1.042356	0.461538	0.291498	0.213018	0.461538
2	1.041000	0.965845	0.461538	0.291498	0.213018	0.461538
3	0.854800	0.880440	0.538462	0.504903	0.662108	0.538462
4	0.556500	0.745034	0.564103	0.563370	0.599145	0.564103
5	0.389300	0.707433	0.615385	0.612698	0.679720	0.615385
6	0.324300	0.718010	0.743590	0.741237	0.740741	0.743590
7	0.132700	0.690520	0.769231	0.772650	0.778388	0.769231
8	0.064500	0.797814	0.794872	0.798779	0.807441	0.794872
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-10
Configuration saved in ./roberta_results/fold_4/checkpoint-10/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-10/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-20
Configuration saved in ./roberta_results/fold_4/checkpoint-20/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-20/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-30
Configuration saved in ./roberta_results/fold_4/checkpoint-30/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-30/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-10] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-40
Configuration saved in ./roberta_results/fold_4/checkpoint-40/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-40/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-20] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-50
Configuration saved in ./roberta_results/fold_4/checkpoint-50/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-50/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-30] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-60
Configuration saved in ./roberta_results/fold_4/checkpoint-60/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-60/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-40] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-70
Configuration saved in ./roberta_results/fold_4/checkpoint-70/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-70/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-50] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-80
Configuration saved in ./roberta_results/fold_4/checkpoint-80/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-80/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-60] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_4/checkpoint-80 (score: 0.7987789987789987).
Deleting older checkpoint [roberta_results/fold_4/checkpoint-70] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Evaluating best model of Fold 4 on its validation set...
 [3/3 00:00]
Fold 4 Validation: F1=0.7988, Accuracy=0.7949
New best model found in Fold 4 with Val F1: 0.7988
\n==================================================\nTraining Fold 5/5\n==================================================
Map: 100%
 159/159 [00:00<00:00, 184.08 examples/s]
Map: 100%
 39/39 [00:00<00:00, 127.21 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.3",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
Starting training for Fold 5...
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 159
  Num Epochs = 8
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 2
  Total optimization steps = 80
  Number of trainable parameters = 124,647,939
 [80/80 03:31, Epoch 8/8]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.098300	1.049692	0.435897	0.264652	0.190007	0.435897
2	1.037900	0.896522	0.487179	0.356505	0.405405	0.487179
3	0.721900	0.514764	0.794872	0.793404	0.801282	0.794872
4	0.560300	0.388946	0.846154	0.844510	0.848403	0.846154
5	0.336700	0.366144	0.820513	0.819753	0.820294	0.820513
6	0.279600	0.339825	0.871795	0.871252	0.872124	0.871795
7	0.166000	0.578535	0.794872	0.782284	0.822464	0.794872
8	0.175500	0.417800	0.820513	0.819753	0.820294	0.820513
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-10
Configuration saved in ./roberta_results/fold_5/checkpoint-10/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-10/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-20
Configuration saved in ./roberta_results/fold_5/checkpoint-20/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-20/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-30
Configuration saved in ./roberta_results/fold_5/checkpoint-30/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-30/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-10] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-40
Configuration saved in ./roberta_results/fold_5/checkpoint-40/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-40/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-20] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-50
Configuration saved in ./roberta_results/fold_5/checkpoint-50/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-50/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-30] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-60
Configuration saved in ./roberta_results/fold_5/checkpoint-60/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-60/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-40] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-70
Configuration saved in ./roberta_results/fold_5/checkpoint-70/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-70/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-50] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-80
Configuration saved in ./roberta_results/fold_5/checkpoint-80/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-80/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-70] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_5/checkpoint-60 (score: 0.8712522045855378).
Deleting older checkpoint [roberta_results/fold_5/checkpoint-80] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: text, Review Type, Paper Title, Review Text, Abstract. If text, Review Type, Paper Title, Review Text, Abstract are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 39
  Batch size = 16
Evaluating best model of Fold 5 on its validation set...
 [3/3 00:00]
Fold 5 Validation: F1=0.8713, Accuracy=0.8718
New best model found in Fold 5 with Val F1: 0.8713
\nAverage Cross-Validation Results:
  Average eval_loss: 0.6465
  Average eval_accuracy: 0.7733
  Average eval_f1: 0.7758
  Average eval_precision: 0.7850
  Average eval_recall: 0.7733
\nLoading the overall best model for final evaluation...
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.3",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Training history plot saved to roberta_training_history_cv_best_fold.png
\nPreparing test set for final evaluation...
Map: 100%
 50/50 [00:00<00:00, 144.37 examples/s]
Final Test Set Evaluation: 100%
 4/4 [00:01<00:00,  2.88it/s]
  Test Set: Accuracy=0.6800, F1=0.6830, Precision=0.7092, Recall=0.6800
Configuration saved in ./roberta_classifier_cv/config.json
Confusion matrix plot saved to roberta_confusion_matrix_cv_test.png
\nTest Set Performance Report (Overall Best Model):
              precision    recall  f1-score   support

AI-Generated       0.90      0.90      0.90        10
   Authentic       0.76      0.57      0.65        23
     Generic       0.52      0.71      0.60        17

    accuracy                           0.68        50
   macro avg       0.73      0.72      0.72        50
weighted avg       0.71      0.68      0.68        50

\nTest Set Performance Summary (Overall Best Model):
  AI-Generated: F1=0.9000, Precision=0.9000, Recall=0.9000
  Authentic: F1=0.6500, Precision=0.7647, Recall=0.5652
  Generic: F1=0.6000, Precision=0.5217, Recall=0.7059
\n  Overall Test: F1=0.6830, Accuracy=0.6800
\nSaving overall best model and tokenizer...
Model weights saved in ./roberta_classifier_cv/model.safetensors
tokenizer config file saved in ./roberta_classifier_cv/tokenizer_config.json
Special tokens file saved in ./roberta_classifier_cv/special_tokens_map.json
"""


"""
Starting RoBERTa model training with 5-fold cross-validation
Model: roberta-base, Max Length: 384, Batch Size: 8 (Train), 16 (Eval)
Gradient Accumulation Steps: 2
Learning Rate: 4e-05, Epochs per fold: 8
Loading data...
Successfully loaded data with cp1252 encoding.
Loaded 399 samples with 3 classes: ['AI-Generated' 'Authentic' 'Generic']
Train/CV set size: 319, Test set size: 80
loading file vocab.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/vocab.json
loading file merges.txt from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/merges.txt
loading file added_tokens.json from cache at None
loading file special_tokens_map.json from cache at None
loading file tokenizer_config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/tokenizer_config.json
loading file tokenizer.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/tokenizer.json
loading file chat_template.jinja from cache at None
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.4",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

Using device: cuda
\n==================================================\nTraining Fold 1/5\n==================================================
Map: 100%
 255/255 [00:01<00:00, 166.36 examples/s]
Map: 100%
 64/64 [00:00<00:00, 126.83 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.4",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 255
  Num Epochs = 8
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 2
  Total optimization steps = 128
  Number of trainable parameters = 124,647,939
Starting training for Fold 1...
 [128/128 02:23, Epoch 8/8]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.108700	1.026283	0.515625	0.350838	0.265869	0.515625
2	0.973100	0.960197	0.515625	0.350838	0.265869	0.515625
3	0.934200	0.944366	0.609375	0.502211	0.440582	0.609375
4	0.839000	0.968018	0.593750	0.553586	0.555563	0.593750
5	0.760300	0.916008	0.609375	0.622862	0.671067	0.609375
6	0.648100	0.829347	0.625000	0.577860	0.591846	0.625000
7	0.558700	0.745094	0.703125	0.672632	0.732422	0.703125
8	0.425200	0.792099	0.656250	0.663091	0.683101	0.656250
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-16
Configuration saved in ./roberta_results/fold_1/checkpoint-16/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-16/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-32
Configuration saved in ./roberta_results/fold_1/checkpoint-32/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-32/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-112] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-48
Configuration saved in ./roberta_results/fold_1/checkpoint-48/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-48/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-16] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-64
Configuration saved in ./roberta_results/fold_1/checkpoint-64/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-64/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-32] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-80
Configuration saved in ./roberta_results/fold_1/checkpoint-80/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-80/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-48] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-96
Configuration saved in ./roberta_results/fold_1/checkpoint-96/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-96/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-64] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-112
Configuration saved in ./roberta_results/fold_1/checkpoint-112/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-112/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-80] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-128
Configuration saved in ./roberta_results/fold_1/checkpoint-128/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-128/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-96] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_1/checkpoint-112 (score: 0.6726317663817664).
Deleting older checkpoint [roberta_results/fold_1/checkpoint-128] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Evaluating best model of Fold 1 on its validation set...
 [4/4 00:00]
Fold 1 Validation: F1=0.6726, Accuracy=0.7031
New best model found in Fold 1 with Val F1: 0.6726
\n==================================================\nTraining Fold 2/5\n==================================================
Map: 100%
 255/255 [00:01<00:00, 171.54 examples/s]
Map: 100%
 64/64 [00:00<00:00, 78.61 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.4",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 255
  Num Epochs = 8
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 2
  Total optimization steps = 128
  Number of trainable parameters = 124,647,939
Starting training for Fold 2...
 [112/128 01:59 < 00:17, 0.92 it/s, Epoch 7/8]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.086400	1.042168	0.500000	0.333333	0.250000	0.500000
2	1.022200	0.975628	0.500000	0.333333	0.250000	0.500000
3	0.928400	0.847935	0.578125	0.576206	0.596719	0.578125
4	0.729900	0.719978	0.656250	0.612397	0.651292	0.656250
5	0.674600	0.849169	0.593750	0.554330	0.556380	0.593750
6	0.546400	0.789369	0.625000	0.600879	0.598916	0.625000
7	0.471000	0.776360	0.609375	0.581355	0.622303	0.609375
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-16
Configuration saved in ./roberta_results/fold_2/checkpoint-16/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-16/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-32
Configuration saved in ./roberta_results/fold_2/checkpoint-32/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-32/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-64] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-48
Configuration saved in ./roberta_results/fold_2/checkpoint-48/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-48/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-16] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-64
Configuration saved in ./roberta_results/fold_2/checkpoint-64/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-64/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-32] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-80
Configuration saved in ./roberta_results/fold_2/checkpoint-80/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-80/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-48] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-96
Configuration saved in ./roberta_results/fold_2/checkpoint-96/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-96/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-80] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-112
Configuration saved in ./roberta_results/fold_2/checkpoint-112/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-112/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-96] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_2/checkpoint-64 (score: 0.6123973371152788).
Deleting older checkpoint [roberta_results/fold_2/checkpoint-112] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Evaluating best model of Fold 2 on its validation set...
 [4/4 00:00]
Fold 2 Validation: F1=0.6124, Accuracy=0.6562
\n==================================================\nTraining Fold 3/5\n==================================================
Map: 100%
 255/255 [00:01<00:00, 170.18 examples/s]
Map: 100%
 64/64 [00:00<00:00, 134.13 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.4",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 255
  Num Epochs = 8
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 2
  Total optimization steps = 128
  Number of trainable parameters = 124,647,939
Starting training for Fold 3...
 [128/128 02:09, Epoch 8/8]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.075800	1.038597	0.500000	0.333333	0.250000	0.500000
2	1.030400	1.011078	0.500000	0.333333	0.250000	0.500000
3	0.981800	0.916765	0.593750	0.483845	0.478987	0.593750
4	0.785500	0.705103	0.578125	0.500431	0.462813	0.578125
5	0.609500	0.742791	0.640625	0.643792	0.650029	0.640625
6	0.430100	0.830479	0.640625	0.640625	0.660833	0.640625
7	0.269900	1.032565	0.593750	0.591775	0.635345	0.593750
8	0.175900	1.079882	0.593750	0.594499	0.605497	0.593750
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-16
Configuration saved in ./roberta_results/fold_3/checkpoint-16/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-16/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-32
Configuration saved in ./roberta_results/fold_3/checkpoint-32/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-32/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-128] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-48
Configuration saved in ./roberta_results/fold_3/checkpoint-48/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-48/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-16] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-64
Configuration saved in ./roberta_results/fold_3/checkpoint-64/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-64/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-32] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-80
Configuration saved in ./roberta_results/fold_3/checkpoint-80/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-80/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-48] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-96
Configuration saved in ./roberta_results/fold_3/checkpoint-96/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-96/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-64] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-112
Configuration saved in ./roberta_results/fold_3/checkpoint-112/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-112/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-96] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-128
Configuration saved in ./roberta_results/fold_3/checkpoint-128/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-128/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-112] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_3/checkpoint-80 (score: 0.6437924830067973).
Deleting older checkpoint [roberta_results/fold_3/checkpoint-128] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Evaluating best model of Fold 3 on its validation set...
 [4/4 00:00]
Fold 3 Validation: F1=0.6438, Accuracy=0.6406
\n==================================================\nTraining Fold 4/5\n==================================================
Map: 100%
 255/255 [00:01<00:00, 169.68 examples/s]
Map: 100%
 64/64 [00:00<00:00, 143.81 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.4",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 255
  Num Epochs = 8
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 2
  Total optimization steps = 128
  Number of trainable parameters = 124,647,939
Starting training for Fold 4...
 [128/128 02:10, Epoch 8/8]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.074700	1.035461	0.500000	0.333333	0.250000	0.500000
2	1.047500	1.019745	0.500000	0.333333	0.250000	0.500000
3	1.024700	0.971546	0.500000	0.333333	0.250000	0.500000
4	0.908900	0.825207	0.609375	0.501738	0.483827	0.609375
5	0.703200	0.785315	0.515625	0.503465	0.639805	0.515625
6	0.504200	0.849427	0.578125	0.571343	0.644553	0.578125
7	0.412600	0.790448	0.703125	0.698351	0.700372	0.703125
8	0.297900	0.843606	0.656250	0.657790	0.668909	0.656250
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-16
Configuration saved in ./roberta_results/fold_4/checkpoint-16/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-16/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-32
Configuration saved in ./roberta_results/fold_4/checkpoint-32/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-32/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-80] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-48
Configuration saved in ./roberta_results/fold_4/checkpoint-48/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-48/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-32] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-64
Configuration saved in ./roberta_results/fold_4/checkpoint-64/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-64/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-16] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-80
Configuration saved in ./roberta_results/fold_4/checkpoint-80/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-80/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-48] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-96
Configuration saved in ./roberta_results/fold_4/checkpoint-96/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-96/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-64] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-112
Configuration saved in ./roberta_results/fold_4/checkpoint-112/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-112/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-80] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-128
Configuration saved in ./roberta_results/fold_4/checkpoint-128/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-128/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-96] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_4/checkpoint-112 (score: 0.6983505674243163).
Deleting older checkpoint [roberta_results/fold_4/checkpoint-128] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 64
  Batch size = 16
Evaluating best model of Fold 4 on its validation set...
 [4/4 00:00]
Fold 4 Validation: F1=0.6984, Accuracy=0.7031
New best model found in Fold 4 with Val F1: 0.6984
\n==================================================\nTraining Fold 5/5\n==================================================
Map: 100%
 256/256 [00:01<00:00, 167.62 examples/s]
Map: 100%
 63/63 [00:00<00:00, 127.83 examples/s]
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.4",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 256
  Num Epochs = 8
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 16
  Gradient Accumulation steps = 2
  Total optimization steps = 128
  Number of trainable parameters = 124,647,939
Starting training for Fold 5...
 [128/128 02:28, Epoch 8/8]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.085700	1.035249	0.507937	0.342189	0.257999	0.507937
2	1.044200	0.968525	0.507937	0.342189	0.257999	0.507937
3	0.966400	0.963080	0.507937	0.342189	0.257999	0.507937
4	0.852100	0.758161	0.587302	0.530945	0.535053	0.587302
5	0.650900	0.816545	0.523810	0.520427	0.520377	0.523810
6	0.541100	1.108818	0.523810	0.495712	0.641876	0.523810
7	0.619600	0.847349	0.571429	0.565494	0.614278	0.571429
8	0.368800	0.905620	0.571429	0.569866	0.568947	0.571429
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 63
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-16
Configuration saved in ./roberta_results/fold_5/checkpoint-16/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-16/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 63
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-32
Configuration saved in ./roberta_results/fold_5/checkpoint-32/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-32/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 63
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-48
Configuration saved in ./roberta_results/fold_5/checkpoint-48/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-48/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-32] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 63
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-64
Configuration saved in ./roberta_results/fold_5/checkpoint-64/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-64/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-16] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 63
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-80
Configuration saved in ./roberta_results/fold_5/checkpoint-80/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-80/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-48] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 63
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-96
Configuration saved in ./roberta_results/fold_5/checkpoint-96/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-96/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-80] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 63
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-112
Configuration saved in ./roberta_results/fold_5/checkpoint-112/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-112/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-64] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 63
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-128
Configuration saved in ./roberta_results/fold_5/checkpoint-128/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-128/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-96] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_5/checkpoint-128 (score: 0.5698664651045603).
Deleting older checkpoint [roberta_results/fold_5/checkpoint-112] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Paper Title, text, Review Type, Abstract, Review Text. If Paper Title, text, Review Type, Abstract, Review Text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 63
  Batch size = 16
Evaluating best model of Fold 5 on its validation set...
 [4/4 00:00]
Fold 5 Validation: F1=0.5699, Accuracy=0.5714
\nAverage Cross-Validation Results:
  Average eval_loss: 0.7808
  Average eval_accuracy: 0.6549
  Average eval_f1: 0.6394
  Average eval_precision: 0.6606
  Average eval_recall: 0.6549
\nLoading the overall best model for final evaluation...
loading configuration file config.json from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/config.json
Model config RobertaConfig {
  "architectures": [
    "RobertaForMaskedLM"
  ],
  "attention_probs_dropout_prob": 0.1,
  "bos_token_id": 0,
  "classifier_dropout": null,
  "eos_token_id": 2,
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768,
  "id2label": {
    "0": "AI-Generated",
    "1": "Authentic",
    "2": "Generic"
  },
  "initializer_range": 0.02,
  "intermediate_size": 3072,
  "label2id": {
    "AI-Generated": 0,
    "Authentic": 1,
    "Generic": 2
  },
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 514,
  "model_type": "roberta",
  "num_attention_heads": 12,
  "num_hidden_layers": 12,
  "pad_token_id": 1,
  "position_embedding_type": "absolute",
  "transformers_version": "4.52.4",
  "type_vocab_size": 1,
  "use_cache": true,
  "vocab_size": 50265
}

loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Training history plot saved to roberta_training_history_cv_best_fold.png
\nPreparing test set for final evaluation...
Map: 100%
 80/80 [00:00<00:00, 125.68 examples/s]
Final Test Set Evaluation: 100%
 5/5 [00:01<00:00,  2.87it/s]
  Test Set: Accuracy=0.7125, F1=0.7059, Precision=0.7117, Recall=0.7125
Configuration saved in ./roberta_classifier_cv/config.json
Confusion matrix plot saved to roberta_confusion_matrix_cv_test.png
\nTest Set Performance Report (Overall Best Model):
              precision    recall  f1-score   support

AI-Generated       0.87      0.81      0.84        16
   Authentic       0.69      0.80      0.74        41
     Generic       0.65      0.48      0.55        23

    accuracy                           0.71        80
   macro avg       0.73      0.70      0.71        80
weighted avg       0.71      0.71      0.71        80

\nTest Set Performance Summary (Overall Best Model):
  AI-Generated: F1=0.8387, Precision=0.8667, Recall=0.8125
  Authentic: F1=0.7416, Precision=0.6875, Recall=0.8049
  Generic: F1=0.5500, Precision=0.6471, Recall=0.4783
\n  Overall Test: F1=0.7059, Accuracy=0.7125
\nSaving overall best model and tokenizer...
Model weights saved in ./roberta_classifier_cv/model.safetensors
tokenizer config file saved in ./roberta_classifier_cv/tokenizer_config.json
Special tokens file saved in ./roberta_classifier_cv/special_tokens_map.json
"""
