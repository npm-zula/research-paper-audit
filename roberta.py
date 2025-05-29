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
MAX_LENGTH = 512
BATCH_SIZE = 8  # Per device train batch size for Trainer
EVAL_BATCH_SIZE = BATCH_SIZE * 2  # Per device eval batch size
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 3e-5
NUM_EPOCHS_PER_FOLD = 15  # Renamed for clarity
WEIGHT_DECAY = 0.01
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
    df_full, class_names, label_encoder = load_data('corpus.csv')
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
            evaluation_strategy="epoch",
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

    # The old evaluate_model and predict functions are now either integrated or can be called separately
    # with the saved model. The main evaluation is part of train_model_with_cv.
    # The old main block's test set evaluation is now handled within train_model_with_cv.


"""
Using device: cuda
\n==================================================\nTraining Fold 1/5\n==================================================
Map: 100%
 220/220 [00:02<00:00, 81.67 examples/s]
Map: 100%
 56/56 [00:00<00:00, 140.96 examples/s]
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

Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
WARNING:huggingface_hub.file_download:Xet Storage is enabled for this repo, but the 'hf_xet' package is not installed. Falling back to regular HTTP download. For better performance, install the package with: `pip install huggingface_hub[hf_xet]` or `pip install hf_xet`
model.safetensors: 100%
 499M/499M [00:02<00:00, 269MB/s]
loading weights file model.safetensors from cache at /root/.cache/huggingface/hub/models--roberta-base/snapshots/e2da8e2f811d1448a5b465c236feacd80ffbac7b/model.safetensors
Some weights of the model checkpoint at roberta-base were not used when initializing RobertaForSequenceClassification: ['lm_head.bias', 'lm_head.dense.bias', 'lm_head.dense.weight', 'lm_head.layer_norm.bias', 'lm_head.layer_norm.weight']
- This IS expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model trained on another task or with another architecture (e.g. initializing a BertForSequenceClassification model from a BertForPreTraining model).
- This IS NOT expected if you are initializing RobertaForSequenceClassification from the checkpoint of a model that you expect to be exactly identical (initializing a BertForSequenceClassification model from a BertForSequenceClassification model).
Some weights of RobertaForSequenceClassification were not initialized from the model checkpoint at roberta-base and are newly initialized: ['classifier.dense.bias', 'classifier.dense.weight', 'classifier.out_proj.bias', 'classifier.out_proj.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
PyTorch: setting up devices
Using auto half precision backend
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 220
  Num Epochs = 15
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 4
  Total optimization steps = 105
  Number of trainable parameters = 124,647,939
Starting training for Fold 1...
 [ 63/105 04:36 < 03:10, 0.22 it/s, Epoch 9/15]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.120400	1.096191	0.303571	0.141389	0.092156	0.303571
2	1.032900	1.015621	0.517857	0.353361	0.268176	0.517857
3	0.968600	0.935922	0.517857	0.353361	0.268176	0.517857
4	0.889800	0.888541	0.517857	0.353361	0.268176	0.517857
5	0.816400	0.839994	0.553571	0.561238	0.573214	0.553571
6	0.716300	0.845827	0.642857	0.637511	0.634037	0.642857
7	0.621900	0.816930	0.571429	0.577555	0.589932	0.571429
8	0.530400	0.837613	0.571429	0.579972	0.622115	0.571429
9	0.423900	0.950415	0.553571	0.562075	0.601656	0.553571
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 56
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-7
Configuration saved in ./roberta_results/fold_1/checkpoint-7/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-7/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 56
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-14
Configuration saved in ./roberta_results/fold_1/checkpoint-14/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-14/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 56
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-21
Configuration saved in ./roberta_results/fold_1/checkpoint-21/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-21/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-7] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 56
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-28
Configuration saved in ./roberta_results/fold_1/checkpoint-28/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-28/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-21] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 56
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-35
Configuration saved in ./roberta_results/fold_1/checkpoint-35/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-35/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-14] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 56
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-42
Configuration saved in ./roberta_results/fold_1/checkpoint-42/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-42/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-28] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 56
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-49
Configuration saved in ./roberta_results/fold_1/checkpoint-49/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-49/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-35] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 56
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-56
Configuration saved in ./roberta_results/fold_1/checkpoint-56/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-56/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-49] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 56
  Batch size = 16
Saving model checkpoint to ./roberta_results/fold_1/checkpoint-63
Configuration saved in ./roberta_results/fold_1/checkpoint-63/config.json
Model weights saved in ./roberta_results/fold_1/checkpoint-63/model.safetensors
Deleting older checkpoint [roberta_results/fold_1/checkpoint-56] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_1/checkpoint-42 (score: 0.6375109895653176).
Deleting older checkpoint [roberta_results/fold_1/checkpoint-63] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 56
  Batch size = 16
Evaluating best model of Fold 1 on its validation set...
 [4/4 00:00]
Fold 1 Validation: F1=0.6375, Accuracy=0.6429
New best model found in Fold 1 with Val F1: 0.6375
\n==================================================\nTraining Fold 2/5\n==================================================
Map: 100%
 221/221 [00:01<00:00, 161.10 examples/s]
Map: 100%
 55/55 [00:00<00:00, 123.35 examples/s]
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
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 221
  Num Epochs = 15
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 4
  Total optimization steps = 105
  Number of trainable parameters = 124,647,939
Starting training for Fold 2...
 [ 28/105 02:03 < 06:06, 0.21 it/s, Epoch 4/15]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.113800	1.073224	0.418182	0.398658	0.431039	0.418182
2	1.041500	0.977526	0.527273	0.364069	0.278017	0.527273
3	0.959700	0.873366	0.527273	0.364069	0.278017	0.527273
4	0.861600	0.823561	0.527273	0.364069	0.278017	0.527273
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-7
Configuration saved in ./roberta_results/fold_2/checkpoint-7/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-7/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-14
Configuration saved in ./roberta_results/fold_2/checkpoint-14/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-14/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-21
Configuration saved in ./roberta_results/fold_2/checkpoint-21/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-21/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-14] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_2/checkpoint-28
Configuration saved in ./roberta_results/fold_2/checkpoint-28/config.json
Model weights saved in ./roberta_results/fold_2/checkpoint-28/model.safetensors
Deleting older checkpoint [roberta_results/fold_2/checkpoint-21] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_2/checkpoint-7 (score: 0.3986584843727701).
Deleting older checkpoint [roberta_results/fold_2/checkpoint-28] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
Evaluating best model of Fold 2 on its validation set...
 [4/4 00:00]
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Fold 2 Validation: F1=0.3987, Accuracy=0.4182
\n==================================================\nTraining Fold 3/5\n==================================================
Map: 100%
 221/221 [00:01<00:00, 164.29 examples/s]
Map: 100%
 55/55 [00:00<00:00, 129.96 examples/s]
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
Starting training for Fold 3...
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 221
  Num Epochs = 15
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 4
  Total optimization steps = 105
  Number of trainable parameters = 124,647,939
 [ 28/105 02:23 < 07:06, 0.18 it/s, Epoch 4/15]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.098600	1.052441	0.527273	0.364069	0.278017	0.527273
2	1.030400	0.961958	0.527273	0.364069	0.278017	0.527273
3	0.965800	0.938730	0.527273	0.364069	0.278017	0.527273
4	0.912800	0.871032	0.527273	0.364069	0.278017	0.527273
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-7
Configuration saved in ./roberta_results/fold_3/checkpoint-7/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-7/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-14
Configuration saved in ./roberta_results/fold_3/checkpoint-14/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-14/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-21
Configuration saved in ./roberta_results/fold_3/checkpoint-21/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-21/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-14] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_3/checkpoint-28
Configuration saved in ./roberta_results/fold_3/checkpoint-28/config.json
Model weights saved in ./roberta_results/fold_3/checkpoint-28/model.safetensors
Deleting older checkpoint [roberta_results/fold_3/checkpoint-21] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_3/checkpoint-7 (score: 0.36406926406926404).
Deleting older checkpoint [roberta_results/fold_3/checkpoint-28] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
Evaluating best model of Fold 3 on its validation set...
 [4/4 00:00]
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Fold 3 Validation: F1=0.3641, Accuracy=0.5273
\n==================================================\nTraining Fold 4/5\n==================================================
Map: 100%
 221/221 [00:01<00:00, 165.38 examples/s]
Map: 100%
 55/55 [00:00<00:00, 123.57 examples/s]
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
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 221
  Num Epochs = 15
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 4
  Total optimization steps = 105
  Number of trainable parameters = 124,647,939
Starting training for Fold 4...
 [ 28/105 02:10 < 06:25, 0.20 it/s, Epoch 4/15]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.084800	1.053711	0.509091	0.343483	0.259174	0.509091
2	1.009700	0.983350	0.509091	0.343483	0.259174	0.509091
3	0.968200	0.924796	0.509091	0.343483	0.259174	0.509091
4	0.905100	0.889329	0.509091	0.343483	0.259174	0.509091
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-7
Configuration saved in ./roberta_results/fold_4/checkpoint-7/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-7/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-14
Configuration saved in ./roberta_results/fold_4/checkpoint-14/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-14/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-21
Configuration saved in ./roberta_results/fold_4/checkpoint-21/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-21/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-14] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_4/checkpoint-28
Configuration saved in ./roberta_results/fold_4/checkpoint-28/config.json
Model weights saved in ./roberta_results/fold_4/checkpoint-28/model.safetensors
Deleting older checkpoint [roberta_results/fold_4/checkpoint-21] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_4/checkpoint-7 (score: 0.3434830230010953).
Deleting older checkpoint [roberta_results/fold_4/checkpoint-28] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
Evaluating best model of Fold 4 on its validation set...
 [4/4 00:00]
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Fold 4 Validation: F1=0.3435, Accuracy=0.5091
\n==================================================\nTraining Fold 5/5\n==================================================
Map: 100%
 221/221 [00:01<00:00, 159.77 examples/s]
Map: 100%
 55/55 [00:00<00:00, 127.69 examples/s]
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
The following columns in the Training set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.
***** Running training *****
  Num examples = 221
  Num Epochs = 15
  Instantaneous batch size per device = 8
  Total train batch size (w. parallel, distributed & accumulation) = 32
  Gradient Accumulation steps = 4
  Total optimization steps = 105
  Number of trainable parameters = 124,647,939
 [ 28/105 01:43 < 05:05, 0.25 it/s, Epoch 4/15]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.092000	1.051012	0.509091	0.343483	0.259174	0.509091
2	1.005100	0.982590	0.509091	0.343483	0.259174	0.509091
3	0.966800	0.954985	0.509091	0.343483	0.259174	0.509091
4	0.914100	0.863259	0.509091	0.343483	0.259174	0.509091
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-7
Configuration saved in ./roberta_results/fold_5/checkpoint-7/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-7/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-14
Configuration saved in ./roberta_results/fold_5/checkpoint-14/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-14/model.safetensors
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-21
Configuration saved in ./roberta_results/fold_5/checkpoint-21/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-21/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-14] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Saving model checkpoint to ./roberta_results/fold_5/checkpoint-28
Configuration saved in ./roberta_results/fold_5/checkpoint-28/config.json
Model weights saved in ./roberta_results/fold_5/checkpoint-28/model.safetensors
Deleting older checkpoint [roberta_results/fold_5/checkpoint-21] due to args.save_total_limit


Training completed. Do not forget to share your model on huggingface.co/models =)


Loading best model from ./roberta_results/fold_5/checkpoint-7 (score: 0.3434830230010953).
Deleting older checkpoint [roberta_results/fold_5/checkpoint-28] due to args.save_total_limit
The following columns in the Evaluation set don't have a corresponding argument in `RobertaForSequenceClassification.forward` and have been ignored: Abstract, Paper Title, Review Type, Review Text, text. If Abstract, Paper Title, Review Type, Review Text, text are not expected by `RobertaForSequenceClassification.forward`,  you can safely ignore this message.

***** Running Evaluation *****
  Num examples = 55
  Batch size = 16
Evaluating best model of Fold 5 on its validation set...
 [4/4 00:00]
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Fold 5 Validation: F1=0.3435, Accuracy=0.5091
\nAverage Cross-Validation Results:
  Average eval_loss: 1.0152
  Average eval_accuracy: 0.5213
  Average eval_f1: 0.4174
  Average eval_precision: 0.3723
  Average eval_recall: 0.5213
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
 70/70 [00:00<00:00, 135.67 examples/s]
Final Test Set Evaluation: 100%
 5/5 [00:02<00:00,  2.51it/s]
  Test Set: Accuracy=0.6143, F1=0.5763, Precision=0.5682, Recall=0.6143
Configuration saved in ./roberta_classifier_cv/config.json
Confusion matrix plot saved to roberta_confusion_matrix_cv_test.png
\nTest Set Performance Report (Overall Best Model):
              precision    recall  f1-score   support

AI-Generated       0.71      0.83      0.77        12
   Authentic       0.64      0.81      0.72        36
     Generic       0.36      0.18      0.24        22

    accuracy                           0.61        70
   macro avg       0.57      0.61      0.58        70
weighted avg       0.57      0.61      0.58        70

\nTest Set Performance Summary (Overall Best Model):
  AI-Generated: F1=0.7692, Precision=0.7143, Recall=0.8333
  Authentic: F1=0.7160, Precision=0.6444, Recall=0.8056
  Generic: F1=0.2424, Precision=0.3636, Recall=0.1818
\n  Overall Test: F1=0.5763, Accuracy=0.6143
"""
