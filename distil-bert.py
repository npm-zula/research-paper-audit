import pandas as pd
import numpy as np
# Added StratifiedKFold
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
# import evaluate # evaluate module is not explicitly used in the final CV logic, can be removed if not needed elsewhere
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm.auto import tqdm
import copy  # Added copy

# Configuration
MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 384
BATCH_SIZE = 16  # Batch size for training within a fold
EVAL_BATCH_SIZE = BATCH_SIZE * 2  # Can be larger for evaluation
LEARNING_RATE = 3e-5
NUM_EPOCHS_PER_FOLD = 6  # Renamed for clarity
CROSS_VAL_FOLDS = 5  # Number of cross-validation folds
SEED = 42  # Combined SEED constant

# Set random seed for reproducibility
torch.manual_seed(SEED)
np.random.seed(SEED)


def plot_confusion_matrix(y_true, y_pred, class_names, filename='distilbert_confusion_matrix.png'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_training_history(history, filename='distilbert_training_history.png'):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['eval_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 3, 2)
    plt.plot(history['eval_accuracy'], label='Accuracy')
    plt.plot(history['eval_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')

    plt.subplot(1, 3, 3)
    plt.plot(history['eval_precision'], label='Precision')
    plt.plot(history['eval_recall'], label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision and Recall')

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

# Load and preprocess data


def load_data(file_path):
    # Try multiple encodings
    encodings = ['cp1252', 'latin-1', 'utf-8']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")

    if df is None:
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')

    # Handle missing values and clean text
    df = df.fillna({'Paper Title': '', 'Abstract': '', 'Review Text': ''})

    # Clean text
    for col in ['Paper Title', 'Abstract', 'Review Text']:
        df[col] = df[col].apply(lambda x: str(x).encode(
            'ascii', 'ignore').decode('ascii'))

    # Combine text features
    df['text'] = (
        "Title: " + df['Paper Title'] +
        " Abstract: " + df['Abstract'] +
        " Review: " + df['Review Text']
    )

    # Drop rows with missing values
    df = df.dropna(subset=['text', 'Review Type'])

    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Review Type'])

    # Return encoder as well
    return df[['text', 'label']], label_encoder.classes_, label_encoder

# Compute metrics for evaluation


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


def main():
    print(
        f"Starting DistilBERT model training with {CROSS_VAL_FOLDS}-fold cross-validation.")
    print(
        f"Model: {MODEL_NAME}, Max Length: {MAX_LENGTH}, Batch Size: {BATCH_SIZE}")
    print(
        f"Learning Rate: {LEARNING_RATE}, Epochs per fold: {NUM_EPOCHS_PER_FOLD}")

    # Load your dataset
    print("Loading and preprocessing data...")
    df_full, class_names, label_encoder = load_data("corpus-final.csv")
    print(
        f"Loaded {len(df_full)} samples with {len(class_names)} classes: {class_names}")

    # Split data into Train/CV and Test sets
    train_cv_df, test_df = train_test_split(
        df_full, test_size=0.2, stratify=df_full['label'], random_state=SEED
    )

    # Initialize tokenizer (once is fine)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

    # Setup cross-validation
    skf = StratifiedKFold(n_splits=CROSS_VAL_FOLDS,
                          shuffle=True, random_state=SEED)

    fold_results_summary = []
    best_overall_val_f1 = 0.0
    best_model_state_dict = None
    best_fold_log_history = None

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(train_cv_df, train_cv_df['label'])):
        print(f"\\n{'='*50}\\nTraining Fold {fold+1}/{CROSS_VAL_FOLDS}\\n{'='*50}")

        current_train_df = train_cv_df.iloc[train_idx]
        current_val_df = train_cv_df.iloc[val_idx]

        # Create Hugging Face datasets for the current fold
        train_dataset_fold = Dataset.from_pandas(current_train_df)
        val_dataset_fold = Dataset.from_pandas(current_val_df)

        # Tokenize datasets for the current fold
        print(f"Tokenizing datasets for Fold {fold+1}...")
        train_dataset_fold = train_dataset_fold.map(
            tokenize_function, batched=True)
        val_dataset_fold = val_dataset_fold.map(
            tokenize_function, batched=True)

        # Convert to torch format
        train_dataset_fold.set_format(
            type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset_fold.set_format(type='torch', columns=[
                                    'input_ids', 'attention_mask', 'label'])

        # Load model for the current fold
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(class_names),
            id2label={i: label for i, label in enumerate(class_names)},
            label2id={label: i for i, label in enumerate(class_names)}
        )
        # model.to(device) # Trainer handles device placement

        # Training arguments for the current fold
        training_args_fold = TrainingArguments(
            output_dir=f"./distilbert_results/fold_{fold+1}",
            eval_strategy="epoch",
            save_strategy="epoch",  # Saves checkpoints every epoch
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=EVAL_BATCH_SIZE,
            num_train_epochs=NUM_EPOCHS_PER_FOLD,
            weight_decay=0.01,
            load_best_model_at_end=True,  # Loads best model of this fold at the end of training
            metric_for_best_model="f1",
            greater_is_better=True,
            fp16=torch.cuda.is_available(),
            logging_dir=f"./logs/fold_{fold+1}",
            logging_strategy="epoch",  # Log at the end of each epoch
            report_to="tensorboard",
            save_total_limit=1  # Only keep the best checkpoint
        )

        # Initialize Trainer for the current fold
        trainer_fold = Trainer(
            model=model,
            args=training_args_fold,
            train_dataset=train_dataset_fold,
            eval_dataset=val_dataset_fold,
            compute_metrics=compute_metrics,
            # Increased patience
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )

        # Train model for the current fold
        print(f"Starting training for Fold {fold+1}...")
        trainer_fold.train()

        # Evaluate the best model of this fold on its validation set
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
            # Save state of the best model from this fold
            best_model_state_dict = copy.deepcopy(
                trainer_fold.model.state_dict())
            best_fold_log_history = copy.deepcopy(
                trainer_fold.state.log_history)
            print(
                f"New best model found in Fold {fold+1} with Val F1: {best_overall_val_f1:.4f}")

    # After all folds
    print("\\nAverage Cross-Validation Results:")
    avg_cv_metrics = pd.DataFrame(fold_results_summary).mean().to_dict()
    for metric, value in avg_cv_metrics.items():
        if metric != 'fold':
            # Match trainer's naming
            print(f"  Average {metric.replace('val_', 'eval_')}: {value:.4f}")

    # Load the overall best model
    print("\\nLoading the overall best model for final evaluation...")
    overall_best_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label={i: label for i, label in enumerate(class_names)},
        label2id={label: i for i, label in enumerate(class_names)}
    )
    overall_best_model.load_state_dict(best_model_state_dict)
    # Ensure model is on the correct device for final eval
    overall_best_model.to(device)

    # Plot training history of the best fold
    if best_fold_log_history:
        history_to_plot = {
            'train_loss': [], 'eval_loss': [], 'eval_accuracy': [],
            'eval_f1': [], 'eval_precision': [], 'eval_recall': []
        }
        # Extract relevant metrics from the best fold's log history
        # The exact structure of log_history might vary slightly based on Trainer version and logging steps
        # This part assumes epoch-level logging for both train and eval.
        temp_train_loss = {}  # Use dict to get last train loss for an epoch
        for log_entry in best_fold_log_history:
            if 'loss' in log_entry and 'eval_loss' not in log_entry:  # Training loss
                # Trainer might log multiple training steps per epoch, we want the one at epoch end or average
                # For simplicity, if logging_strategy is 'epoch', 'loss' should be epoch training loss.
                epoch_num = int(log_entry.get('epoch', 0))
                if epoch_num > 0:  # Ensure epoch number is valid
                    temp_train_loss[epoch_num] = log_entry['loss']

            elif 'eval_loss' in log_entry:  # Evaluation metrics
                epoch_num = int(log_entry.get('epoch', 0))
                if epoch_num > 0:  # Ensure epoch number is valid
                    if epoch_num in temp_train_loss:  # Ensure corresponding train loss exists
                        history_to_plot['train_loss'].append(
                            temp_train_loss[epoch_num])
                        history_to_plot['eval_loss'].append(
                            log_entry['eval_loss'])
                        history_to_plot['eval_accuracy'].append(
                            log_entry['eval_accuracy'])
                        history_to_plot['eval_f1'].append(log_entry['eval_f1'])
                        history_to_plot['eval_precision'].append(
                            log_entry['eval_precision'])
                        history_to_plot['eval_recall'].append(
                            log_entry['eval_recall'])

        if history_to_plot['eval_f1']:  # Check if we have data to plot
            plot_training_history(
                history_to_plot, filename='distilbert_training_history_cv_best_fold.png')
        else:
            print("Could not extract sufficient history for plotting the best fold.")

    # Prepare test dataset for final evaluation
    print("\\nPreparing test set for final evaluation...")
    test_dataset_final = Dataset.from_pandas(test_df)
    test_dataset_final = test_dataset_final.map(
        tokenize_function, batched=True)
    test_dataset_final.set_format(
        type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Final evaluation on the Test set using the overall best model
    print("Evaluating overall best model on the Test set...")

    # Need a Trainer instance for evaluation or manual evaluation loop
    # Using a manual loop for clarity here, similar to other scripts
    overall_best_model.eval()
    all_predictions_test = []
    all_labels_test = []

    test_loader_final = torch.utils.data.DataLoader(
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

    # Calculate final test metrics
    test_accuracy = np.mean(np.array(all_predictions_test)
                            == np.array(all_labels_test))
    test_precision, test_recall, test_f1, _ = precision_recall_fscore_support(
        all_labels_test, all_predictions_test, average='weighted', zero_division=0
    )
    print(
        f"  Test Set: Accuracy={test_accuracy:.4f}, F1={test_f1:.4f}, Precision={test_precision:.4f}, Recall={test_recall:.4f}")

    # Plot confusion matrix for the test set
    plot_confusion_matrix(all_labels_test, all_predictions_test,
                          class_names, filename='distilbert_confusion_matrix_cv_test.png')

    # Generate and print classification report for the test set
    report_dict_test = classification_report(all_labels_test, all_predictions_test,
                                             target_names=class_names, output_dict=True, zero_division=0)
    report_str_test = classification_report(all_labels_test, all_predictions_test,
                                            target_names=class_names, zero_division=0)

    print("\\nTest Set Performance Report (Overall Best Model):")
    print(report_str_test)

    print("\\nTest Set Performance Summary (Overall Best Model):")
    for cls in class_names:
        print(f"  {cls}: F1={report_dict_test[cls]['f1-score']:.4f}, "
              f"Precision={report_dict_test[cls]['precision']:.4f}, "
              f"Recall={report_dict_test[cls]['recall']:.4f}")

    print(f"\\n  Overall Test: F1={report_dict_test['weighted avg']['f1-score']:.4f}, "
          f"Accuracy={report_dict_test['accuracy']:.4f}")

    # Save the overall best model and tokenizer
    print("\\nSaving overall best model and tokenizer...")
    overall_best_model.save_pretrained("./distilbert_classifier_cv")
    tokenizer.save_pretrained("./distilbert_classifier_cv")

    print("\\nTraining, cross-validation, and final test evaluation complete.")
    print(f"Best model and tokenizer saved to ./distilbert_classifier_cv")
    print("Visualization files saved: distilbert_training_history_cv_best_fold.png, distilbert_confusion_matrix_cv_test.png")


if __name__ == "__main__":
    main()


"""
Starting DistilBERT model training with 5-fold cross-validation.
Model: distilbert-base-uncased, Max Length: 384, Batch Size: 16
Learning Rate: 3e-05, Epochs per fold: 6
Loading and preprocessing data...
Loaded 248 samples with 3 classes: ['AI-Generated' 'Authentic' 'Generic']
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
Using device: cuda
\n==================================================\nTraining Fold 1/5\n==================================================
Tokenizing datasets for Fold 1...
Map: 100%
 158/158 [00:00<00:00, 350.88 examples/s]
Map: 100%
 40/40 [00:00<00:00, 230.23 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 1...
 [60/60 00:38, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.031200	0.962915	0.450000	0.279310	0.202500	0.450000
2	0.877200	0.829504	0.600000	0.554472	0.573810	0.600000
3	0.728600	0.719617	0.700000	0.700000	0.700000	0.700000
4	0.598700	0.656397	0.725000	0.725813	0.727647	0.725000
5	0.521800	0.620984	0.725000	0.725813	0.727647	0.725000
6	0.469800	0.612622	0.675000	0.673373	0.672672	0.675000
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 1 on its validation set...
 [2/2 00:00]
Fold 1 Validation: F1=0.7258, Accuracy=0.7250
New best model found in Fold 1 with Val F1: 0.7258
\n==================================================\nTraining Fold 2/5\n==================================================
Tokenizing datasets for Fold 2...
Map: 100%
 158/158 [00:00<00:00, 617.22 examples/s]
Map: 100%
 40/40 [00:00<00:00, 467.23 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 2...
 [60/60 00:46, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.064200	0.983545	0.450000	0.279310	0.202500	0.450000
2	0.930100	0.871521	0.450000	0.325521	0.288571	0.450000
3	0.790500	0.745825	0.675000	0.630526	0.676667	0.675000
4	0.687300	0.676562	0.650000	0.627273	0.637500	0.650000
5	0.606300	0.634827	0.650000	0.627273	0.637500	0.650000
6	0.563400	0.622705	0.700000	0.690000	0.696364	0.700000
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 2 on its validation set...
 [2/2 00:00]
Fold 2 Validation: F1=0.6900, Accuracy=0.7000
\n==================================================\nTraining Fold 3/5\n==================================================
Tokenizing datasets for Fold 3...
Map: 100%
 158/158 [00:00<00:00, 432.50 examples/s]
Map: 100%
 40/40 [00:00<00:00, 358.15 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 3...
 [60/60 00:45, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.058500	0.974829	0.450000	0.279310	0.202500	0.450000
2	0.924800	0.914478	0.550000	0.433333	0.425000	0.550000
3	0.827900	0.790649	0.725000	0.701061	0.744231	0.725000
4	0.719700	0.728558	0.725000	0.701550	0.738000	0.725000
5	0.637100	0.703662	0.725000	0.719077	0.722727	0.725000
6	0.589900	0.691705	0.725000	0.719077	0.722727	0.725000
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 3 on its validation set...
 [2/2 00:00]
Fold 3 Validation: F1=0.7191, Accuracy=0.7250
\n==================================================\nTraining Fold 4/5\n==================================================
Tokenizing datasets for Fold 4...
Map: 100%
 159/159 [00:00<00:00, 594.10 examples/s]
Map: 100%
 39/39 [00:00<00:00, 476.25 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 4...
 [60/60 00:47, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.055900	0.978465	0.461538	0.291498	0.213018	0.461538
2	0.909300	0.872020	0.615385	0.531202	0.790210	0.615385
3	0.770400	0.762188	0.641026	0.562821	0.633333	0.641026
4	0.649700	0.696665	0.717949	0.701590	0.738608	0.717949
5	0.572900	0.669490	0.717949	0.698396	0.752650	0.717949
6	0.526200	0.654422	0.743590	0.731136	0.772894	0.743590
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 4 on its validation set...
 [2/2 00:00]
Fold 4 Validation: F1=0.7311, Accuracy=0.7436
New best model found in Fold 4 with Val F1: 0.7311
\n==================================================\nTraining Fold 5/5\n==================================================
Tokenizing datasets for Fold 5...
Map: 100%
 159/159 [00:00<00:00, 177.75 examples/s]
Map: 100%
 39/39 [00:00<00:00, 80.85 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 5...
 [60/60 00:46, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.043800	1.016864	0.435897	0.264652	0.190007	0.435897
2	0.955500	0.894982	0.487179	0.356505	0.405405	0.487179
3	0.826800	0.806528	0.692308	0.655564	0.731054	0.692308
4	0.721000	0.685634	0.692308	0.641803	0.732669	0.692308
5	0.627900	0.644750	0.717949	0.691498	0.753917	0.717949
6	0.598400	0.624230	0.769231	0.750469	0.803419	0.769231
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 5 on its validation set...
 [2/2 00:00]
Fold 5 Validation: F1=0.7505, Accuracy=0.7692
New best model found in Fold 5 with Val F1: 0.7505
\nAverage Cross-Validation Results:
  Average eval_loss: 0.6523
  Average eval_accuracy: 0.7326
  Average eval_f1: 0.7233
  Average eval_precision: 0.7446
  Average eval_recall: 0.7326
\nLoading the overall best model for final evaluation...
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
\nPreparing test set for final evaluation...
Map: 100%
 50/50 [00:00<00:00, 451.17 examples/s]
Evaluating overall best model on the Test set...
Final Test Set Evaluation: 100%
 2/2 [00:00<00:00,  4.00it/s]
  Test Set: Accuracy=0.6200, F1=0.5735, Precision=0.5795, Recall=0.6200
\nTest Set Performance Report (Overall Best Model):
              precision    recall  f1-score   support

AI-Generated       0.83      1.00      0.91        10
   Authentic       0.58      0.78      0.67        23
     Generic       0.43      0.18      0.25        17

    accuracy                           0.62        50
   macro avg       0.61      0.65      0.61        50
weighted avg       0.58      0.62      0.57        50

\nTest Set Performance Summary (Overall Best Model):
  AI-Generated: F1=0.9091, Precision=0.8333, Recall=1.0000
  Authentic: F1=0.6667, Precision=0.5806, Recall=0.7826
  Generic: F1=0.2500, Precision=0.4286, Recall=0.1765
\n  Overall Test: F1=0.5735, Accuracy=0.6200
\nSaving overall best model and tokenizer...
\nTraining, cross-validation, and final test evaluation complete.
Best model and tokenizer saved to ./distilbert_classifier_cv
Visualization files saved: distilbert_training_history_cv_best_fold.png, distilbert_confusion_matrix_cv_test.png
"""


"""
Starting DistilBERT model training with 5-fold cross-validation.
Model: distilbert-base-uncased, Max Length: 384, Batch Size: 16
Learning Rate: 3e-05, Epochs per fold: 6
Loading and preprocessing data...
Loaded 399 samples with 3 classes: ['AI-Generated' 'Authentic' 'Generic']
/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
Using device: cuda
\n==================================================\nTraining Fold 1/5\n==================================================
Tokenizing datasets for Fold 1...
Map: 100%
 255/255 [00:00<00:00, 528.23 examples/s]
Map: 100%
 64/64 [00:00<00:00, 469.99 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 1...
 [96/96 00:44, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.034900	0.977623	0.515625	0.350838	0.265869	0.515625
2	0.934300	0.866596	0.625000	0.520615	0.468056	0.625000
3	0.803300	0.788883	0.640625	0.552462	0.740196	0.640625
4	0.687100	0.776808	0.640625	0.571296	0.730329	0.640625
5	0.612700	0.757717	0.671875	0.613364	0.758977	0.671875
6	0.578200	0.760914	0.718750	0.688138	0.747494	0.718750
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 1 on its validation set...
 [2/2 00:00]
Fold 1 Validation: F1=0.6881, Accuracy=0.7188
New best model found in Fold 1 with Val F1: 0.6881
\n==================================================\nTraining Fold 2/5\n==================================================
Tokenizing datasets for Fold 2...
Map: 100%
 255/255 [00:00<00:00, 447.75 examples/s]
Map: 100%
 64/64 [00:00<00:00, 463.74 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 2...
 [96/96 01:05, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.011500	0.954399	0.500000	0.333333	0.250000	0.500000
2	0.887000	0.886215	0.578125	0.464496	0.474311	0.578125
3	0.771700	0.792347	0.687500	0.628111	0.807692	0.687500
4	0.662400	0.738750	0.687500	0.640642	0.736639	0.687500
5	0.583900	0.718414	0.671875	0.627207	0.655320	0.671875
6	0.535400	0.703655	0.703125	0.673899	0.722183	0.703125
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 2 on its validation set...
 [2/2 00:00]
Fold 2 Validation: F1=0.6739, Accuracy=0.7031
\n==================================================\nTraining Fold 3/5\n==================================================
Tokenizing datasets for Fold 3...
Map: 100%
 255/255 [00:00<00:00, 501.67 examples/s]
Map: 100%
 64/64 [00:00<00:00, 476.79 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 3...
 [96/96 01:33, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.029100	0.970848	0.500000	0.333333	0.250000	0.500000
2	0.930700	0.882786	0.562500	0.443414	0.469792	0.562500
3	0.829900	0.831921	0.640625	0.598633	0.695159	0.640625
4	0.719700	0.773867	0.671875	0.646294	0.679167	0.671875
5	0.621000	0.748499	0.671875	0.662678	0.680054	0.671875
6	0.565700	0.742212	0.640625	0.635658	0.654167	0.640625
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 3 on its validation set...
 [2/2 00:00]
Fold 3 Validation: F1=0.6627, Accuracy=0.6719
\n==================================================\nTraining Fold 4/5\n==================================================
Tokenizing datasets for Fold 4...
Map: 100%
 255/255 [00:00<00:00, 469.22 examples/s]
Map: 100%
 64/64 [00:00<00:00, 465.69 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 4...
 [96/96 01:19, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.011200	0.981846	0.500000	0.333333	0.250000	0.500000
2	0.910200	0.898193	0.625000	0.550325	0.590545	0.625000
3	0.789800	0.918638	0.609375	0.517528	0.780702	0.609375
4	0.682900	0.809565	0.656250	0.634264	0.658381	0.656250
5	0.594500	0.803768	0.640625	0.629562	0.629220	0.640625
6	0.541600	0.796362	0.640625	0.622255	0.640394	0.640625
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 4 on its validation set...
 [2/2 00:00]
Fold 4 Validation: F1=0.6343, Accuracy=0.6562
\n==================================================\nTraining Fold 5/5\n==================================================
Tokenizing datasets for Fold 5...
Map: 100%
 256/256 [00:00<00:00, 467.78 examples/s]
Map: 100%
 63/63 [00:00<00:00, 448.94 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 5...
 [96/96 01:14, Epoch 6/6]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.010300	0.973710	0.507937	0.342189	0.257999	0.507937
2	0.908900	0.887796	0.650794	0.575795	0.793063	0.650794
3	0.776600	0.836000	0.634921	0.538370	0.747497	0.634921
4	0.672700	0.788001	0.634921	0.579191	0.691643	0.634921
5	0.591300	0.785594	0.634921	0.579191	0.691643	0.634921
6	0.540000	0.771967	0.619048	0.567619	0.605820	0.619048
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 5 on its validation set...
 [2/2 00:00]
Fold 5 Validation: F1=0.5792, Accuracy=0.6349
\nAverage Cross-Validation Results:
  Average eval_loss: 0.7621
  Average eval_accuracy: 0.6770
  Average eval_f1: 0.6476
  Average eval_precision: 0.7000
  Average eval_recall: 0.6770
\nLoading the overall best model for final evaluation...
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
\nPreparing test set for final evaluation...
Map: 100%
 80/80 [00:00<00:00, 444.99 examples/s]
Evaluating overall best model on the Test set...
Final Test Set Evaluation: 100%
 3/3 [00:00<00:00,  3.60it/s]
  Test Set: Accuracy=0.6750, F1=0.6106, Precision=0.7033, Recall=0.6750
\nTest Set Performance Report (Overall Best Model):
              precision    recall  f1-score   support

AI-Generated       0.80      0.75      0.77        16
   Authentic       0.64      0.95      0.76        41
     Generic       0.75      0.13      0.22        23

    accuracy                           0.68        80
   macro avg       0.73      0.61      0.59        80
weighted avg       0.70      0.68      0.61        80

\nTest Set Performance Summary (Overall Best Model):
  AI-Generated: F1=0.7742, Precision=0.8000, Recall=0.7500
  Authentic: F1=0.7647, Precision=0.6393, Recall=0.9512
  Generic: F1=0.2222, Precision=0.7500, Recall=0.1304
\n  Overall Test: F1=0.6106, Accuracy=0.6750
\nSaving overall best model and tokenizer...
\nTraining, cross-validation, and final test evaluation complete.
Best model and tokenizer saved to ./distilbert_classifier_cv
Visualization files saved: distilbert_training_history_cv_best_fold.png, distilbert_confusion_matrix_cv_test.png
"""
