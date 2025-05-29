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
MAX_LENGTH = 512
BATCH_SIZE = 16  # Batch size for training within a fold
EVAL_BATCH_SIZE = BATCH_SIZE * 2  # Can be larger for evaluation
LEARNING_RATE = 2e-5
NUM_EPOCHS_PER_FOLD = 10  # Renamed for clarity
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
Model: distilbert-base-uncased, Max Length: 512, Batch Size: 16
Learning Rate: 2e-05, Epochs per fold: 10
Loading and preprocessing data...
Loaded 346 samples with 3 classes: ['AI-Generated' 'Authentic' 'Generic']
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
 220/220 [00:01<00:00, 208.75 examples/s]
Map: 100%
 56/56 [00:00<00:00, 200.40 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 1...
 [140/140 02:11, Epoch 10/10]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	0.999500	0.950893	0.517857	0.353361	0.268176	0.517857
2	0.896200	0.857361	0.607143	0.494494	0.473039	0.607143
3	0.765900	0.787576	0.642857	0.529524	0.458075	0.642857
4	0.661900	0.727502	0.678571	0.602742	0.770651	0.678571
5	0.562300	0.688832	0.696429	0.653727	0.694792	0.696429
6	0.500000	0.667872	0.714286	0.667532	0.739373	0.714286
7	0.464500	0.671569	0.660714	0.654178	0.654533	0.660714
8	0.394100	0.656399	0.696429	0.691593	0.688799	0.696429
9	0.355500	0.658273	0.678571	0.676098	0.674513	0.678571
10	0.331800	0.652672	0.678571	0.676098	0.674513	0.678571
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 1 on its validation set...
 [2/2 00:00]
Fold 1 Validation: F1=0.6916, Accuracy=0.6964
New best model found in Fold 1 with Val F1: 0.6916
\n==================================================\nTraining Fold 2/5\n==================================================
Tokenizing datasets for Fold 2...
Map: 100%
 221/221 [00:01<00:00, 155.04 examples/s]
Map: 100%
 55/55 [00:00<00:00, 142.57 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 2...
 [112/140 02:52 < 00:43, 0.64 it/s, Epoch 8/10]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.030800	0.955194	0.527273	0.364069	0.278017	0.527273
2	0.920400	0.856383	0.527273	0.364069	0.278017	0.527273
3	0.814500	0.785782	0.636364	0.528076	0.482197	0.636364
4	0.715400	0.738201	0.672727	0.585564	0.646263	0.672727
5	0.613900	0.711366	0.636364	0.615469	0.612902	0.636364
6	0.534200	0.712236	0.618182	0.601595	0.603535	0.618182
7	0.464000	0.704907	0.600000	0.610105	0.626141	0.600000
8	0.409800	0.706622	0.545455	0.555555	0.568822	0.545455
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 2 on its validation set...
 [2/2 00:00]
Fold 2 Validation: F1=0.6155, Accuracy=0.6364
\n==================================================\nTraining Fold 3/5\n==================================================
Tokenizing datasets for Fold 3...
Map: 100%
 221/221 [00:00<00:00, 439.18 examples/s]
Map: 100%
 55/55 [00:00<00:00, 427.55 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 3...
 [140/140 03:26, Epoch 10/10]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.058700	0.951163	0.527273	0.364069	0.278017	0.527273
2	0.934000	0.860929	0.527273	0.364069	0.278017	0.527273
3	0.829100	0.786488	0.636364	0.543301	0.784787	0.636364
4	0.734700	0.698739	0.690909	0.651716	0.708658	0.690909
5	0.664500	0.636790	0.727273	0.704556	0.734732	0.727273
6	0.569800	0.609788	0.727273	0.709360	0.724444	0.727273
7	0.498800	0.577763	0.727273	0.713154	0.730847	0.727273
8	0.439000	0.560534	0.781818	0.779383	0.781629	0.781818
9	0.403600	0.563504	0.800000	0.799465	0.801356	0.800000
10	0.376900	0.558458	0.781818	0.779383	0.781629	0.781818
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 3 on its validation set...
 [2/2 00:00]
Fold 3 Validation: F1=0.7995, Accuracy=0.8000
New best model found in Fold 3 with Val F1: 0.7995
\n==================================================\nTraining Fold 4/5\n==================================================
Tokenizing datasets for Fold 4...
Map: 100%
 221/221 [00:00<00:00, 422.91 examples/s]
Map: 100%
 55/55 [00:00<00:00, 326.86 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 4...
 [140/140 03:54, Epoch 10/10]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.020100	0.977091	0.509091	0.343483	0.259174	0.509091
2	0.917900	0.873717	0.509091	0.343483	0.259174	0.509091
3	0.811500	0.797305	0.672727	0.577989	0.787475	0.672727
4	0.702500	0.794722	0.654545	0.617080	0.637229	0.654545
5	0.626300	0.736652	0.709091	0.700606	0.701573	0.709091
6	0.542900	0.695270	0.727273	0.717778	0.727035	0.727273
7	0.486500	0.684967	0.709091	0.695534	0.707401	0.709091
8	0.428100	0.676682	0.763636	0.742730	0.793793	0.763636
9	0.407000	0.689582	0.727273	0.723892	0.724783	0.727273
10	0.369100	0.685324	0.727273	0.722885	0.726227	0.727273
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 4 on its validation set...
 [2/2 00:00]
Fold 4 Validation: F1=0.7427, Accuracy=0.7636
\n==================================================\nTraining Fold 5/5\n==================================================
Tokenizing datasets for Fold 5...
Map: 100%
 221/221 [00:00<00:00, 384.27 examples/s]
Map: 100%
 55/55 [00:00<00:00, 320.17 examples/s]
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Starting training for Fold 5...
 [140/140 03:45, Epoch 10/10]
Epoch	Training Loss	Validation Loss	Accuracy	F1	Precision	Recall
1	1.016600	0.962988	0.509091	0.343483	0.259174	0.509091
2	0.905200	0.897115	0.509091	0.343483	0.259174	0.509091
3	0.808100	0.820894	0.672727	0.581218	0.800791	0.672727
4	0.697500	0.752242	0.636364	0.619055	0.622112	0.636364
5	0.605700	0.708987	0.672727	0.659993	0.675666	0.672727
6	0.510300	0.663477	0.709091	0.689948	0.722807	0.709091
7	0.446700	0.660920	0.690909	0.691866	0.693905	0.690909
8	0.388300	0.646184	0.709091	0.698711	0.711212	0.709091
9	0.354300	0.654140	0.709091	0.698711	0.711212	0.709091
10	0.338300	0.651258	0.709091	0.698711	0.711212	0.709091
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Evaluating best model of Fold 5 on its validation set...
 [2/2 00:00]
Fold 5 Validation: F1=0.6987, Accuracy=0.7091
\nAverage Cross-Validation Results:
  Average eval_loss: 0.6508
  Average eval_accuracy: 0.7211
  Average eval_f1: 0.7096
  Average eval_precision: 0.7216
  Average eval_recall: 0.7211
\nLoading the overall best model for final evaluation...
Some weights of DistilBertForSequenceClassification were not initialized from the model checkpoint at distilbert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight', 'pre_classifier.bias', 'pre_classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
\nPreparing test set for final evaluation...
Map: 100%
 70/70 [00:00<00:00, 469.28 examples/s]
Evaluating overall best model on the Test set...
Final Test Set Evaluation: 100%
 3/3 [00:01<00:00,  3.10it/s]
  Test Set: Accuracy=0.6857, F1=0.6797, Precision=0.6837, Recall=0.6857
\nTest Set Performance Report (Overall Best Model):
              precision    recall  f1-score   support

AI-Generated       1.00      0.83      0.91        12
   Authentic       0.69      0.81      0.74        36
     Generic       0.50      0.41      0.45        22

    accuracy                           0.69        70
   macro avg       0.73      0.68      0.70        70
weighted avg       0.68      0.69      0.68        70

\nTest Set Performance Summary (Overall Best Model):
  AI-Generated: F1=0.9091, Precision=1.0000, Recall=0.8333
  Authentic: F1=0.7436, Precision=0.6905, Recall=0.8056
  Generic: F1=0.4500, Precision=0.5000, Recall=0.4091
\n  Overall Test: F1=0.6797, Accuracy=0.6857
\nSaving overall best model and tokenizer...
\nTraining, cross-validation, and final test evaluation complete.
Best model and tokenizer saved to ./distilbert_classifier_cv
Visualization files saved: distilbert_training_history_cv_best_fold.png, distilbert_confusion_matrix_cv_test.png
"""
