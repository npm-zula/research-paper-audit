import pandas as pd
import torch
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
import copy

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)  # Add for sklearn and numpy consistency

# Configuration
CROSS_VAL_FOLDS = 5
MODEL_NAME = 'bert-base-uncased'
MAX_LENGTH = 512
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10  # Number of epochs per fold

# Define the Dataset class


class ReviewDataset(Dataset):
    def __init__(self, abstracts, labels, tokenizer, max_length=512):
        self.abstracts = abstracts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.abstracts)

    def __getitem__(self, idx):
        abstract = str(self.abstracts[idx])
        encoding = self.tokenizer(
            abstract,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    # Filename updated in function
    plt.savefig('bert_confusion_matrix_cv_test.png')
    plt.close()


def plot_training_history(train_losses, val_losses, val_accuracies, val_f1s, val_precisions, val_recalls):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 3, 2)
    plt.plot(val_accuracies, label='Accuracy')
    plt.plot(val_f1s, label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')

    plt.subplot(1, 3, 3)
    plt.plot(val_precisions, label='Precision')
    plt.plot(val_recalls, label='Recall')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision and Recall')

    plt.tight_layout()
    plt.savefig('bert_training_history_cv_best_fold.png')  # Updated filename
    plt.close()


def train_model():
    # Load and preprocess data with different encoding
    try:
        # Try reading with 'latin-1' encoding
        # Assuming corpus-final.csv is in the same directory
        df = pd.read_csv('/content/corpus-final.csv', encoding='latin-1')
    except:
        try:
            # If that fails, try with 'cp1252' encoding
            df = pd.read_csv('/content/corpus-final.csv', encoding='cp1252')
        except:
            # If all else fails, try with utf-8 and error handling
            df = pd.read_csv('/content/corpus-final.csv',
                             encoding='utf-8', errors='replace')

    # Clean the text data
    df['Abstract'] = df['Abstract'].fillna('')
    df['Abstract'] = df['Abstract'].str.encode(
        'ascii', 'ignore').str.decode('ascii')

    # Drop rows with missing values
    df = df.dropna(subset=['Abstract', 'Review Type'])

    # Encode labels
    le = LabelEncoder()
    df['labels'] = le.fit_transform(df['Review Type'])
    class_names = le.classes_

    # Split data into Train/CV and Test sets
    train_cv_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df['labels']
    )

    X_cv = train_cv_df['Abstract'].values
    y_cv = train_cv_df['labels'].values

    # Initialize tokenizer (once is fine as it's static)
    tokenizer = BertTokenizer.from_pretrained(MODEL_NAME)

    # Setup cross-validation
    skf = StratifiedKFold(n_splits=CROSS_VAL_FOLDS,
                          shuffle=True, random_state=SEED)

    fold_results_summary = []
    best_overall_val_f1 = 0.0
    best_model_state_dict = None

    # To store metrics for plotting the best fold's history
    best_fold_train_losses_hist = []
    best_fold_val_losses_hist = []
    best_fold_val_accuracies_hist = []
    best_fold_val_f1s_hist = []
    best_fold_val_precisions_hist = []
    best_fold_val_recalls_hist = []

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_cv, y_cv)):
        print(f"\\n{'='*50}\\nTraining Fold {fold+1}/{CROSS_VAL_FOLDS}\\n{'='*50}")

        abstracts_train_fold, labels_train_fold = X_cv[train_idx], y_cv[train_idx]
        abstracts_val_fold, labels_val_fold = X_cv[val_idx], y_cv[val_idx]

        # Initialize model for each fold
        model = BertForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(class_names)
        )
        model.to(device)

        # Create datasets and dataloaders for the current fold
        train_dataset_fold = ReviewDataset(
            abstracts_train_fold, labels_train_fold, tokenizer, MAX_LENGTH)
        val_dataset_fold = ReviewDataset(
            abstracts_val_fold, labels_val_fold, tokenizer, MAX_LENGTH)
        train_loader_fold = DataLoader(
            train_dataset_fold, batch_size=BATCH_SIZE, shuffle=True)
        # Can use larger batch for validation
        val_loader_fold = DataLoader(
            val_dataset_fold, batch_size=BATCH_SIZE * 2)

        optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

        # Lists to store metrics for the current fold's epochs
        current_fold_train_losses = []
        current_fold_val_losses = []
        current_fold_val_accuracies = []
        current_fold_val_f1s = []
        current_fold_val_precisions = []
        current_fold_val_recalls = []

        # Training loop for the current fold
        for epoch in range(NUM_EPOCHS):
            model.train()
            total_loss_fold = 0

            for batch in tqdm(train_loader_fold, desc=f'Fold {fold+1}, Epoch {epoch + 1}/{NUM_EPOCHS}'):
                optimizer.zero_grad()
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels_batch = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels_batch
                )
                loss = outputs.loss
                total_loss_fold += loss.item()
                loss.backward()
                optimizer.step()

            avg_train_loss_fold = total_loss_fold / len(train_loader_fold)
            current_fold_train_losses.append(avg_train_loss_fold)

            # Validation for the current fold
            model.eval()
            val_loss_fold = 0
            all_predictions_fold = []
            all_labels_fold = []

            with torch.no_grad():
                for batch in val_loader_fold:
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels_batch = batch['labels'].to(device)
                    outputs = model(
                        input_ids=input_ids, attention_mask=attention_mask, labels=labels_batch)
                    val_loss_fold += outputs.loss.item()
                    predictions = torch.argmax(outputs.logits, dim=1)
                    all_predictions_fold.extend(predictions.cpu().numpy())
                    all_labels_fold.extend(labels_batch.cpu().numpy())

            avg_val_loss_fold = val_loss_fold / len(val_loader_fold)
            accuracy_fold = np.mean(
                np.array(all_predictions_fold) == np.array(all_labels_fold))
            precision_fold, recall_fold, f1_fold, _ = precision_recall_fscore_support(
                all_labels_fold, all_predictions_fold, average='weighted', zero_division=0
            )

            current_fold_val_losses.append(avg_val_loss_fold)
            current_fold_val_accuracies.append(accuracy_fold)
            current_fold_val_f1s.append(f1_fold)
            current_fold_val_precisions.append(precision_fold)
            current_fold_val_recalls.append(recall_fold)

            print(f'Fold {fold+1}, Epoch {epoch + 1}: Train Loss: {avg_train_loss_fold:.4f}, Val Loss: {avg_val_loss_fold:.4f}, Val Acc: {accuracy_fold:.4f}, Val F1: {f1_fold:.4f}')

        # After all epochs for the current fold
        fold_final_val_f1 = current_fold_val_f1s[-1]  # F1 from the last epoch
        fold_results_summary.append({
            'fold': fold + 1,
            'val_loss': current_fold_val_losses[-1],
            'val_accuracy': current_fold_val_accuracies[-1],
            'val_f1': fold_final_val_f1,
            'val_precision': current_fold_val_precisions[-1],
            'val_recall': current_fold_val_recalls[-1]
        })

        if fold_final_val_f1 > best_overall_val_f1:
            best_overall_val_f1 = fold_final_val_f1
            best_model_state_dict = copy.deepcopy(model.state_dict())
            # Save training history of this best fold
            best_fold_train_losses_hist = list(current_fold_train_losses)
            best_fold_val_losses_hist = list(current_fold_val_losses)
            best_fold_val_accuracies_hist = list(current_fold_val_accuracies)
            best_fold_val_f1s_hist = list(current_fold_val_f1s)
            best_fold_val_precisions_hist = list(current_fold_val_precisions)
            best_fold_val_recalls_hist = list(current_fold_val_recalls)
            print(
                f"New best model found in Fold {fold+1} with F1: {best_overall_val_f1:.4f}")

    # After all folds
    print("\\nAverage Cross-Validation Results:")
    avg_cv_metrics = pd.DataFrame(fold_results_summary).mean().to_dict()
    for metric, value in avg_cv_metrics.items():
        if metric != 'fold':
            print(f"  Average {metric}: {value:.4f}")

    # Load the best model
    best_model = BertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(class_names))
    best_model.load_state_dict(best_model_state_dict)
    best_model.to(device)

    print("\\nSaving best model and tokenizer...")
    best_model.save_pretrained("./bert_classifier_cv")
    tokenizer.save_pretrained("./bert_classifier_cv")

    # Plot training history of the best fold
    if best_fold_train_losses_hist:  # Check if any best fold was found
        plot_training_history(
            best_fold_train_losses_hist,
            best_fold_val_losses_hist,
            best_fold_val_accuracies_hist,
            best_fold_val_f1s_hist,
            best_fold_val_precisions_hist,
            best_fold_val_recalls_hist
        )

    # Final evaluation on the Test set using the best model from CV
    print("\\nPerforming final evaluation on the Test set...")
    test_abstracts = test_df['Abstract'].values
    test_labels_true = test_df['labels'].values

    test_dataset = ReviewDataset(
        test_abstracts, test_labels_true, tokenizer, MAX_LENGTH)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2)

    best_model.eval()
    all_predictions_test = []
    all_labels_test = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Evaluating on Test Set"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels_batch = batch['labels'].to(
                device)  # True labels for test set

            # No labels for inference
            outputs = best_model(input_ids=input_ids,
                                 attention_mask=attention_mask)
            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions_test.extend(predictions.cpu().numpy())
            # Collect true labels for report
            all_labels_test.extend(labels_batch.cpu().numpy())

    # Plot confusion matrix for the test set
    # Filename updated in function
    plot_confusion_matrix(all_labels_test, all_predictions_test, class_names)

    # Generate and print classification report for the test set
    test_report_dict = classification_report(all_labels_test, all_predictions_test,
                                             target_names=class_names, output_dict=True, zero_division=0)
    test_report_str = classification_report(all_labels_test, all_predictions_test,
                                            target_names=class_names, zero_division=0)

    print("\\nTest Set Performance Report:")
    print(test_report_str)

    print("\\nTest Set Performance Summary:")
    for cls in class_names:
        print(f"  {cls}: F1={test_report_dict[cls]['f1-score']:.4f}, "
              f"Precision={test_report_dict[cls]['precision']:.4f}, "
              f"Recall={test_report_dict[cls]['recall']:.4f}")

    print(f"\\n  Overall Test: F1={test_report_dict['weighted avg']['f1-score']:.4f}, "
          f"Accuracy={test_report_dict['accuracy']:.4f}")

    return best_model, tokenizer, class_names


if __name__ == "__main__":
    print(
        f"Starting BERT model training with {CROSS_VAL_FOLDS}-fold cross-validation.")
    print(
        f"Model: {MODEL_NAME}, Max Length: {MAX_LENGTH}, Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}, Epochs per fold: {NUM_EPOCHS}")

    final_model, final_tokenizer, final_class_names = train_model()

    print("\\nTraining, cross-validation, and final test evaluation complete.")
    print(f"Best model and tokenizer saved to ./bert_classifier_cv")
    print("Visualization files saved: bert_training_history_cv_best_fold.png, bert_confusion_matrix_cv_test.png")


"""
\nAverage Cross-Validation Results:
  Average val_loss: 0.9833
  Average val_accuracy: 0.6486
  Average val_f1: 0.6445
  Average val_precision: 0.6675
  Average val_recall: 0.6486

Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.

\nSaving best model and tokenizer...
\nPerforming final evaluation on the Test set...

Evaluating on Test Set: 100%|██████████| 5/5 [00:02<00:00,  2.10it/s]

\nTest Set Performance Report:
              precision    recall  f1-score   support

AI-Generated       0.83      0.83      0.83        12
   Authentic       0.67      0.72      0.69        36
     Generic       0.53      0.45      0.49        22

    accuracy                           0.66        70
   macro avg       0.68      0.67      0.67        70
weighted avg       0.65      0.66      0.65        70

\nTest Set Performance Summary:
  AI-Generated: F1=0.8333, Precision=0.8333, Recall=0.8333
  Authentic: F1=0.6933, Precision=0.6667, Recall=0.7222
  Generic: F1=0.4878, Precision=0.5263, Recall=0.4545
\n  Overall Test: F1=0.6527, Accuracy=0.6571
\nTraining, cross-validation, and final test evaluation complete.
"""
