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
MAX_LENGTH = 384
BATCH_SIZE = 32
LEARNING_RATE = 4e-5
NUM_EPOCHS = 7  # Number of epochs per fold

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
        # Assuming corpus-thefinal.csv is in the same directory
        df = pd.read_csv('/content/corpus-thefinal.csv', encoding='latin-1')
    except:
        try:
            # If that fails, try with 'cp1252' encoding
            df = pd.read_csv('/content/corpus-thefinal.csv', encoding='cp1252')
        except:
            # If all else fails, try with utf-8 and error handling
            df = pd.read_csv('/content/corpus-thefinal.csv',
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
Starting BERT model training with 5-fold cross-validation.
Model: bert-base-uncased, Max Length: 384, Batch Size: 16
Learning Rate: 3e-05, Epochs per fold: 6
Using device: cuda
\n==================================================\nTraining Fold 1/5\n==================================================
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fold 1, Epoch 1/6: 100%|██████████| 10/10 [00:10<00:00,  1.05s/it]
Fold 1, Epoch 1: Train Loss: 1.0764, Val Loss: 1.0221, Val Acc: 0.4500, Val F1: 0.2793
Fold 1, Epoch 2/6: 100%|██████████| 10/10 [00:11<00:00,  1.18s/it]
Fold 1, Epoch 2: Train Loss: 1.0139, Val Loss: 0.9397, Val Acc: 0.4500, Val F1: 0.2793
Fold 1, Epoch 3/6: 100%|██████████| 10/10 [00:10<00:00,  1.05s/it]
Fold 1, Epoch 3: Train Loss: 0.9021, Val Loss: 0.8191, Val Acc: 0.5250, Val F1: 0.5299
Fold 1, Epoch 4/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 1, Epoch 4: Train Loss: 0.7591, Val Loss: 0.8636, Val Acc: 0.5250, Val F1: 0.4308
Fold 1, Epoch 5/6: 100%|██████████| 10/10 [00:10<00:00,  1.01s/it]
Fold 1, Epoch 5: Train Loss: 0.6170, Val Loss: 0.6718, Val Acc: 0.6750, Val F1: 0.6641
Fold 1, Epoch 6/6: 100%|██████████| 10/10 [00:10<00:00,  1.01s/it]
Fold 1, Epoch 6: Train Loss: 0.4853, Val Loss: 0.6235, Val Acc: 0.6750, Val F1: 0.6778
New best model found in Fold 1 with F1: 0.6778
\n==================================================\nTraining Fold 2/5\n==================================================
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fold 2, Epoch 1/6: 100%|██████████| 10/10 [00:10<00:00,  1.02s/it]
Fold 2, Epoch 1: Train Loss: 1.1043, Val Loss: 1.0148, Val Acc: 0.5000, Val F1: 0.4015
Fold 2, Epoch 2/6: 100%|██████████| 10/10 [00:10<00:00,  1.02s/it]
Fold 2, Epoch 2: Train Loss: 0.9968, Val Loss: 0.9622, Val Acc: 0.3750, Val F1: 0.3008
Fold 2, Epoch 3/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 2, Epoch 3: Train Loss: 0.8368, Val Loss: 0.8469, Val Acc: 0.4250, Val F1: 0.3719
Fold 2, Epoch 4/6: 100%|██████████| 10/10 [00:10<00:00,  1.04s/it]
Fold 2, Epoch 4: Train Loss: 0.7001, Val Loss: 0.8027, Val Acc: 0.6250, Val F1: 0.6048
Fold 2, Epoch 5/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 2, Epoch 5: Train Loss: 0.5966, Val Loss: 0.9145, Val Acc: 0.5750, Val F1: 0.4595
Fold 2, Epoch 6/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 2, Epoch 6: Train Loss: 0.4441, Val Loss: 0.6457, Val Acc: 0.7750, Val F1: 0.7764
New best model found in Fold 2 with F1: 0.7764
\n==================================================\nTraining Fold 3/5\n==================================================
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fold 3, Epoch 1/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 3, Epoch 1: Train Loss: 1.0336, Val Loss: 0.9717, Val Acc: 0.4500, Val F1: 0.2793
Fold 3, Epoch 2/6: 100%|██████████| 10/10 [00:10<00:00,  1.02s/it]
Fold 3, Epoch 2: Train Loss: 0.8783, Val Loss: 0.8767, Val Acc: 0.6000, Val F1: 0.5394
Fold 3, Epoch 3/6: 100%|██████████| 10/10 [00:10<00:00,  1.02s/it]
Fold 3, Epoch 3: Train Loss: 0.6558, Val Loss: 0.8718, Val Acc: 0.5750, Val F1: 0.5785
Fold 3, Epoch 4/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 3, Epoch 4: Train Loss: 0.4334, Val Loss: 0.9983, Val Acc: 0.5750, Val F1: 0.5611
Fold 3, Epoch 5/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 3, Epoch 5: Train Loss: 0.2678, Val Loss: 1.1643, Val Acc: 0.6250, Val F1: 0.6224
Fold 3, Epoch 6/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 3, Epoch 6: Train Loss: 0.1443, Val Loss: 1.4349, Val Acc: 0.6250, Val F1: 0.6132
\n==================================================\nTraining Fold 4/5\n==================================================
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fold 4, Epoch 1/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 4, Epoch 1: Train Loss: 1.0808, Val Loss: 1.0603, Val Acc: 0.4615, Val F1: 0.2915
Fold 4, Epoch 2/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 4, Epoch 2: Train Loss: 1.0210, Val Loss: 1.0254, Val Acc: 0.4615, Val F1: 0.2915
Fold 4, Epoch 3/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 4, Epoch 3: Train Loss: 0.9024, Val Loss: 0.8706, Val Acc: 0.5385, Val F1: 0.4725
Fold 4, Epoch 4/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 4, Epoch 4: Train Loss: 0.6921, Val Loss: 0.7745, Val Acc: 0.5641, Val F1: 0.5741
Fold 4, Epoch 5/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 4, Epoch 5: Train Loss: 0.4195, Val Loss: 0.8632, Val Acc: 0.5128, Val F1: 0.5248
Fold 4, Epoch 6/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 4, Epoch 6: Train Loss: 0.2423, Val Loss: 1.1566, Val Acc: 0.5385, Val F1: 0.5495
\n==================================================\nTraining Fold 5/5\n==================================================
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fold 5, Epoch 1/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 5, Epoch 1: Train Loss: 1.0676, Val Loss: 0.9698, Val Acc: 0.4359, Val F1: 0.2745
Fold 5, Epoch 2/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 5, Epoch 2: Train Loss: 0.9150, Val Loss: 0.8841, Val Acc: 0.6154, Val F1: 0.5304
Fold 5, Epoch 3/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 5, Epoch 3: Train Loss: 0.7813, Val Loss: 0.8005, Val Acc: 0.6923, Val F1: 0.6960
Fold 5, Epoch 4/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 5, Epoch 4: Train Loss: 0.6020, Val Loss: 0.6746, Val Acc: 0.7692, Val F1: 0.7629
Fold 5, Epoch 5/6: 100%|██████████| 10/10 [00:10<00:00,  1.03s/it]
Fold 5, Epoch 5: Train Loss: 0.4301, Val Loss: 0.6207, Val Acc: 0.7692, Val F1: 0.7618
Fold 5, Epoch 6/6: 100%|██████████| 10/10 [00:10<00:00,  1.04s/it]
Fold 5, Epoch 6: Train Loss: 0.3034, Val Loss: 0.4985, Val Acc: 0.8205, Val F1: 0.8179
New best model found in Fold 5 with F1: 0.8179
\nAverage Cross-Validation Results:
  Average val_loss: 0.8718
  Average val_accuracy: 0.6868
  Average val_f1: 0.6870
  Average val_precision: 0.7175
  Average val_recall: 0.6868
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
\nSaving best model and tokenizer...
\nPerforming final evaluation on the Test set...
Evaluating on Test Set: 100%|██████████| 2/2 [00:01<00:00,  1.49it/s]
\nTest Set Performance Report:
              precision    recall  f1-score   support

AI-Generated       0.91      1.00      0.95        10
   Authentic       0.83      0.65      0.73        23
     Generic       0.62      0.76      0.68        17

    accuracy                           0.76        50
   macro avg       0.79      0.81      0.79        50
weighted avg       0.78      0.76      0.76        50

\nTest Set Performance Summary:
  AI-Generated: F1=0.9524, Precision=0.9091, Recall=1.0000
  Authentic: F1=0.7317, Precision=0.8333, Recall=0.6522
  Generic: F1=0.6842, Precision=0.6190, Recall=0.7647
\n  Overall Test: F1=0.7597, Accuracy=0.7600
\nTraining, cross-validation, and final test evaluation complete.
Best model and tokenizer saved to ./bert_classifier_cv
Visualization files saved: bert_training_history_cv_best_fold.png, bert_confusion_matrix_cv_test.png
"""

"""
Starting BERT model training with 5-fold cross-validation.
Model: bert-base-uncased, Max Length: 384, Batch Size: 32
Learning Rate: 4e-05, Epochs per fold: 7
Using device: cuda
\n==================================================\nTraining Fold 1/5\n==================================================
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fold 1, Epoch 1/7: 100%|██████████| 8/8 [00:15<00:00,  1.88s/it]
Fold 1, Epoch 1: Train Loss: 1.0318, Val Loss: 1.0052, Val Acc: 0.5156, Val F1: 0.3508
Fold 1, Epoch 2/7: 100%|██████████| 8/8 [00:15<00:00,  1.96s/it]
Fold 1, Epoch 2: Train Loss: 1.0068, Val Loss: 0.9798, Val Acc: 0.5156, Val F1: 0.3508
Fold 1, Epoch 3/7: 100%|██████████| 8/8 [00:15<00:00,  1.94s/it]
Fold 1, Epoch 3: Train Loss: 0.9565, Val Loss: 0.9426, Val Acc: 0.5156, Val F1: 0.3508
Fold 1, Epoch 4/7: 100%|██████████| 8/8 [00:15<00:00,  1.91s/it]
Fold 1, Epoch 4: Train Loss: 0.8123, Val Loss: 0.8775, Val Acc: 0.6094, Val F1: 0.5135
Fold 1, Epoch 5/7: 100%|██████████| 8/8 [00:14<00:00,  1.86s/it]
Fold 1, Epoch 5: Train Loss: 0.6564, Val Loss: 0.8432, Val Acc: 0.6406, Val F1: 0.5807
Fold 1, Epoch 6/7: 100%|██████████| 8/8 [00:14<00:00,  1.87s/it]
Fold 1, Epoch 6: Train Loss: 0.4927, Val Loss: 0.8426, Val Acc: 0.6562, Val F1: 0.6384
Fold 1, Epoch 7/7: 100%|██████████| 8/8 [00:15<00:00,  1.88s/it]
Fold 1, Epoch 7: Train Loss: 0.3169, Val Loss: 0.8729, Val Acc: 0.6250, Val F1: 0.6171
New best model found in Fold 1 with F1: 0.6171
\n==================================================\nTraining Fold 2/5\n==================================================
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fold 2, Epoch 1/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 2, Epoch 1: Train Loss: 1.0473, Val Loss: 1.0238, Val Acc: 0.5000, Val F1: 0.3333
Fold 2, Epoch 2/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 2, Epoch 2: Train Loss: 1.0208, Val Loss: 1.0082, Val Acc: 0.5000, Val F1: 0.3333
Fold 2, Epoch 3/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 2, Epoch 3: Train Loss: 0.9518, Val Loss: 0.9393, Val Acc: 0.5625, Val F1: 0.4514
Fold 2, Epoch 4/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 2, Epoch 4: Train Loss: 0.8367, Val Loss: 0.9001, Val Acc: 0.5938, Val F1: 0.4838
Fold 2, Epoch 5/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 2, Epoch 5: Train Loss: 0.6845, Val Loss: 0.8222, Val Acc: 0.6094, Val F1: 0.5018
Fold 2, Epoch 6/7: 100%|██████████| 8/8 [00:15<00:00,  1.90s/it]
Fold 2, Epoch 6: Train Loss: 0.5146, Val Loss: 0.9688, Val Acc: 0.5312, Val F1: 0.4554
Fold 2, Epoch 7/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 2, Epoch 7: Train Loss: 0.3372, Val Loss: 0.9919, Val Acc: 0.5625, Val F1: 0.5563
\n==================================================\nTraining Fold 3/5\n==================================================
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fold 3, Epoch 1/7: 100%|██████████| 8/8 [00:15<00:00,  1.90s/it]
Fold 3, Epoch 1: Train Loss: 1.0386, Val Loss: 1.0012, Val Acc: 0.5000, Val F1: 0.3333
Fold 3, Epoch 2/7: 100%|██████████| 8/8 [00:15<00:00,  1.88s/it]
Fold 3, Epoch 2: Train Loss: 0.9414, Val Loss: 0.9971, Val Acc: 0.4688, Val F1: 0.3889
Fold 3, Epoch 3/7: 100%|██████████| 8/8 [00:15<00:00,  1.90s/it]
Fold 3, Epoch 3: Train Loss: 0.8137, Val Loss: 0.8341, Val Acc: 0.5156, Val F1: 0.4574
Fold 3, Epoch 4/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 3, Epoch 4: Train Loss: 0.6570, Val Loss: 0.8075, Val Acc: 0.5625, Val F1: 0.5338
Fold 3, Epoch 5/7: 100%|██████████| 8/8 [00:15<00:00,  1.88s/it]
Fold 3, Epoch 5: Train Loss: 0.5503, Val Loss: 0.9428, Val Acc: 0.5156, Val F1: 0.4585
Fold 3, Epoch 6/7: 100%|██████████| 8/8 [00:15<00:00,  1.88s/it]
Fold 3, Epoch 6: Train Loss: 0.4197, Val Loss: 0.9366, Val Acc: 0.5156, Val F1: 0.5157
Fold 3, Epoch 7/7: 100%|██████████| 8/8 [00:15<00:00,  1.88s/it]
Fold 3, Epoch 7: Train Loss: 0.2657, Val Loss: 1.1558, Val Acc: 0.5312, Val F1: 0.5372
\n==================================================\nTraining Fold 4/5\n==================================================
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fold 4, Epoch 1/7: 100%|██████████| 8/8 [00:15<00:00,  1.88s/it]
Fold 4, Epoch 1: Train Loss: 1.0551, Val Loss: 1.0317, Val Acc: 0.5000, Val F1: 0.3333
Fold 4, Epoch 2/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 4, Epoch 2: Train Loss: 1.0235, Val Loss: 1.0121, Val Acc: 0.5000, Val F1: 0.3333
Fold 4, Epoch 3/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 4, Epoch 3: Train Loss: 0.9565, Val Loss: 0.9706, Val Acc: 0.5156, Val F1: 0.4279
Fold 4, Epoch 4/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 4, Epoch 4: Train Loss: 0.8249, Val Loss: 0.9331, Val Acc: 0.4531, Val F1: 0.4201
Fold 4, Epoch 5/7: 100%|██████████| 8/8 [00:15<00:00,  1.88s/it]
Fold 4, Epoch 5: Train Loss: 0.6405, Val Loss: 0.8880, Val Acc: 0.4688, Val F1: 0.4745
Fold 4, Epoch 6/7: 100%|██████████| 8/8 [00:15<00:00,  1.88s/it]
Fold 4, Epoch 6: Train Loss: 0.4313, Val Loss: 1.1462, Val Acc: 0.5000, Val F1: 0.4750
Fold 4, Epoch 7/7: 100%|██████████| 8/8 [00:15<00:00,  1.90s/it]
Fold 4, Epoch 7: Train Loss: 0.2719, Val Loss: 1.2836, Val Acc: 0.5156, Val F1: 0.4919
\n==================================================\nTraining Fold 5/5\n==================================================
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
Fold 5, Epoch 1/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 5, Epoch 1: Train Loss: 1.0318, Val Loss: 1.0067, Val Acc: 0.5079, Val F1: 0.3422
Fold 5, Epoch 2/7: 100%|██████████| 8/8 [00:15<00:00,  1.91s/it]
Fold 5, Epoch 2: Train Loss: 0.9898, Val Loss: 1.0190, Val Acc: 0.5397, Val F1: 0.4797
Fold 5, Epoch 3/7: 100%|██████████| 8/8 [00:15<00:00,  1.90s/it]
Fold 5, Epoch 3: Train Loss: 0.8952, Val Loss: 0.9895, Val Acc: 0.5397, Val F1: 0.4265
Fold 5, Epoch 4/7: 100%|██████████| 8/8 [00:15<00:00,  1.91s/it]
Fold 5, Epoch 4: Train Loss: 0.7213, Val Loss: 0.9490, Val Acc: 0.5714, Val F1: 0.5633
Fold 5, Epoch 5/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 5, Epoch 5: Train Loss: 0.6076, Val Loss: 0.9715, Val Acc: 0.6032, Val F1: 0.5878
Fold 5, Epoch 6/7: 100%|██████████| 8/8 [00:15<00:00,  1.88s/it]
Fold 5, Epoch 6: Train Loss: 0.4451, Val Loss: 1.0946, Val Acc: 0.6349, Val F1: 0.5650
Fold 5, Epoch 7/7: 100%|██████████| 8/8 [00:15<00:00,  1.89s/it]
Fold 5, Epoch 7: Train Loss: 0.3768, Val Loss: 1.0727, Val Acc: 0.5873, Val F1: 0.5713
\nAverage Cross-Validation Results:
  Average val_loss: 1.0754
  Average val_accuracy: 0.5643
  Average val_f1: 0.5548
  Average val_precision: 0.5936
  Average val_recall: 0.5643
Some weights of BertForSequenceClassification were not initialized from the model checkpoint at bert-base-uncased and are newly initialized: ['classifier.bias', 'classifier.weight']
You should probably TRAIN this model on a down-stream task to be able to use it for predictions and inference.
\nSaving best model and tokenizer...
\nPerforming final evaluation on the Test set...
Evaluating on Test Set: 100%|██████████| 2/2 [00:02<00:00,  1.04s/it]
\nTest Set Performance Report:
              precision    recall  f1-score   support

AI-Generated       0.90      0.56      0.69        16
   Authentic       0.68      0.78      0.73        41
     Generic       0.43      0.43      0.43        23

    accuracy                           0.64        80
   macro avg       0.67      0.59      0.62        80
weighted avg       0.65      0.64      0.64        80

\nTest Set Performance Summary:
  AI-Generated: F1=0.6923, Precision=0.9000, Recall=0.5625
  Authentic: F1=0.7273, Precision=0.6809, Recall=0.7805
  Generic: F1=0.4348, Precision=0.4348, Recall=0.4348
\n  Overall Test: F1=0.6362, Accuracy=0.6375
\nTraining, cross-validation, and final test evaluation complete.
Best model and tokenizer saved to ./bert_classifier_cv
Visualization files saved: bert_training_history_cv_best_fold.png, bert_confusion_matrix_cv_test.png
"""
