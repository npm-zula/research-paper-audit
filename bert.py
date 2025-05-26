import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support

# Set random seed for reproducibility
torch.manual_seed(42)

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
    plt.savefig('bert_confusion_matrix.png')
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
    plt.savefig('bert_training_history.png')
    plt.close()


def train_model():
    # Load and preprocess data with different encoding
    try:
        # Try reading with 'latin-1' encoding
        df = pd.read_csv('corpus.csv', encoding='latin-1')
    except:
        try:
            # If that fails, try with 'cp1252' encoding
            df = pd.read_csv('corpus.csv', encoding='cp1252')
        except:
            # If all else fails, try with utf-8 and error handling
            df = pd.read_csv('corpus.csv', encoding='utf-8', errors='replace')

    # Clean the text data
    df['Abstract'] = df['Abstract'].fillna('')
    df['Abstract'] = df['Abstract'].str.encode(
        'ascii', 'ignore').str.decode('ascii')

    # Drop rows with missing values
    df = df.dropna(subset=['Abstract', 'Review Type'])

    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(df['Review Type'])
    class_names = le.classes_

    # Split data
    abstracts_train, abstracts_val, labels_train, labels_val = train_test_split(
        df['Abstract'].values, labels, test_size=0.2, random_state=42
    )

    # Initialize tokenizer and model
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertForSequenceClassification.from_pretrained(
        'bert-base-uncased',
        num_labels=len(le.classes_)
    )

    # Create datasets
    train_dataset = ReviewDataset(abstracts_train, labels_train, tokenizer)
    val_dataset = ReviewDataset(abstracts_val, labels_val, tokenizer)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16)

    # Setup training
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    num_epochs = 10

    # Lists to store metrics for plotting
    train_losses = []
    val_losses = []
    val_accuracies = []
    val_f1s = []
    val_precisions = []
    val_recalls = []

    print(f"Starting BERT model training with model: bert-base-uncased")
    print(f"Max Length: 512, Batch Size: 8")
    print(f"Learning Rate: 2e-5, Epochs: {num_epochs}")

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{num_epochs}'):
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )

            loss = outputs.loss
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        # Calculate average loss for this epoch
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # Validation
        model.eval()
        val_loss = 0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )

                val_loss += outputs.loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)

                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        # Calculate validation loss
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)

        # Calculate evaluation metrics
        accuracy = np.mean(np.array(all_predictions) == np.array(all_labels))
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_predictions, average='weighted'
        )

        # Store metrics for plotting
        val_accuracies.append(accuracy)
        val_f1s.append(f1)
        val_precisions.append(precision)
        val_recalls.append(recall)

        print(f'Epoch {epoch + 1}/{num_epochs}:')
        print(f'  Training Loss: {avg_train_loss:.4f}')
        print(f'  Validation Loss: {avg_val_loss:.4f}')
        print(f'  Validation Accuracy: {accuracy:.4f}')
        print(f'  Validation F1 Score: {f1:.4f}')
        print(f'  Validation Precision: {precision:.4f}')
        print(f'  Validation Recall: {recall:.4f}')

    # Plot training history
    plot_training_history(train_losses, val_losses,
                          val_accuracies, val_f1s, val_precisions, val_recalls)

    # Final evaluation
    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask
            )

            predictions = torch.argmax(outputs.logits, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, class_names)

    # Generate classification report
    report = classification_report(all_labels, all_predictions,
                                   target_names=class_names, output_dict=True)

    print("\nFinal Evaluation Results:")
    print("=" * 50)
    for cls in class_names:
        print(f"{cls}: F1={report[cls]['f1-score']:.4f}, "
              f"Precision={report[cls]['precision']:.4f}, "
              f"Recall={report[cls]['recall']:.4f}")

    print(f"\nOverall: F1={report['weighted avg']['f1-score']:.4f}, "
          f"Accuracy={report['accuracy']:.4f}")
    print("=" * 50)

    print("\nBERT Performance Summary:")
    print(f"BERT weighted F1-score: {report['weighted avg']['f1-score']:.4f}")
    print(f"BERT accuracy: {report['accuracy']:.4f}")

    return model, tokenizer, le


if __name__ == "__main__":
    model, tokenizer, label_encoder = train_model()

    # Save the model and tokenizer
    model.save_pretrained('./review_classifier')
    tokenizer.save_pretrained('./review_classifier')

    print("\nTraining and evaluation complete. Model saved to ./review_classifier")
    print("Visualization files saved: bert_training_history.png, bert_confusion_matrix.png")

"""
Epoch 1/10: 100%|██████████| 25/25 [00:17<00:00,  1.39it/s]
Epoch 1/10:
  Training Loss: 1.0102
  Validation Loss: 0.9068
  Validation Accuracy: 0.5400
  Validation F1 Score: 0.4594
  Validation Precision: 0.6300
  Validation Recall: 0.5400
Epoch 2/10: 100%|██████████| 25/25 [00:17<00:00,  1.44it/s]
Epoch 2/10:
  Training Loss: 0.8119
  Validation Loss: 0.7632
  Validation Accuracy: 0.5800
  Validation F1 Score: 0.5908
  Validation Precision: 0.6070
  Validation Recall: 0.5800
Epoch 3/10: 100%|██████████| 25/25 [00:17<00:00,  1.41it/s]
Epoch 3/10:
  Training Loss: 0.6087
  Validation Loss: 0.6122
  Validation Accuracy: 0.6400
  Validation F1 Score: 0.6300
  Validation Precision: 0.6249
  Validation Recall: 0.6400
Epoch 4/10: 100%|██████████| 25/25 [00:18<00:00,  1.38it/s]
Epoch 4/10:
  Training Loss: 0.4088
  Validation Loss: 0.5565
  Validation Accuracy: 0.6800
  Validation F1 Score: 0.6621
  Validation Precision: 0.6602
  Validation Recall: 0.6800
Epoch 5/10: 100%|██████████| 25/25 [00:18<00:00,  1.36it/s]
Epoch 5/10:
  Training Loss: 0.2826
  Validation Loss: 0.5974
  Validation Accuracy: 0.6200
  Validation F1 Score: 0.6254
  Validation Precision: 0.6394
  Validation Recall: 0.6200
Epoch 6/10: 100%|██████████| 25/25 [00:18<00:00,  1.33it/s]
Epoch 6/10:
  Training Loss: 0.2037
  Validation Loss: 0.6484
  Validation Accuracy: 0.6200
  Validation F1 Score: 0.6200
  Validation Precision: 0.6200
  Validation Recall: 0.6200
Epoch 7/10: 100%|██████████| 25/25 [00:18<00:00,  1.32it/s]
Epoch 7/10:
  Training Loss: 0.1378
  Validation Loss: 0.7243
  Validation Accuracy: 0.6200
  Validation F1 Score: 0.6200
  Validation Precision: 0.6220
  Validation Recall: 0.6200
Epoch 8/10: 100%|██████████| 25/25 [00:18<00:00,  1.35it/s]
Epoch 8/10:
  Training Loss: 0.1040
  Validation Loss: 0.7599
  Validation Accuracy: 0.6400
  Validation F1 Score: 0.6457
  Validation Precision: 0.6536
  Validation Recall: 0.6400
Epoch 9/10: 100%|██████████| 25/25 [00:18<00:00,  1.35it/s]
Epoch 9/10:
  Training Loss: 0.0814
  Validation Loss: 0.7533
  Validation Accuracy: 0.6400
  Validation F1 Score: 0.6457
  Validation Precision: 0.6536
  Validation Recall: 0.6400
Epoch 10/10: 100%|██████████| 25/25 [00:18<00:00,  1.34it/s]
Epoch 10/10:
  Training Loss: 0.0724
  Validation Loss: 0.7508
  Validation Accuracy: 0.6800
  Validation F1 Score: 0.6776
  Validation Precision: 0.6759
  Validation Recall: 0.6800

Final Evaluation Results:
==================================================
AI-Generated: F1=0.9231, Precision=0.9231, Recall=0.9231
Authentic: F1=0.6809, Precision=0.6667, Recall=0.6957
Generic: F1=0.4444, Precision=0.4615, Recall=0.4286

Overall: F1=0.6776, Accuracy=0.6800
==================================================

BERT Performance Summary:
BERT weighted F1-score: 0.6776
BERT accuracy: 0.6800
"""

"""
Epoch 1/10:
  Training Loss: 1.0376
  Validation Loss: 1.0397
  Validation Accuracy: 0.4714
  Validation F1 Score: 0.3021
  Validation Precision: 0.2222
  Validation Recall: 0.4714
Epoch 2/10: 100%|██████████| 35/35 [00:27<00:00,  1.28it/s]
/usr/local/lib/python3.11/dist-packages/sklearn/metrics/_classification.py:1565: UndefinedMetricWarning: Precision is ill-defined and being set to 0.0 in labels with no predicted samples. Use `zero_division` parameter to control this behavior.
  _warn_prf(average, modifier, f"{metric.capitalize()} is", len(result))
Epoch 2/10:
  Training Loss: 0.9662
  Validation Loss: 0.9184
  Validation Accuracy: 0.4714
  Validation F1 Score: 0.3021
  Validation Precision: 0.2222
  Validation Recall: 0.4714
Epoch 3/10: 100%|██████████| 35/35 [00:26<00:00,  1.32it/s]
Epoch 3/10:
  Training Loss: 0.7716
  Validation Loss: 0.8045
  Validation Accuracy: 0.6286
  Validation F1 Score: 0.6176
  Validation Precision: 0.6702
  Validation Recall: 0.6286
Epoch 4/10: 100%|██████████| 35/35 [00:26<00:00,  1.32it/s]
Epoch 4/10:
  Training Loss: 0.5517
  Validation Loss: 0.6839
  Validation Accuracy: 0.6714
  Validation F1 Score: 0.6273
  Validation Precision: 0.6862
  Validation Recall: 0.6714
Epoch 5/10: 100%|██████████| 35/35 [00:26<00:00,  1.31it/s]
Epoch 5/10:
  Training Loss: 0.3462
  Validation Loss: 0.7586
  Validation Accuracy: 0.6000
  Validation F1 Score: 0.5809
  Validation Precision: 0.6837
  Validation Recall: 0.6000
Epoch 6/10: 100%|██████████| 35/35 [00:26<00:00,  1.32it/s]
Epoch 6/10:
  Training Loss: 0.2532
  Validation Loss: 0.6173
  Validation Accuracy: 0.6571
  Validation F1 Score: 0.6641
  Validation Precision: 0.6809
  Validation Recall: 0.6571
Epoch 7/10: 100%|██████████| 35/35 [00:26<00:00,  1.32it/s]
Epoch 7/10:
  Training Loss: 0.1665
  Validation Loss: 0.6751
  Validation Accuracy: 0.7143
  Validation F1 Score: 0.7170
  Validation Precision: 0.7211
  Validation Recall: 0.7143
Epoch 8/10: 100%|██████████| 35/35 [00:26<00:00,  1.32it/s]
Epoch 8/10:
  Training Loss: 0.1590
  Validation Loss: 0.7906
  Validation Accuracy: 0.7000
  Validation F1 Score: 0.6886
  Validation Precision: 0.6936
  Validation Recall: 0.7000
Epoch 9/10: 100%|██████████| 35/35 [00:26<00:00,  1.33it/s]
Epoch 9/10:
  Training Loss: 0.1061
  Validation Loss: 0.7774
  Validation Accuracy: 0.7429
  Validation F1 Score: 0.7368
  Validation Precision: 0.7391
  Validation Recall: 0.7429
Epoch 10/10: 100%|██████████| 35/35 [00:26<00:00,  1.31it/s]
Epoch 10/10:
  Training Loss: 0.0899
  Validation Loss: 0.8657
  Validation Accuracy: 0.6714
  Validation F1 Score: 0.6692
  Validation Precision: 0.6693
  Validation Recall: 0.6714

Final Evaluation Results:
==================================================
AI-Generated: F1=0.8462, Precision=0.7857, Recall=0.9167
Authentic: F1=0.6875, Precision=0.7097, Recall=0.6667
Generic: F1=0.5600, Precision=0.5600, Recall=0.5600

Overall: F1=0.6692, Accuracy=0.6714
==================================================

BERT Performance Summary:
BERT weighted F1-score: 0.6692
BERT accuracy: 0.6714
"""
