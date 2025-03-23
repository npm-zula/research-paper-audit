import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback,
)
from datasets import Dataset
import evaluate
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support
from tqdm.auto import tqdm

# Configuration
MODEL_NAME = "distilbert-base-uncased"  # Efficient baseline model
MAX_LENGTH = 512  # Max context length for transformer models
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('distilbert_confusion_matrix.png')
    plt.close()

def plot_training_history(history):
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
    plt.savefig('distilbert_training_history.png')
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
        df[col] = df[col].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii'))
    
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
    
    return df[['text', 'label']], label_encoder.classes_

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
    print(f"Starting DistilBERT model training with model: {MODEL_NAME}")
    print(f"Max Length: {MAX_LENGTH}, Batch Size: {BATCH_SIZE}")
    print(f"Learning Rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}")

    # Load your dataset
    print("Loading and preprocessing data...")
    df, class_names = load_data("corpus.csv")
    print(f"Loaded {len(df)} samples with {len(class_names)} classes: {class_names}")

    # Split data
    train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['label'], random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['label'], random_state=42)

    # Create Hugging Face datasets
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    # Initialize tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'],
            padding="max_length",
            max_length=MAX_LENGTH,
            truncation=True,
        )

    # Tokenize datasets
    print("Tokenizing datasets...")
    train_dataset = train_dataset.map(tokenize_function, batched=True)
    val_dataset = val_dataset.map(tokenize_function, batched=True)
    test_dataset = test_dataset.map(tokenize_function, batched=True)

    # Convert to torch format
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Load model with proper class labels
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(class_names),
        id2label={i: label for i, label in enumerate(class_names)},
        label2id={label: i for i, label in enumerate(class_names)}
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir="./distilbert_results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=LEARNING_RATE,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=NUM_EPOCHS,
        weight_decay=0.01,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        fp16=torch.cuda.is_available(),
        logging_dir="./logs",
        logging_strategy="steps",
        logging_steps=10,
        report_to="tensorboard",
    )

    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)],
    )

    # Train model
    print("Starting training...")
    train_results = trainer.train()
    
    # Get training history
    history = {
        'train_loss': [],
        'eval_loss': [],
        'eval_accuracy': [],
        'eval_f1': [],
        'eval_precision': [],
        'eval_recall': []
    }
    
    for log in trainer.state.log_history:
        if 'loss' in log and 'epoch' in log and 'eval_loss' not in log:
            history['train_loss'].append(log['loss'])
        elif 'eval_loss' in log:
            history['eval_loss'].append(log['eval_loss'])
            history['eval_accuracy'].append(log['eval_accuracy'])
            history['eval_f1'].append(log['eval_f1'])
            history['eval_precision'].append(log['eval_precision'])
            history['eval_recall'].append(log['eval_recall'])
            
            # Print epoch results
            epoch = int(log['epoch'])
            print(f"Epoch {epoch}:")
            print(f"  Training Loss: {history['train_loss'][-1]:.4f}")
            print(f"  Validation Loss: {log['eval_loss']:.4f}")
            print(f"  Validation Accuracy: {log['eval_accuracy']:.4f}")
            print(f"  Validation F1 Score: {log['eval_f1']:.4f}")
            print(f"  Validation Precision: {log['eval_precision']:.4f}")
            print(f"  Validation Recall: {log['eval_recall']:.4f}")
    
    # Plot training history
    plot_training_history(history)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.evaluate(test_dataset)
    print("Test results:")
    for metric, value in test_results.items():
        if metric.startswith('eval_'):
            print(f"  {metric[5:]}: {value:.4f}")
    
    # Get detailed metrics by class
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(trainer.get_eval_dataloader(test_dataset), desc="Detailed evaluation"):
        batch = {k: v.to(device) for k, v in batch.items()}
        
        with torch.no_grad():
            outputs = model(**batch)
        
        predictions = torch.argmax(outputs.logits, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(batch["labels"].cpu().numpy())
    
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
    
    print("\nDistilBERT Performance Summary:")
    print(f"DistilBERT weighted F1-score: {report['weighted avg']['f1-score']:.4f}")
    print(f"DistilBERT accuracy: {report['accuracy']:.4f}")

    # Save model
    model.save_pretrained("./distilbert_classifier")
    tokenizer.save_pretrained("./distilbert_classifier")

    print("\nTraining and evaluation complete. Model saved to ./distilbert_classifier")
    print("Visualization files saved: distilbert_training_history.png, distilbert_confusion_matrix.png")

if __name__ == "__main__":
    main()


"""
Epoch 1:
  Training Loss: 1.0547
  Validation Loss: 0.9651
  Validation Accuracy: 0.4595
  Validation F1 Score: 0.2893
  Validation Precision: 0.2111
  Validation Recall: 0.4595
Epoch 2:
  Training Loss: 0.9204
  Validation Loss: 0.8673
  Validation Accuracy: 0.5405
  Validation F1 Score: 0.4509
  Validation Precision: 0.5290
  Validation Recall: 0.5405
Epoch 3:
  Training Loss: 0.8306
  Validation Loss: 0.7360
  Validation Accuracy: 0.7027
  Validation F1 Score: 0.6824
  Validation Precision: 0.6914
  Validation Recall: 0.7027
Epoch 4:
  Training Loss: 0.6832
  Validation Loss: 0.6586
  Validation Accuracy: 0.7027
  Validation F1 Score: 0.6917
  Validation Precision: 0.6907
  Validation Recall: 0.7027
Epoch 5:
  Training Loss: 0.5729
  Validation Loss: 0.6159
  Validation Accuracy: 0.6757
  Validation F1 Score: 0.6721
  Validation Precision: 0.6706
  Validation Recall: 0.6757
Epoch 6:
  Training Loss: 0.4728
  Validation Loss: 0.6530
  Validation Accuracy: 0.7027
  Validation F1 Score: 0.7010
  Validation Precision: 0.7166
  Validation Recall: 0.7027
Epoch 7:
  Training Loss: 0.4226
  Validation Loss: 0.6529
  Validation Accuracy: 0.6757
  Validation F1 Score: 0.6726
  Validation Precision: 0.6961
  Validation Recall: 0.6757
Epoch 8:
  Training Loss: 0.3618
  Validation Loss: 0.5780
  Validation Accuracy: 0.7027
  Validation F1 Score: 0.6977
  Validation Precision: 0.6948
  Validation Recall: 0.7027

Evaluating on test set...
 [3/3 00:00]
Test results:
  loss: 0.5684
  accuracy: 0.7632
  f1: 0.7639
  precision: 0.7758
  recall: 0.7632
  runtime: 0.2014
  samples_per_second: 188.6570
  steps_per_second: 14.8940
Detailed evaluation: 100%
 3/3 [00:00<00:00, 15.03it/s]

Final Evaluation Results:
==================================================
AI-Generated: F1=1.0000, Precision=1.0000, Recall=1.0000
Authentic: F1=0.7097, Precision=0.7857, Recall=0.6471
Generic: F1=0.6897, Precision=0.6250, Recall=0.7692

Overall: F1=0.7639, Accuracy=0.7632
==================================================

DistilBERT Performance Summary:
DistilBERT weighted F1-score: 0.7639
DistilBERT accuracy: 0.7632

Training and evaluation complete. Model saved to ./distilbert_classifier
Visualization files saved: distilbert_training_history.png, distilbert_confusion_matrix.png
\nEpoch   Training Loss   Validation Loss Accuracy    F1\n1   1.083400    1.009396    0.459459    0.289289\n2   0.977400    0.909536    0.459459    0.289289\n3   0.893200    0.767690    0.675676    0.644673\n4   0.744200    0.663561    0.756757    0.729066\n5   0.623800    0.605594    0.702703    0.682436\n6   0.527500    0.597340    0.648649    0.642523\n [3/3 00:00]\nTest results: {'eval_loss': 0.7004138827323914, 'eval_accuracy': 0.6842105263157895, 'eval_f1': 0.6605263157894736, 'eval_runtime': 0.2194, 'eval_samples_per_second': 173.2, 'eval_steps_per_second': 13.674, 'epoch': 6.0}\n
"""