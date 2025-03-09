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
    get_scheduler,
)
from datasets import Dataset
import evaluate
from tqdm.auto import tqdm
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers.utils import logging
import random
from captum.attr import LayerIntegratedGradients

# Set up logging
logging.set_verbosity_info()

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Configuration
MODEL_NAME = "roberta-base"  # RoBERTa base model
MAX_LENGTH = 512  # Max context length for transformer models
BATCH_SIZE = 8  # Smaller batch size with gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 4  # Effectively gives us batch size of 32
LEARNING_RATE = 3e-5  # Slightly higher learning rate with scheduler
NUM_EPOCHS = 15  # More epochs with early stopping
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1  # Warm up for first 10% of training steps
CROSS_VAL_FOLDS = 5  # Number of cross-validation folds

# Function to load and preprocess data with robust encoding handling
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
        # Last resort: try with error handling
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
    
    # Handle missing values more robustly
    for col in ['Paper Title', 'Abstract', 'Review Text', 'Review Type']:
        if col in df.columns:
            df[col] = df[col].fillna('')
    
    # Clean text data - remove non-ASCII characters
    for col in ['Paper Title', 'Abstract', 'Review Text']:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii'))
    
    # Create weighted text combination with special tokens to help model distinguish sections
    df['text'] = (
        "<title>" + df['Paper Title'] + "</title> " + 
        "<abstract>" + df['Abstract'] + "</abstract> " + 
        "<review>" + df['Review Text'] + "</review>"
    )
    
    # Drop rows with missing values in critical columns
    df = df.dropna(subset=['text', 'Review Type'])
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Review Type'])
    
    return df, label_encoder.classes_

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
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['eval_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history['eval_accuracy'], label='Accuracy')
    plt.plot(history['eval_f1'], label='F1 Score')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    plt.close()

# Function to plot confusion matrix
def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.close()

# Model explainability function using Integrated Gradients
def explain_prediction(text, model, tokenizer, class_names, device):
    # Tokenize input
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=MAX_LENGTH)
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
    df, class_names = load_data('corpus.csv')
    print(f"Loaded {len(df)} samples with {len(class_names)} classes: {class_names}")
    
    # Initialize tokenizer
    tokenizer = RobertaTokenizer.from_pretrained(MODEL_NAME)
    
    # Setup cross-validation
    skf = StratifiedKFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=SEED)
    
    # Store results for each fold
    fold_results = []
    best_model = None
    best_tokenizer = None
    best_f1 = 0.0
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n{'='*50}\nTraining Fold {fold+1}/{CROSS_VAL_FOLDS}\n{'='*50}")
        
        # Split data for this fold
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        # Create Hugging Face datasets
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        # Tokenize datasets
        train_dataset = train_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True
        )
        val_dataset = val_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True
        )
        
        # Set format for pytorch
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        # Load model with proper class labels
        model = RobertaForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(class_names),
            id2label={i: label for i, label in enumerate(class_names)},
            label2id={label: i for i, label in enumerate(class_names)}
        )
        model.to(device)
        
        # Calculate training steps for scheduler
        num_training_steps = len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
        num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
        
        # Training arguments with learning rate scheduler
        training_args = TrainingArguments(
            output_dir=f"./results/fold-{fold+1}",
            evaluation_strategy="epoch",
            save_strategy="epoch",
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=torch.cuda.is_available(),
            logging_dir=f"./logs/fold-{fold+1}",
            logging_strategy="steps",
            logging_steps=10,
            report_to="tensorboard",
            warmup_steps=num_warmup_steps,
            seed=SEED
        )
        
        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
        )
        
        # Train model
        print("Training model...")
        trainer.train()
        
        # Evaluate on validation set
        print("Evaluating model...")
        eval_results = trainer.evaluate()
        fold_results.append(eval_results)
        
        print(f"Fold {fold+1} results:")
        for metric, value in eval_results.items():
            print(f"{metric}: {value:.4f}")
        
        # Save best model across folds
        if eval_results['eval_f1'] > best_f1:
            best_f1 = eval_results['eval_f1']
            best_model = model
            best_tokenizer = tokenizer
            print(f"New best model found with F1: {best_f1:.4f}")
    
    # Calculate average metrics across folds
    avg_results = {}
    for metric in fold_results[0].keys():
        avg_results[metric] = sum(fold[metric] for fold in fold_results) / len(fold_results)
    
    print("\nAverage results across all folds:")
    for metric, value in avg_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Save best model
    print("\nSaving best model...")
    best_model.save_pretrained("./roberta_classifier")
    best_tokenizer.save_pretrained("./roberta_classifier")
    
    return best_model, best_tokenizer, class_names

# Prediction function with confidence scores
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
    confidence_scores = {class_names[i]: float(probabilities[0][i]) for i in range(len(class_names))}
    
    return {
        "prediction": predicted_class,
        "confidence": float(probabilities[0][predicted_class_idx]),
        "all_scores": confidence_scores
    }

# Function to evaluate model on test set and generate detailed report
def evaluate_model(model, tokenizer, test_df, class_names, device=None):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model.to(device)
    model.eval()
    
    # Create test dataset
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Create dataloader
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2)
    
    # Collect predictions
    all_predictions = []
    all_labels = []
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        
        predictions = torch.argmax(outputs.logits, dim=1)
        
        all_predictions.extend(predictions.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    # Generate classification report
    report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
    
    # Plot confusion matrix
    plot_confusion_matrix(all_labels, all_predictions, class_names)
    
    return report

# Main execution
if __name__ == "__main__":
    print(f"Starting RoBERTa model training with {CROSS_VAL_FOLDS}-fold cross-validation")
    print(f"Model: {MODEL_NAME}, Max Length: {MAX_LENGTH}, Batch Size: {BATCH_SIZE}")
    print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning Rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}")
    
    # Train model with cross-validation
    model, tokenizer, class_names = train_model_with_cv()
    
    # Load test data for final evaluation
    print("\nPerforming final evaluation on test set...")
    test_df, _ = load_data('corpus.csv')
    
    # Split off a test set (20% of data)
    _, test_df = train_test_split(test_df, test_size=0.2, stratify=test_df['label'], random_state=SEED)
    
    # Evaluate on test set
    test_report = evaluate_model(model, tokenizer, test_df, class_names)
    
    print("\nTest Set Performance:")
    for cls in class_names:
        print(f"{cls}: F1={test_report[cls]['f1-score']:.4f}, Precision={test_report[cls]['precision']:.4f}, Recall={test_report[cls]['recall']:.4f}")
    
    print(f"\nOverall: F1={test_report['weighted avg']['f1-score']:.4f}, Accuracy={test_report['accuracy']:.4f}")
    
    # Example of model explainability
    print("\nModel Explainability Example:")
    example_text = "<title>Advances in Natural Language Processing</title> <abstract>This paper presents a novel approach to sentiment analysis using transformer models.</abstract> <review>The methodology is sound but lacks comparison with recent work.</review>"
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    explanation = explain_prediction(example_text, model, tokenizer, class_names, device)
    
    print(f"Prediction: {explanation['prediction']}")
    print(f"Confidence: {explanation['confidence']:.4f}")
    print("Top influential words:")
    for word, importance in explanation['top_words'][:10]:
        print(f"  {word}: {importance:.4f}")
    
    print("\nTraining and evaluation complete. Model saved to ./roberta_classifier")
    print("Visualization files saved: training_history.png, confusion_matrix.png")


    """"
    Average results across all folds:
eval_loss: 0.6725
eval_accuracy: 0.7510
eval_f1: 0.7509
eval_precision: 0.7543
eval_recall: 0.7510
eval_runtime: 0.6929
eval_samples_per_second: 119.8550
eval_steps_per_second: 8.6643
epoch: 11.8571
    """