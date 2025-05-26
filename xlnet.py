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
MODEL_NAME = "xlnet-base-cased"  # XLNet base model
MAX_LENGTH = 512  # Max context length
BATCH_SIZE = 8  # Smaller batch size with gradient accumulation
GRADIENT_ACCUMULATION_STEPS = 4  # Effectively gives us batch size of 32
LEARNING_RATE = 1.5e-5  # Adjusted learning rate
NUM_EPOCHS = 10  # Adjusted number of epochs
WEIGHT_DECAY = 0.05  # Adjusted weight decay
WARMUP_RATIO = 0.1
CROSS_VAL_FOLDS = 3

# Focal Loss implementation for handling class imbalance
class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

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
    for col in ['Paper Title', 'Abstract', 'Review Text', 'Review Type']:
        if col in df.columns:
            df[col] = df[col].fillna('')
            if col != 'Review Type':
                df[col] = df[col].apply(lambda x: str(x).encode('ascii', 'ignore').decode('ascii'))
    
    # Create weighted text combination with section importance weights
    df['text'] = (
        "[TITLE] " + df['Paper Title'] + " [/TITLE] " +
        "[ABSTRACT] " + df['Abstract'] + " [/ABSTRACT] " +
        "[REVIEW] " + df['Review Text'] + " [/REVIEW]"
    )
    
    df = df.dropna(subset=['text', 'Review Type'])
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Review Type'])
    
    return df, label_encoder.classes_

def tokenize_function(examples, tokenizer):
    return tokenizer(
        examples['text'],
        padding="max_length",
        max_length=MAX_LENGTH,
        truncation=True,
        return_tensors="pt"
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

def process_log_history_for_plotting(log_history):
    """
    Processes the trainer.state.log_history to extract epoch-wise metrics
    for plotting.
    """
    processed_history = {
        'train_epochs': [], 'train_loss': [],
        'eval_epochs': [], 'eval_loss': [], 'eval_accuracy': [], 'eval_f1': [],
        'eval_precision': [], 'eval_recall': []
    }
    temp_train_losses_for_epoch = {}  # epoch_float -> list of losses

    for log in log_history:
        epoch = log.get('epoch')
        if epoch is None:
            continue

        # Check if it's a training log (has 'loss' but not 'eval_loss')
        if 'loss' in log and 'eval_loss' not in log:
            if epoch not in temp_train_losses_for_epoch:
                temp_train_losses_for_epoch[epoch] = []
            temp_train_losses_for_epoch[epoch].append(log['loss'])
        
        # Check if it's an evaluation log (has 'eval_loss')
        elif 'eval_loss' in log:
            processed_history['eval_epochs'].append(epoch)
            processed_history['eval_loss'].append(log['eval_loss'])
            processed_history['eval_accuracy'].append(log.get('eval_accuracy')) # Use .get for safety
            processed_history['eval_f1'].append(log.get('eval_f1'))
            processed_history['eval_precision'].append(log.get('eval_precision'))
            processed_history['eval_recall'].append(log.get('eval_recall'))

    # Average training losses for each epoch it was logged
    # Sort epochs to ensure chronological order for plotting training loss
    sorted_train_epochs = sorted(temp_train_losses_for_epoch.keys())
    for epoch in sorted_train_epochs:
        if temp_train_losses_for_epoch[epoch]: # Ensure there are losses to average
            processed_history['train_epochs'].append(epoch)
            processed_history['train_loss'].append(np.mean(temp_train_losses_for_epoch[epoch]))
            
    # Ensure all eval metrics lists are of the same length as eval_epochs
    # This handles cases where a metric might be missing from a log entry
    # (though unlikely with Hugging Face Trainer standard logging)
    len_eval_epochs = len(processed_history['eval_epochs'])
    for key in ['eval_loss', 'eval_accuracy', 'eval_f1', 'eval_precision', 'eval_recall']:
        if len(processed_history[key]) != len_eval_epochs:
            # This case should ideally not happen with standard trainer logs.
            # If it does, it indicates an issue with log contents or parsing.
            # For robustness, one might pad with np.nan or filter,
            # but for now, we assume consistent logging from Trainer.
            pass # Or print a warning

    return processed_history

def plot_training_history(history):
    plt.figure(figsize=(20, 6))  # Adjusted figure size
    
    plt.subplot(1, 3, 1)
    if history['train_epochs'] and history['train_loss']: # Check if data exists
        plt.plot(history['train_epochs'], history['train_loss'], label='Training Loss', marker='o')
    if history['eval_epochs'] and history['eval_loss']: # Check if data exists
        plt.plot(history['eval_epochs'], history['eval_loss'], label='Validation Loss', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 3, 2)
    if history['eval_epochs'] and history['eval_accuracy']: # Check if data exists
        plt.plot(history['eval_epochs'], history['eval_accuracy'], label='Accuracy', marker='o')
    if history['eval_epochs'] and history['eval_f1']: # Check if data exists
        plt.plot(history['eval_epochs'], history['eval_f1'], label='F1 Score', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Validation Metrics')
    
    plt.subplot(1, 3, 3)
    if history['eval_epochs'] and history['eval_precision']: # Check if data exists
        plt.plot(history['eval_epochs'], history['eval_precision'], label='Precision', marker='o')
    if history['eval_epochs'] and history['eval_recall']: # Check if data exists
        plt.plot(history['eval_epochs'], history['eval_recall'], label='Recall', marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Score')
    plt.legend()
    plt.title('Precision and Recall')
    
    plt.tight_layout()
    plt.savefig('xlnet_training_history_corrected.png') # Changed filename
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig('xlnet_confusion_matrix.png')
    plt.close()

def explain_prediction(text, model, tokenizer, class_names, device):
    inputs = tokenizer(text, return_tensors="pt", padding="max_length",
                      truncation=True, max_length=MAX_LENGTH)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = F.softmax(outputs.logits, dim=-1)
    predicted_class_idx = torch.argmax(probabilities, dim=1).item()
    predicted_class = class_names[predicted_class_idx]
    
    # Setup for attribution
    lig = LayerIntegratedGradients(model, model.transformer.word_embedding)
    
    attributions, delta = lig.attribute(
        inputs=inputs['input_ids'],
        baselines=torch.zeros_like(inputs['input_ids']),
        target=predicted_class_idx,
        return_convergence_delta=True,
        n_steps=50
    )
    
    tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
    attributions = attributions.sum(dim=-1).squeeze(0)
    attributions = attributions / torch.norm(attributions)
    attributions = attributions.cpu().detach().numpy()
    
    word_importances = []
    for token, importance in zip(tokens, attributions):
        if token not in [tokenizer.pad_token, tokenizer.cls_token, tokenizer.sep_token]:
            word_importances.append((token, float(importance)))
    
    word_importances.sort(key=lambda x: abs(x[1]), reverse=True)
    
    return {
        "prediction": predicted_class,
        "confidence": float(probabilities[0][predicted_class_idx]),
        "top_words": word_importances[:20]
    }

def train_model_with_cv():
    print("Loading data...")
    df, class_names = load_data('/corpus/corpus-final.csv') # Updated dataset path to be absolute
    print(f"Loaded {len(df)} samples with {len(class_names)} classes: {class_names}")
    
    tokenizer = XLNetTokenizer.from_pretrained(MODEL_NAME)
    
    skf = StratifiedKFold(n_splits=CROSS_VAL_FOLDS, shuffle=True, random_state=SEED)
    
    fold_results = []
    best_model_state_dict = None # Store state_dict instead of the whole model object for memory efficiency
    best_tokenizer = None
    best_f1 = 0.0
    best_fold_log_history = None # To store log_history of the best fold
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Store training history (not used directly for plotting anymore, but kept for averages)
    aggregated_training_history = {
        'train_loss': [],
        'eval_loss': [],
        'eval_accuracy': [],
        'eval_f1': [],
        'eval_precision': [],
        'eval_recall': []
    }
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(df, df['label'])):
        print(f"\n{'='*50}\nTraining Fold {fold+1}/{CROSS_VAL_FOLDS}\n{'='*50}")
        
        train_df = df.iloc[train_idx].reset_index(drop=True)
        val_df = df.iloc[val_idx].reset_index(drop=True)
        
        train_dataset = Dataset.from_pandas(train_df)
        val_dataset = Dataset.from_pandas(val_df)
        
        train_dataset = train_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True
        )
        val_dataset = val_dataset.map(
            lambda examples: tokenize_function(examples, tokenizer),
            batched=True
        )
        
        train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
        
        model = XLNetForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(class_names),
            id2label={i: label for i, label in enumerate(class_names)},
            label2id={label: i for i, label in enumerate(class_names)}
        )
        model.to(device)
        
        num_training_steps = len(train_dataset) * NUM_EPOCHS // (BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS)
        num_warmup_steps = int(num_training_steps * WARMUP_RATIO)
        
        training_args = TrainingArguments(
            output_dir=f"./results/xlnet-fold-{fold+1}",
            evaluation_strategy="steps", # Was eval_strategy, ensuring correct param name and value
            save_strategy="steps",       # Match evaluation_strategy
            eval_steps=10,             # Match logging_steps for frequent evaluation
            save_steps=10,             # Match eval_steps
            learning_rate=LEARNING_RATE,
            per_device_train_batch_size=BATCH_SIZE,
            per_device_eval_batch_size=BATCH_SIZE * 2,
            gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
            num_train_epochs=NUM_EPOCHS,
            weight_decay=WEIGHT_DECAY,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            fp16=torch.cuda.is_available(),
            logging_dir=f"./logs/xlnet-fold-{fold+1}",
            logging_strategy="steps",
            logging_steps=10,
            report_to="tensorboard",
            warmup_steps=num_warmup_steps,
            seed=SEED,
            save_total_limit=3  # Limit the number of checkpoints saved
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            compute_metrics=compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)], # Adjusted patience
        )
        
        print("Training model...")
        train_result = trainer.train()
        eval_result = trainer.evaluate() # This re-evaluates the best model loaded by EarlyStopping
        
        # Store results for this fold
        fold_results.append(eval_result)
        
        # Update aggregated training history (for average reporting)
        # Note: train_result.training_loss is the average training loss over all epochs for this fold.
        # For detailed epoch-wise loss, we use trainer.state.log_history below.
        aggregated_training_history['train_loss'].append(train_result.training_loss if train_result.training_loss is not None else np.nan)
        aggregated_training_history['eval_loss'].append(eval_result['eval_loss'])
        aggregated_training_history['eval_accuracy'].append(eval_result['eval_accuracy'])
        aggregated_training_history['eval_f1'].append(eval_result['eval_f1'])
        aggregated_training_history['eval_precision'].append(eval_result['eval_precision'])
        aggregated_training_history['eval_recall'].append(eval_result['eval_recall'])
        
        print(f"Fold {fold+1} results:")
        for metric, value in eval_result.items():
            print(f"{metric}: {value:.4f}")
        
        if eval_result['eval_f1'] > best_f1:
            best_f1 = eval_result['eval_f1']
            # Save model state_dict to save memory, especially in Colab
            best_model_state_dict = model.state_dict()
            best_tokenizer = tokenizer
            best_fold_log_history = trainer.state.log_history # Capture log history for the best fold
            print(f"New best model found with F1: {best_f1:.4f} in fold {fold+1}")
    
    avg_results = {}
    for metric in fold_results[0].keys():
        avg_results[metric] = sum(fold[metric] for fold in fold_results) / len(fold_results)
    
    print("\nAverage results across all folds:")
    for metric, value in avg_results.items():
        print(f"{metric}: {value:.4f}")
    
    # Reconstruct the best model from state_dict
    if best_model_state_dict:
        print("\nLoading best model from state_dict...")
        # Need to initialize a model instance first
        final_model = XLNetForSequenceClassification.from_pretrained(
            MODEL_NAME,
            num_labels=len(class_names),
            id2label={i: label for i, label in enumerate(class_names)},
            label2id={label: i for i, label in enumerate(class_names)}
        )
        final_model.load_state_dict(best_model_state_dict)
        final_model.to(device) # Ensure it's on the correct device
        print("Saving best model...")
        final_model.save_pretrained("./xlnet_classifier")
        best_tokenizer.save_pretrained("./xlnet_classifier")
    else:
        print("No best model state_dict found. Training might have failed or produced no improvements.")
        return None, None, class_names # Or handle error appropriately

    # Plot training history using the logs from the best fold
    if best_fold_log_history:
        print("Processing log history for plotting...")
        plot_data = process_log_history_for_plotting(best_fold_log_history)
        print("Plotting training history of the best fold...")
        plot_training_history(plot_data)
    else:
        print("No log history found for the best fold to plot.")
        
    return final_model, best_tokenizer, class_names

if __name__ == "__main__":
    print(f"Starting XLNet model training with {CROSS_VAL_FOLDS}-fold cross-validation")
    print(f"Model: {MODEL_NAME}, Max Length: {MAX_LENGTH}, Batch Size: {BATCH_SIZE}")
    print(f"Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"Learning Rate: {LEARNING_RATE}, Epochs: {NUM_EPOCHS}")
    
    model, tokenizer, class_names = train_model_with_cv()
    
    if model is None or tokenizer is None:
        print("Model training failed or no best model was identified. Exiting.")
        # Consider exiting or raising an error
        exit()
        
    print("\nPerforming final evaluation on test set...")
    test_df, _ = load_data('/content/corpus-final.csv') # Updated dataset path
    _, test_df = train_test_split(test_df, test_size=0.2, stratify=test_df['label'],
                                 random_state=SEED)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    test_dataset = Dataset.from_pandas(test_df)
    test_dataset = test_dataset.map(
        lambda examples: tokenize_function(examples, tokenizer),
        batched=True
    )
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    
    # Evaluate on test set
    def evaluate_model(model, tokenizer, test_dataset, class_names, device):
        model.to(device)
        model.eval()
        
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE * 2)
        
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
        
        report = classification_report(all_labels, all_predictions, target_names=class_names, output_dict=True)
        
        # Plot confusion matrix
        plot_confusion_matrix(all_labels, all_predictions, class_names)
        
        return report
    
    test_report = evaluate_model(model, tokenizer, test_dataset, class_names, device)
    
    print("\nTest Set Performance:")
    for cls in class_names:
        print(f"{cls}: F1={test_report[cls]['f1-score']:.4f}, Precision={test_report[cls]['precision']:.4f}, Recall={test_report[cls]['recall']:.4f}")
    
    print(f"\nOverall: F1={test_report['weighted avg']['f1-score']:.4f}, Accuracy={test_report['accuracy']:.4f}")
    
    # Example of model explainability
    print("\nModel Explainability Example:")
    example_text = "[TITLE] Advances in Natural Language Processing [/TITLE] [ABSTRACT] This paper presents a novel approach to sentiment analysis using transformer models. [/ABSTRACT] [REVIEW] The methodology is sound but lacks comparison with recent work. [/REVIEW]"
    
    explanation = explain_prediction(example_text, model, tokenizer, class_names, device)
    
    print(f"Prediction: {explanation['prediction']}")
    print(f"Confidence: {explanation['confidence']:.4f}")
    print("Top influential words:")
    for word, importance in explanation['top_words'][:10]:
        print(f"  {word}: {importance:.4f}")
    
    print("\nTraining and evaluation complete. Model saved to ./xlnet_classifier")
    print("Visualization files saved: xlnet_training_history_corrected.png, xlnet_confusion_matrix.png")
    
    # Compare with RoBERTa results
    print("\nXLNet vs RoBERTa Performance Comparison:")
    print("XLNet weighted F1-score: {:.4f}".format(test_report['weighted avg']['f1-score']))
    print("XLNet accuracy: {:.4f}".format(test_report['accuracy']))
    print("\nNote: XLNet's permutation-based language modeling may provide better context understanding")
    print("for complex document classification tasks compared to RoBERTa's masked language modeling.")
    print("XLNet is particularly effective at capturing bidirectional context and long-range dependencies.")

    """
    Average results across all folds:
eval_loss: 0.5362
eval_accuracy: 0.7671
eval_f1: 0.7664
eval_precision: 0.7669
eval_recall: 0.7671
eval_runtime: 6.6263
eval_samples_per_second: 12.5257
eval_steps_per_second: 0.9053
epoch: 10.5238
    """

    """
    Test Set Performance:
AI-Generated: F1=0.9091, Precision=1.0000, Recall=0.8333
Authentic: F1=0.7500, Precision=0.6818, Recall=0.8333
Generic: F1=0.4737, Precision=0.5625, Recall=0.4091

Overall: F1=0.6904, Accuracy=0.7000

    """