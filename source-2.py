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
from datasets import Dataset, ClassLabel
import evaluate
import torch

# Configuration
MODEL_NAME = "distilbert-base-uncased"  # Efficient baseline model
MAX_LENGTH = 512  # Max context length for transformer models
BATCH_SIZE = 16
LEARNING_RATE = 2e-5
NUM_EPOCHS = 10

# Load and preprocess data
def load_data(file_path):
    df = pd.read_csv(file_path, encoding='cp1252')
    
    # Handle missing values
    df = df.fillna({'Paper Title': '', 'Abstract': '', 'Review Text': ''})
    
    # Combine text features
    df['text'] = (
        "Title: " + df['Paper Title'] + 
        " Abstract: " + df['Abstract'] + 
        " Review: " + df['Review Text']
    )
    
    # Encode labels
    label_encoder = LabelEncoder()
    df['label'] = label_encoder.fit_transform(df['Review Type'])
    
    return df[['text', 'label']], label_encoder.classes_

# Load your dataset
df, class_names = load_data("/content/corpus.csv")

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

# Metrics
accuracy_metric = evaluate.load("accuracy")
f1_metric = evaluate.load("f1")

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    accuracy = accuracy_metric.compute(predictions=predictions, references=labels)["accuracy"]
    f1 = f1_metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    return {"accuracy": accuracy, "f1": f1}

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
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
    report_to="none",
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
trainer.train()

# Evaluate on test set
test_results = trainer.evaluate(test_dataset)
print("Test results:", test_results)

# Save model
model.save_pretrained("./final_model")
tokenizer.save_pretrained("./final_model")

# Example inference
def predict(text, model, tokenizer):
    inputs = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class_idx = torch.argmax(probabilities).item()
    
    return model.config.id2label[predicted_class_idx], probabilities.tolist()[0]