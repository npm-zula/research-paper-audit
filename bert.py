import pandas as pd
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tqdm import tqdm

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
    df['Abstract'] = df['Abstract'].str.encode('ascii', 'ignore').str.decode('ascii')
    
    # Drop rows with missing values
    df = df.dropna(subset=['Abstract', 'Review Type'])
    
    # Encode labels
    le = LabelEncoder()
    labels = le.fit_transform(df['Review Type'])
    
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

        # Validation
        model.eval()
        val_accuracy = 0
        val_steps = 0
        
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
                val_accuracy += (predictions == labels).sum().item()
                val_steps += labels.size(0)

        avg_train_loss = total_loss / len(train_loader)
        avg_val_accuracy = val_accuracy / val_steps

        print(f'Average training loss: {avg_train_loss:.4f}')
        print(f'Validation Accuracy: {avg_val_accuracy:.4f}')

    return model, tokenizer, le

if __name__ == "__main__":
    model, tokenizer, label_encoder = train_model()
    
    # Save the model and tokenizer
    model.save_pretrained('./review_classifier')
    tokenizer.save_pretrained('./review_classifier')


    """
    Epoch 1/10: 100%|██████████| 25/25 [00:17<00:00,  1.42it/s]
Average training loss: 1.0102
Validation Accuracy: 0.5400
Epoch 2/10: 100%|██████████| 25/25 [00:17<00:00,  1.40it/s]
Average training loss: 0.8119
Validation Accuracy: 0.5800
Epoch 3/10: 100%|██████████| 25/25 [00:18<00:00,  1.37it/s]
Average training loss: 0.6088
Validation Accuracy: 0.6400
Epoch 4/10: 100%|██████████| 25/25 [00:18<00:00,  1.36it/s]
Average training loss: 0.4091
Validation Accuracy: 0.6800
Epoch 5/10: 100%|██████████| 25/25 [00:18<00:00,  1.37it/s]
Average training loss: 0.2837
Validation Accuracy: 0.6400
Epoch 6/10: 100%|██████████| 25/25 [00:18<00:00,  1.38it/s]
Average training loss: 0.2050
Validation Accuracy: 0.6400
Epoch 7/10: 100%|██████████| 25/25 [00:18<00:00,  1.37it/s]
Average training loss: 0.1334
Validation Accuracy: 0.6200
Epoch 8/10: 100%|██████████| 25/25 [00:18<00:00,  1.37it/s]
Average training loss: 0.1043
Validation Accuracy: 0.6400
Epoch 9/10: 100%|██████████| 25/25 [00:18<00:00,  1.37it/s]
Average training loss: 0.0824
Validation Accuracy: 0.6400
Epoch 10/10: 100%|██████████| 25/25 [00:18<00:00,  1.38it/s]
Average training loss: 0.0720
Validation Accuracy: 0.6600
    """