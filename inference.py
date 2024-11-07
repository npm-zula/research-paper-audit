import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np

class ReviewClassifier:
    def __init__(self, model_path='./review_classifier'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model = BertForSequenceClassification.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Define label mapping
        self.labels = ['AI-Generated', 'Generic', 'Authentic']

    def predict(self, text, max_length=512):
        # Prepare the text
        encoding = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        # Move to device
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        # Get prediction
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            predictions = torch.softmax(outputs.logits, dim=1)
            predicted_class = torch.argmax(predictions, dim=1)
            confidence_scores = predictions[0].cpu().numpy()

        return {
            'predicted_class': self.labels[predicted_class.item()],
            'confidence_scores': {
                label: float(score) 
                for label, score in zip(self.labels, confidence_scores)
            }
        }

def main():
    # Initialize classifier
    classifier = ReviewClassifier()

    # Example texts to classify
    example_texts = [
        """This paper introduces a groundbreaking approach to natural language processing, 
        leveraging advanced neural architectures to achieve state-of-the-art results on 
        multiple benchmarks. The proposed method demonstrates significant improvements 
        over existing baselines.""",
        
        """The manuscript needs significant revision. The methodology section lacks detail 
        and the results are not properly discussed. The authors should provide more 
        statistical analysis to support their conclusions.""",
        
        """This is a well-executed study with robust methodology and clear presentation 
        of results. The authors have thoroughly addressed all potential confounding 
        factors and provided comprehensive statistical analysis."""
    ]

    # Make predictions
    print("Making predictions...\n")
    for i, text in enumerate(example_texts, 1):
        result = classifier.predict(text)
        print(f"Example {i}:")
        print(f"Text: {text[:100]}...")
        print(f"Predicted Class: {result['predicted_class']}")
        print("\nConfidence Scores:")
        for label, score in result['confidence_scores'].items():
            print(f"{label}: {score:.4f}")
        print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    main()
