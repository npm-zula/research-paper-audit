import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import numpy as np

# Load the dataset
file_path = 'corpus-thefinal.csv'
encodings = ['cp1252', 'latin-1', 'utf-8']
df = None
for enc in encodings:
    try:
        df = pd.read_csv(file_path, encoding=enc)
        print(f"Loaded data using {enc} encoding.")
        break
    except UnicodeDecodeError:
        print(f"Failed with {enc}. Trying next encoding.")
if df is None:
    df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
    print("Loaded with utf-8 (errors replaced).")

# Inspect the data
print(f"\nDataset Shape: {df.shape}")
print(f"\nTotal rows in dataset: {len(df)}")
print(f"\nColumns: {df.columns.tolist()}")
print("\nFirst 5 rows:")
print(df.head())

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

# Fill missing text fields (if any)
for col in ['Paper Title', 'Abstract', 'Review Text']:
    if col in df.columns:
        df[col] = df[col].fillna('').astype(str)

# Combine text fields for analysis
df['full_text'] = df['Paper Title'] + ' ' + \
    df['Abstract'] + ' ' + df['Review Text']

# Review Type distribution
if 'Review Type' in df.columns:
    print("\nNumber of rows for each Review Type:")
    review_type_counts = df['Review Type'].value_counts()
    print(review_type_counts)

    plt.figure(figsize=(8, 6))
    sns.countplot(y='Review Type', data=df, order=review_type_counts.index)
    plt.title('Distribution of Review Types')
    plt.xlabel('Count')
    plt.ylabel('Review Type')
    plt.tight_layout()
    plt.show()
else:
    print("'Review Type' column not found.")

# Generate WordClouds


def generate_wordcloud(text, title):
    wordcloud = WordCloud(width=800, height=400,
                          background_color='white').generate(text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.title(title)
    plt.show()


print("\nGenerating WordClouds...")
generate_wordcloud(' '.join(df['Abstract']), 'WordCloud for Abstracts')
generate_wordcloud(' '.join(df['Review Text']), 'WordCloud for Review Texts')

# Check for class balance
if 'Review Type' in df.columns:
    print("\nClass Distribution (%):")
    print(df['Review Type'].value_counts(normalize=True) * 100)

# Show basic statistics
print("\nDescriptive statistics for text length (full_text):")
df['text_length'] = df['full_text'].apply(len)
print(df['text_length'].describe())
sns.histplot(df['text_length'], bins=30, kde=True)
plt.title('Distribution of Text Length')
plt.xlabel('Number of Characters')
plt.ylabel('Frequency')
plt.show()

print("\nAnalysis complete.")
