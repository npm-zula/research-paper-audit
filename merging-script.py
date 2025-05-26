#!/usr/bin/env python3
"""
Script to merge corpus.csv and peer_review_dataset(Sheet1).csv
Maintains the format of corpus.csv: Paper Title, Review Text, Review Type, Abstract
Output: corpus-final.csv
"""

import pandas as pd
import numpy as np


def clean_text(text):
    """Clean text by removing extra whitespace and handling NaN values"""
    if pd.isna(text):
        return ""
    return str(text).strip()


def load_and_process_corpus(file_path):
    """Load and process the original corpus.csv file"""
    print(f"Loading corpus data from: {file_path}")

    # Try multiple encodings
    encodings = ['utf-8', 'cp1252', 'latin-1']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(f"Successfully loaded corpus.csv with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")

    if df is None:
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
        print("Loaded corpus.csv with UTF-8 encoding and error replacement")

    # Clean and standardize column names
    df.columns = df.columns.str.strip()

    # Expected columns: Paper Title, Review Text, Review Type, Abstract
    expected_cols = ['Paper Title', 'Review Text', 'Review Type', 'Abstract']

    # Ensure all expected columns exist
    for col in expected_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in corpus.csv")
            df[col] = ""

    # Select and reorder columns to match expected format
    df = df[expected_cols].copy()

    # Clean text fields
    for col in df.columns:
        df[col] = df[col].apply(clean_text)

    print(f"Corpus data loaded: {len(df)} records")
    return df


def load_and_process_peer_review(file_path):
    """Load and process the peer_review_dataset(Sheet1).csv file"""
    print(f"Loading peer review data from: {file_path}")

    # Try multiple encodings
    encodings = ['utf-8', 'cp1252', 'latin-1']
    df = None

    for encoding in encodings:
        try:
            df = pd.read_csv(file_path, encoding=encoding)
            print(
                f"Successfully loaded peer_review_dataset with {encoding} encoding")
            break
        except UnicodeDecodeError:
            continue
        except Exception as e:
            print(f"Error with {encoding} encoding: {e}")

    if df is None:
        df = pd.read_csv(file_path, encoding='utf-8', errors='replace')
        print("Loaded peer_review_dataset with UTF-8 encoding and error replacement")

    # Clean and standardize column names
    df.columns = df.columns.str.strip()

    # Expected columns in peer review: Paper Title, Abstract, Review Text, Review Type
    # Map to corpus format: Paper Title, Review Text, Review Type, Abstract

    # Rename columns to match corpus format
    column_mapping = {
        'Paper Title ': 'Paper Title',  # Handle potential trailing space
        'Paper Title': 'Paper Title',
        'Abstract': 'Abstract',
        'Review Text': 'Review Text',
        'Review Type': 'Review Type'
    }

    # Apply column mapping
    df = df.rename(columns=column_mapping)

    # Ensure all required columns exist
    required_cols = ['Paper Title', 'Review Text', 'Review Type', 'Abstract']
    for col in required_cols:
        if col not in df.columns:
            print(f"Warning: Column '{col}' not found in peer_review_dataset")
            df[col] = ""

    # Reorder columns to match corpus format: Paper Title, Review Text, Review Type, Abstract
    df = df[required_cols].copy()

    # Clean text fields
    for col in df.columns:
        df[col] = df[col].apply(clean_text)

    print(f"Peer review data loaded: {len(df)} records")
    return df


def standardize_review_types(df):
    """Standardize review type labels to match corpus format"""
    print("Standardizing review type labels...")

    # Standardize variations of AI-Generated
    df['Review Type'] = df['Review Type'].replace({
        'AI Generated': 'AI-Generated',
        'ai-generated': 'AI-Generated',
        'ai generated': 'AI-Generated',
        'Ai-Generated': 'AI-Generated',
        'Ai Generated': 'AI-Generated'
    })

    # Standardize other variations
    df['Review Type'] = df['Review Type'].replace({
        'authentic': 'Authentic',
        'generic': 'Generic',
        'AUTHENTIC': 'Authentic',
        'GENERIC': 'Generic'
    })

    return df


def merge_datasets(corpus_df, peer_review_df):
    """Merge the two datasets"""
    print("Merging datasets...")

    # Add source identifier for tracking
    corpus_df['source'] = 'corpus'
    peer_review_df['source'] = 'peer_review'

    # Combine datasets
    merged_df = pd.concat([corpus_df, peer_review_df], ignore_index=True)

    # Standardize review type labels
    merged_df = standardize_review_types(merged_df)

    # Remove duplicates based on Paper Title and Review Text
    print("Removing duplicates...")
    initial_count = len(merged_df)
    merged_df = merged_df.drop_duplicates(
        subset=['Paper Title', 'Review Text'], keep='first')
    final_count = len(merged_df)

    print(f"Removed {initial_count - final_count} duplicate records")

    # Drop the source column for final output
    merged_df = merged_df.drop(columns=['source'])

    return merged_df


def save_merged_dataset(df, output_path):
    """Save the merged dataset"""
    print(f"Saving merged dataset to: {output_path}")

    # Ensure the column order matches corpus.csv format
    column_order = ['Paper Title', 'Review Text', 'Review Type', 'Abstract']
    df = df[column_order]

    # Save to CSV
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Successfully saved {len(df)} records to {output_path}")


def print_statistics(df, dataset_name):
    """Print statistics about the dataset"""
    print(f"\n{dataset_name} Statistics:")
    print(f"Total records: {len(df)}")
    print(f"Review Type distribution:")
    print(df['Review Type'].value_counts())
    print(f"Missing values per column:")
    print(df.isnull().sum())
    print("-" * 50)


def main():
    """Main function to merge the datasets"""
    print("=" * 60)
    print("Dataset Merging Script")
    print("=" * 60)

    # File paths
    corpus_path = "corpus.csv"
    peer_review_path = "peer_review_dataset(Sheet1).csv"
    output_path = "corpus-final.csv"

    try:
        # Load datasets
        corpus_df = load_and_process_corpus(corpus_path)
        peer_review_df = load_and_process_peer_review(peer_review_path)

        # Standardize review types
        peer_review_df = standardize_review_types(peer_review_df)

        # Print statistics
        print_statistics(corpus_df, "Corpus Dataset")
        print_statistics(peer_review_df, "Peer Review Dataset")

        # Merge datasets
        merged_df = merge_datasets(corpus_df, peer_review_df)

        # Print merged statistics
        print_statistics(merged_df, "Merged Dataset")

        # Save merged dataset
        save_merged_dataset(merged_df, output_path)

        print("\n" + "=" * 60)
        print("Dataset merging completed successfully!")
        print(f"Final dataset saved as: {output_path}")
        print(f"Total records in merged dataset: {len(merged_df)}")
        print("=" * 60)

    except Exception as e:
        print(f"Error during merging process: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
