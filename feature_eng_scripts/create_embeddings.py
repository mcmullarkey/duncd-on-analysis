import pandas as pd
from transformers import pipeline

def main():
    create_embeddings("data/labeling-app/podcast_episodes.csv")

def create_embeddings(csv_file):
    # Example DataFrame
    df = pd.read_csv(csv_file, encoding='utf-8', encoding_errors='ignore')
    print(df.size)

    # Load HuggingFace pipeline for embeddings (using BERT-based model)
    embedding_pipeline = pipeline('feature-extraction', model='answerdotai/ModernBERT-base', tokenizer='answerdotai/ModernBERT-base')

    # Function to extract embeddings
    def get_embeddings(text):
        if isinstance(text, str):
            print("Is a string")
            # Generate embeddings and average over the token dimension to get a fixed-length vector
            embeddings = embedding_pipeline(text)
            return [sum(token_vector) / len(token_vector) for token_vector in zip(*embeddings[0])]
        else:
            print("Not a string")
            return []

    # Apply to the DataFrame column
    df['embeddings'] = df['description'].apply(get_embeddings)
    
    # Add a column for embedding dimensions
    df['embedding_dimensions'] = df['embeddings'].apply(lambda x: len(x))

    # Display the DataFrame with embeddings
    print(df)
    
    # Write embeddings
    df.to_parquet("data/labeling-app/description_embeddings.parquet", index=False)

if __name__ == '__main__':
    main()