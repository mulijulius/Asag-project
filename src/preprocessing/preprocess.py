def clean_text(text):
    """Cleans the input text by removing unwanted characters."""
    import re
    # Remove special characters and numbers
    cleaned = re.sub(r'[^a-zA-Z\s]', '', text)
    return cleaned.strip()


def preprocess_dataframe(df):
    """Processes the DataFrame by applying text cleaning to a specific column."""
    # Assuming the DataFrame has a column 'text'
    df['text'] = df['text'].apply(clean_text)
    return df
