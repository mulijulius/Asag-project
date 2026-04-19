import pandas as pd


def load_mohler_dataset(file_path):
    # Load the Mohler dataset
    df = pd.read_csv(file_path)
    df = df[['question_id', 'question_text', 'student_answer', 'reference_answer', 'score', 'split']]
    df['score'] = df['score'].fillna(0).clip(0, 1)
    print_summary(df)
    return df


def load_scientbank_dataset(file_path):
    # Load the ScientBank dataset
    df = pd.read_csv(file_path)
    df = df[['question_id', 'question_text', 'student_answer', 'reference_answer', 'score', 'split']]
    df['score'] = df['score'].fillna(0).clip(0, 1)
    print_summary(df)
    return df


def load_beetle_dataset(file_path):
    # Load the Beetle dataset
    df = pd.read_csv(file_path)
    df = df[['question_id', 'question_text', 'student_answer', 'reference_answer', 'score', 'split']]
    df['score'] = df['score'].fillna(0).clip(0, 1)
    print_summary(df)
    return df


def load_asag2024_dataset(file_path):
    # Load the Asag2024 dataset
    df = pd.read_csv(file_path)
    df = df[['question_id', 'question_text', 'student_answer', 'reference_answer', 'score', 'split']]
    df['score'] = df['score'].fillna(0).clip(0, 1)
    print_summary(df)
    return df


def load_dataset_by_name(name, file_path):
    loaders = {
        'mohler': load_mohler_dataset,
        'scientbank': load_scientbank_dataset,
        'beetle': load_beetle_dataset,
        'asag2024': load_asag2024_dataset,
    }
    loader = loaders.get(name)
    if loader:
        return loader(file_path)
    else:
        raise ValueError(f"Unknown dataset name: {name}")


def print_summary(df):
    print(f"Loaded dataset with {len(df)} records.")
    print(df.describe())

