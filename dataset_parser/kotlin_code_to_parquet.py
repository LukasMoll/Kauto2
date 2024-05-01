import os
import pandas as pd
import argparse

def read_kotlin_files_to_dataframe(directory):
    """
    Reads all Kotlin files from the given directory and returns a DataFrame.
    Args:
    directory (str): The path to the directory containing Kotlin files.
    Returns:
    pd.DataFrame: A DataFrame with a single column 'text' containing the contents of each Kotlin file.
    """
    kotlin_files_contents = []
    for file_name in os.listdir(directory):
        if file_name.endswith('.kt'):
            file_path = os.path.join(directory, file_name)
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                kotlin_files_contents.append(content)

    return pd.DataFrame(kotlin_files_contents, columns=['text'])

def split_and_save(df, train_output, eval_output):
    """
    Splits the DataFrame into training and evaluation sets and saves them to Parquet files.
    Args:
    df (pd.DataFrame): The DataFrame to split and save.
    train_output (str): Path to save the training data Parquet file.
    eval_output (str): Path to save the evaluation data Parquet file.
    """
    # Split the data
    train_df = df.sample(frac=0.9, random_state=42)  # 90% for training
    eval_df = df.drop(train_df.index)  # Remaining 10% for evaluation

    # Save the data to Parquet
    train_df.to_parquet(train_output, index=False)
    eval_df.to_parquet(eval_output, index=False)
    print(f"Training data saved to {train_output} with {len(train_df)} rows.")
    print(f"Evaluation data saved to {eval_output} with {len(eval_df)} rows.")

def main():
    parser = argparse.ArgumentParser(description='Extract Kotlin files, split into train and eval datasets, and save to Parquet files.')
    parser.add_argument('source_dir', type=str, help='The source directory to search for Kotlin files.')
    parser.add_argument('train_parquet', type=str, help='The path to save the training data Parquet file.')
    parser.add_argument('eval_parquet', type=str, help='The path to save the evaluation data Parquet file.')

    args = parser.parse_args()

    # Read Kotlin files to a DataFrame
    df = read_kotlin_files_to_dataframe(args.source_dir)
    df = df.sample(frac=1).reset_index(drop=True)
    # Split the DataFrame and save to specified paths
    split_and_save(df, args.train_parquet, args.eval_parquet)

if __name__ == "__main__":
    main()
