import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import DATA_FILE


def load_dataset(file_path=DATA_FILE):
    return pd.read_csv(file_path)


def prepare_data(df):
    input_texts = []
    target_texts = []

    for idx, row in df.iterrows():
        product_id = row['product_id']
        avg_rating = row['avg_rating']
        review_body = row['review_body']
        review_summary = row['review_summary']

        input_text = f"{product_id} <REVIEW_SEP> {avg_rating} <RATING_SEP> {review_body}"
        target_text = review_summary

        input_texts.append(input_text)
        target_texts.append(target_text)

    return input_texts, target_texts


def split_data(input_texts, target_texts, test_size=0.1):
    return train_test_split(input_texts, target_texts, test_size=test_size)
