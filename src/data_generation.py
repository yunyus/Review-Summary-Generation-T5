import os
import pandas as pd
import numpy as np
from openai import OpenAI

api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)


def generate_summary(text):
    system_prompt = (
        "Sen, kullanıcı yorumlarını özetlemek için programlanmış bir yapay zekasın. "
        "Sana yaklaşık 10 adet kullanıcı yorumu sunulacak. "
        "Bu yorumları dikkatlice oku ve olumlu ve olumsuz tüm detayları içeren, "
        "300-350 karakter uzunluğunda özlü bir özet yaz. Özetin anlaşılır, "
        "doğru ve bilgilendirici olmasına dikkat et."
    )

    user_prompt = "Aşağıdaki kullanıcı yorumlarını özetle:\n\n" + \
        "\n".join(text)

    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        model="gpt-3.5-turbo-1106",
    )

    summary = response.choices[0].message.content
    return summary


df = pd.read_csv('data/data.csv', on_bad_lines='skip')

# Sort by product_id
df.sort_values(by='product_id', inplace=True)

# Constants for the original dataset statistics
# You can change these values according to your dataset
mean_target = 4.4
std_dev_target = 1.0


def chunk_data_while_obeying_stats(reviews, ratings, chunk_size, mean_target, std_dev_target):
    chunks = []
    for i in range(0, len(reviews), chunk_size):
        chunk_reviews = reviews[i:i + chunk_size]
        chunk_ratings = ratings[i:i + chunk_size]

        if len(chunk_reviews) < chunk_size:
            continue

        mean_rating = np.mean(chunk_ratings)
        std_dev_rating = np.std(chunk_ratings)

        if (abs(mean_rating - mean_target) <= 1) and (abs(std_dev_rating - std_dev_target) <= 1):
            chunks.append((chunk_reviews, chunk_ratings))

    return chunks


product_ids = []
review_bodies = []
avg_ratings = []
review_summaries = []

output_file = 'data/summarized_reviews.csv'

if not os.path.exists(output_file):
    new_df = pd.DataFrame({
        'product_id': [],
        'review_body': [],
        'avg_rating': [],
        'review_summary': []
    })
    new_df.to_csv(output_file, index=False)

for product_id, group in df.groupby('product_id'):
    reviews = group['review_body'].tolist()
    ratings = group['review_rating'].tolist()

    chunks = chunk_data_while_obeying_stats(
        reviews, ratings, 10, mean_target, std_dev_target)

    for chunk_reviews, chunk_ratings in chunks:
        summary = generate_summary(chunk_reviews)
        avg_rating = np.mean(chunk_ratings)

        product_ids.append(product_id)
        review_bodies.append(" ".join(chunk_reviews))
        avg_ratings.append(avg_rating)
        review_summaries.append(summary)

        new_df = pd.DataFrame({
            'product_id': [product_id],
            'review_body': [" ".join(chunk_reviews)],
            'avg_rating': [avg_rating],
            'review_summary': [summary]
        })

        new_df.to_csv(output_file, mode='a', header=False, index=False)

print("New CSV file 'summarized_reviews.csv' has been created.")
