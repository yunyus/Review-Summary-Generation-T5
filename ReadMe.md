# Review Summary Generator

This project provides a framework for fine-tuning a T5 model to generate summaries of product reviews. The project includes data loading, preprocessing, model training, and inference functionalities.

## Project Structure

```
review_summary_generator/
│
├── data/
│   └── data.csv
│   └── summarized_reviews.csv
├── src/
│   ├── config.py
│   ├── data_preparation.py
│   ├── dataset.py
│   ├── model_training.py
│   ├── inference.py
│   ├── utils.py
│   ├── train.py
│   └── generate_summary.py
│   └── data_generation.py
├── requirements.txt
└── README.md
```

## Setup

1. Clone the repository

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Place your dataset in the `data` directory. The dataset should be a CSV file named `data.csv` with columns `product_id`, `avg_rating`, `review_body`, `review_summary`. You can find a sample dataset on Kaggle at this [link](https://www.kaggle.com/datasets/yusufkesmenn/product-reviews-with-summarized-feedback/settings).

## Usage

### Data Generation

If you want to generate a review summaries dataset, you can use the provided script:

```
python src/data_generation.py
```

### Training

To train the model, run:

```
python src/train.py
```

### Inference

To generate a review summary, run:

```
python src/generate_summary.py
```
