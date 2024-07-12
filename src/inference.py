import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration
from src.config import MODEL_NAME, DEVICE


def load_model(model_path):
    model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.load_state_dict(torch.load(model_path))
    model.to(DEVICE)
    model.eval()
    return model


def generate_review_summary(model, tokenizer, product_id, avg_rating, review_body):
    input_text = f"{product_id} <REVIEW_SEP> {avg_rating} <RATING_SEP> {review_body}"
    input_encoding = tokenizer(
        input_text,
        max_length=250,
        padding='max_length',
        truncation=True,
        return_tensors="pt"
    )
    input_ids = input_encoding.input_ids.to(DEVICE)
    attention_mask = input_encoding.attention_mask.to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=100,
            num_beams=4,
            early_stopping=True
        )
    summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return summary


def main():
    model_path = 'checkpoints/t5_model_epoch_5.pt'  # Path to the trained model
    model = load_model(model_path)
    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    # Example usage, body structure is not important
    product_id = "Urun123"
    avg_rating = 4.5
    review_body = """
    1. Ürün gerçekten harika, beklediğimden çok daha iyi çıktı.
    2. Kalitesi çok iyi ancak fiyatı biraz yüksek.
    3. Kargo çok hızlı geldi, teşekkürler.
    4. Ürün anlatıldığı gibi. Memnun kaldım.
    5. Renkleri çok canlı, fotoğraflardan daha güzel.
    6. Fiyatına göre çok iyi bir ürün, tavsiye ederim.
    7. Beklentilerimi karşıladı, tekrar alabilirim.
    8. Paketleme özenliydi, hasarsız elime ulaştı.
    9. Kullanımı çok kolay ve pratik.
    10. Tasarımı çok şık, malzeme kalitesi yüksek.
    """

    summary = generate_review_summary(
        model, tokenizer, product_id, avg_rating, review_body)
    print("Generated Review Summary:", summary)


if __name__ == "__main__":
    main()
