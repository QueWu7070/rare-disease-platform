"""Run NER inference with a fine-tuned Rare-RoBERTa checkpoint.

Extracts Disease (DIS), Symptom (SYM), and Medication (DRUG) entities
from Chinese clinical free-text using character-level BIO decoding.
"""
import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification

LABELS = ["O", "B-DIS", "I-DIS", "B-SYM", "I-SYM", "B-DRUG", "I-DRUG"]
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def extract_entities(text: str, model_dir: str) -> list:
    """Extract named entities from a Chinese clinical note.

    Args:
        text:Input clinical narrative (Chinese free text).
        model_dir: Path to a fine-tuned Rare-RoBERTa checkpoint.

    Returns:
        List of dicts with keys'type' and 'text', one per detected entity span.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForTokenClassification.from_pretrained(model_dir)
    model.eval()

    chars = list(text)
    encoding = tokenizer(
        chars,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=512,
    )

    with torch.no_grad():
        logits = model(**encoding).logits

    pred_ids = logits.argmax(dim=-1)[0].tolist()
    word_ids = encoding.word_ids(batch_index=0)

    entities = []
    current = None
    prev_word_id = None

    for token_idx, word_id in enumerate(word_ids):
        # Skip special tokens and sub-word duplicates
        if word_id is None or word_id == prev_word_id:
            prev_word_id = word_id
            continue
        prev_word_id = word_id

        label = ID2LABEL[pred_ids[token_idx]]

        if label.startswith("B-"):
            if current:
                entities.append(current)
            current = {"type": label[2:], "text": chars[word_id]}
        elif label.startswith("I-") and current and current["type"] == label[2:]:
            current["text"] += chars[word_id]
        else:
            if current:
                entities.append(current)
            current = None

    if current:
        entities.append(current)

    return entities


def main():
    parser = argparse.ArgumentParser(
        description="Rare-RoBERTa NER inference on a single clinical note."
    )
    parser.add_argument("--model", default="outputs/rare-roberta",
                        help="Path to fine-tuned checkpoint directory.")
    parser.add_argument("--text", required=True,
                        help="Clinical narrative string.")
    args = parser.parse_args()

    for entity in extract_entities(args.text, args.model):
        print(f"[{entity['type']}]{entity['text']}")


if __name__ == "__main__":
    main()
