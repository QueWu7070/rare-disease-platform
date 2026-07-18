"""Fine-tune Rare-RoBERTa for clinical named entity recognition (NER).

Base model : hfl/chinese-roberta-wwm-ext (12Transformer layers, 768 hidden units)
Library: Hugging Face Transformers v4.11.3, PyTorch v1.7.1
Hyperparams: max_seq_len=512, batch_size=32, lr=2e-5 (AdamW), 10 epochs,early stopping (patience=3)
Evaluation : strict exact-match — a prediction is a true positive only whenboth entity span and entity type match the annotation exactly

Hardware: NVIDIA Tesla V100 (32 GB VRAM), CUDA 11.0, Ubuntu 20.04
             FP16 mixed-precision training enabled for V100.
"""
import argparse
import numpy as np
from datasets import Dataset
from seqeval.metrics import precision_score, recall_score, f1_score
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,EarlyStoppingCallback,
)

# Base model: Chinese RoBERTa with whole-word masking
MODEL_NAME = "hfl/chinese-roberta-wwm-ext"

# BIO label set covering Disease, Symptom, and Medication entity types
LABELS = ["O", "B-DIS", "I-DIS", "B-SYM", "I-SYM", "B-DRUG", "I-DRUG"]
LABEL2ID = {label: idx for idx, label in enumerate(LABELS)}
ID2LABEL = {idx: label for label, idx in LABEL2ID.items()}


def load_conll(path: str) -> Dataset:
    """Load a character-level BIO CoNLL file into a HuggingFace Dataset.

    Format: one token per line as'CHAR TAG', blank lines between sentences.
    Tags must belong to LABELS; integer IDs are assigned via LABEL2ID.
    """
    sentences = []
    tokens, tags = [], []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                if tokens:
                    sentences.append({"tokens": tokens, "ner_tags": tags})
                    tokens, tags = [], []
                continue
            char, tag = line.split()
            tokens.append(char)
            tags.append(LABEL2ID[tag])
    if tokens:
        sentences.append({"tokens": tokens, "ner_tags": tags})
    return Dataset.from_list(sentences)


def tokenize_and_align(examples: dict, tokenizer, max_len: int = 512) -> dict:
    """Tokenise inputs and align BIO labels to sub-word token positions.

    Sub-word continuations and special tokens ([CLS], [SEP], [PAD]) receive
    label -100 so they are ignored by the cross-entropy loss.
    """
    tokenized = tokenizer(
        examples["tokens"],
        truncation=True,
        max_length=max_len,
        is_split_into_words=True,
    )
    all_labels = []
    for i, tags in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev_word_id = None
        label_ids = []
        for word_id in word_ids:
            if word_id is None:
                label_ids.append(-100)          # special token
            elif word_id != prev_word_id:
                label_ids.append(tags[word_id]) # first sub-word of a character
            else:
                label_ids.append(-100)          # subsequent sub-words
            prev_word_id = word_id
        all_labels.append(label_ids)
    tokenized["labels"] = all_labels
    return tokenized


def build_compute_metrics():
    """Return a metric function compatible with Trainer.evaluate().

    Converts integer predictions back to string labels, filters -100 positions,
    and computes seqeval precision / recall / F1 under strict exact-match.
    """
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)

        true_label_seqs, pred_label_seqs = [], []
        for pred_row, label_row in zip(predictions, labels):
            true_row, pred_row_str = [], []
            for pred_id, label_id in zip(pred_row, label_row):
                if label_id == -100:
                    continue
                true_row.append(ID2LABEL[int(label_id)])
                pred_row_str.append(ID2LABEL[int(pred_id)])
            true_label_seqs.append(true_row)
            pred_label_seqs.append(pred_row_str)

        return {
            "precision": precision_score(true_label_seqs, pred_label_seqs),
            "recall":recall_score(true_label_seqs, pred_label_seqs),
            "f1":        f1_score(true_label_seqs, pred_label_seqs),
        }
    return compute_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Fine-tune Rare-RoBERTa for clinical NER."
    )
    parser.add_argument("--train",default="data/ner_sample.conll",
                        help="Training CoNLL file (15,000 CBLUE sentences for full run).")
    parser.add_argument("--eval",       default="data/ner_sample.conll",
                        help="Evaluation CoNLL file (200 synthetic cases for full run).")
    parser.add_argument("--output",     default="outputs/rare-roberta")
    parser.add_argument("--epochs",     type=int,default=10)
    parser.add_argument("--batch_size", type=int,   default=32)
    parser.add_argument("--lr",         type=float, default=2e-5)
    parser.add_argument("--max_len",    type=int,   default=512)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def tokenize_fn(examples):
        return tokenize_and_align(examples, tokenizer, args.max_len)

    remove_cols = ["tokens", "ner_tags"]
    train_ds = load_conll(args.train).map(tokenize_fn, batched=True,remove_columns=remove_cols)
    eval_ds  = load_conll(args.eval).map(tokenize_fn,batched=True,
                                remove_columns=remove_cols)

    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,               # AdamW is the default optimiser
        weight_decay=0.01,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        logging_steps=10,
        seed=42,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=build_compute_metrics(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],
    )

    trainer.train()
    metrics = trainer.evaluate()
    print("Evaluation metrics:", metrics)
    trainer.save_model(args.output)
    tokenizer.save_pretrained(args.output)
    print(f"Model checkpoint saved to: {args.output}")


if __name__ == "__main__":
    main()
