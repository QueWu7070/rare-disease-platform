"""
Domain-adaptive pre-training (DAPT) for Rare-RoBERTa.

Continues masked-language-model pre-training of RoBERTa-wwm-ext on an unlabelled
rare-disease clinical corpus before downstream NER fine-tuning. This is the
"domain-adapted" stage: the resulting checkpoint is consumed by
train_rare_roberta.py via --model_name_or_path.

Target runtime: Python 3.8, torch 1.7.1+cu110, transformers 4.11.3,
datasets 1.18.4, single NVIDIA Tesla V100 (32GB).
"""
import argparse

from datasets import load_dataset
from transformers import (
    AutoModelForMaskedLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

BASE_MODEL = "hfl/chinese-roberta-wwm-ext"


def main():
    parser = argparse.ArgumentParser(description="DAPT for Rare-RoBERTa.")
    parser.add_argument(
        "--corpus",
        required=True,
        help="Plain-text file, one clinical sentence/paragraph per line.",
    )
    parser.add_argument("--output_dir", default="./rare-roberta-dapt")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=512)
    parser.add_argument("--mlm_probability", type=float, default=0.15)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForMaskedLM.from_pretrained(BASE_MODEL)

    dataset = load_dataset("text", data_files={"train": args.corpus})

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=args.max_len,
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Whole-word masking is inherited from the base model's tokenizer; here we
    # use standard MLM collation with the reported 15% masking probability.
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=True,
        mlm_probability=args.mlm_probability,
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=True,               # mixed precision on V100
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=collator,
    )
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Domain-adapted checkpoint saved to {args.output_dir}")


if __name__ == "__main__":
    main()
