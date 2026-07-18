"""
Convert the CBLUE CMeEE dataset to BIO CoNLL format for NER training.

CMeEE stores each example as a JSON object with a text field and an entities
list, where each entity carries character offsets (start_idx, end_idx) and a
type. This script emits one character per line with its BIO tag, and a blank
line between examples.

Only the entity types relevant to this study are retained; all others are
mapped to 'O'. Adjust TYPE_MAP to match your label set.

Target runtime: Python 3.8 (standard library only).
"""
import argparse
import json

# Map CMeEE entity types to the study's tag set. Unmapped types become 'O'.
TYPE_MAP = {
    "dis": "DISEASE",   # disease
    "sym": "SYMPTOM",   # clinical symptom
}


def convert_example(text, entities):
    """Convert one CMeEE example to a list of (char, tag) pairs."""
    tags = ["O"] * len(text)
    for ent in entities:
        etype = TYPE_MAP.get(ent.get("type"))
        if etype is None:
            continue
        start = ent["start_idx"]
        end = ent["end_idx"]  # inclusive in CMeEE
        if start < 0 or end >= len(text) or start > end:
            continue
        tags[start] = f"B-{etype}"
        for i in range(start + 1, end + 1):
            tags[i] = f"I-{etype}"
    return list(zip(text, tags))


def convert_file(input_path, output_path):
    """Convert a CMeEE JSON file to a CoNLL file."""
    with open(input_path, encoding="utf-8") as f:
        data = json.load(f)

    n_examples = 0
    with open(output_path, "w", encoding="utf-8") as out:
        for example in data:
            text = example["text"]
            entities = example.get("entities", [])
            pairs = convert_example(text, entities)
            for char, tag in pairs:
                if char.strip() == "":
                    # Skip whitespace characters, which carry no useful tag.
                    continue
                out.write(f"{char} {tag}\n")
            out.write("\n")
            n_examples += 1

    print(f"Converted {n_examples} examples -> {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Convert CMeEE JSON to CoNLL.")
    parser.add_argument("--input", required=True, help="CMeEE JSON file.")
    parser.add_argument("--output", required=True, help="Output CoNLL file.")
    args = parser.parse_args()
    convert_file(args.input, args.output)


if __name__ == "__main__":
    main()
