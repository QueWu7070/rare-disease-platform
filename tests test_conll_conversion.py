"""Smoke tests for the CBLUE -> CoNLL conversion script."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))

import convert_cblue_to_conll as conv  # noqa: E402


def test_convert_example_tags_entities():
    text = "确诊戈谢病"
    entities = [{"start_idx": 2, "end_idx": 4, "type": "dis"}]
    pairs = conv.convert_example(text, entities)
    tags = [tag for _, tag in pairs]
    assert tags == ["O", "O", "B-DISEASE", "I-DISEASE", "I-DISEASE"]


def test_unmapped_type_becomes_o():
    text = "北京"
    entities = [{"start_idx": 0, "end_idx": 1, "type": "loc"}]
    pairs = conv.convert_example(text, entities)
    tags = [tag for _, tag in pairs]
    assert tags == ["O", "O"]


def test_out_of_range_entity_ignored():
    text = "戈谢病"
    entities = [{"start_idx": 0, "end_idx": 99, "type": "dis"}]
    pairs = conv.convert_example(text, entities)
    tags = [tag for _, tag in pairs]
    assert tags == ["O", "O", "O"]
