"""Smoke tests for the terminology mapping module."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "governance"))

from terminology_mapping import TerminologyMapper  # noqa: E402

DICT_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "terminology_dict.csv")


def test_exact_match():
    mapper = TerminologyMapper(DICT_PATH)
    result = mapper.map_term("戈谢病")
    assert result is not None
    assert result["match_type"] == "exact"
    assert result["standard_code"] == "ORPHA:355"


def test_approximate_match():
    mapper = TerminologyMapper(DICT_PATH)
    # Partial term should still map to the correct entry via n-gram similarity.
    result = mapper.map_term("戈谢氏病")
    assert result is not None
    assert result["standard_code"] == "ORPHA:355"


def test_no_match_returns_none():
    mapper = TerminologyMapper(DICT_PATH)
    result = mapper.map_term("完全不相关的词语内容")
    assert result is None
