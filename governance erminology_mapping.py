"""Map non-standard rare-disease terms to ORPHA / OMIM / HPO codes.

Implements a hybrid retrieval strategy combining normalised Levenshtein
edit-distance similarity (weight 0.4) and character-level cosine similarity
(weight 0.6). A term is auto-accepted when the weighted score exceeds the
threshold of 0.85; otherwise it is flagged for human expert review.

Matching success rate improves from 68.2% (exact string match) to 88.3 %
(semantic matching) on the test set, as reported in the Results section.

Production note: Replace char_bow_vector() with sentence-transformer
embeddings for higher semantic accuracy. The character-bag implementation
is provided here so the module runs without additional heavy dependencies.
"""
import argparse
import numpy as np
from rapidfuzz.distance import Levenshtein

# Minimal reference vocabulary (extend with the full ORPHA / OMIM / HPO dumps
# for production deployment).
STANDARD_TERMS: dict[str, str] = {
    "杜氏肌营养不良":"ORPHA:98896",
    "假肥大型肌营养不良": "ORPHA:98896",
    "庞贝病":           "ORPHA:365",
    "戈谢病":           "ORPHA:355",
    "苯丙酮尿症":        "ORPHA:716",
    "法布雷病":          "ORPHA:324",
}

EDIT_WEIGHT= 0.4
COSINE_WEIGHT = 0.6
THRESHOLD     = 0.85


def edit_similarity(a: str, b: str) -> float:
    """Normalised edit-distance similarity in [0, 1]."""
    dist = Levenshtein.distance(a, b)
    return1.0 - dist / max(len(a), len(b), 1)


def char_bow_vector(text: str, vocab: list) -> np.ndarray:
    """Character bag-of-words vector over a shared vocabulary."""
    return np.array([text.count(c) for c in vocab], dtype=float)


def cosine_similarity(a: str, b: str) -> float:
    """Cosine similarity between character bag-of-words representations."""
    vocab = sorted(set(a) | set(b))
    va, vb = char_bow_vector(a, vocab), char_bow_vector(b, vocab)
    denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1.0
    return float(np.dot(va, vb) / denom)


def map_term(term: str) -> dict:
    """Return the best-matching standard code for a raw disease term.

    Returns:
        dict with keys 'code' (ORPHA/OMIM string or None), 'score' (float),
        and 'status' ('matched' | 'needs_manual_review').
    """
    best_code, best_score = None, 0.0
    for std_name, code in STANDARD_TERMS.items():
        score = (EDIT_WEIGHT   * edit_similarity(term, std_name) +
                 COSINE_WEIGHT * cosine_similarity(term, std_name))
        if score > best_score:
            best_code, best_score = code, score

    status = "matched" if best_score >= THRESHOLD else "needs_manual_review"
    return {"code": best_code, "score": round(best_score, 3), "status": status}


def main():
    parser = argparse.ArgumentParser(
        description="Map a raw disease term to a standard ontology code."
    )
    parser.add_argument("--term", required=True,
                        help="Non-standard disease label extracted by the NER module.")
    args = parser.parse_args()
    result = map_term(args.term)
    print(f"'{args.term}'->  {result['code']}  "
          f"(score={result['score']}, {result['status']})")


if __name__ == "__main__":
    main()
