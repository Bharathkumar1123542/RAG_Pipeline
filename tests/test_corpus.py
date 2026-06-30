"""tests/test_corpus.py — Unit tests for CorpusGenerator."""
import pytest

from rag.corpus import CorpusGenerator
from rag.models import VALID_CATEGORIES, ProductDocument


class TestCorpusGenerator:
    def test_correct_count(self):
        docs = CorpusGenerator(seed=42).generate(100)
        assert len(docs) == 100

    def test_returns_product_documents(self):
        docs = CorpusGenerator(seed=0).generate(10)
        for doc in docs:
            assert isinstance(doc, ProductDocument)

    def test_category_distribution(self):
        n = 500
        docs = CorpusGenerator(seed=42).generate(n)
        counts = {}
        for doc in docs:
            counts[doc.category] = counts.get(doc.category, 0) + 1
        expected_per_cat = n / len(VALID_CATEGORIES)
        tolerance = 0.20  # ±20%
        for cat in VALID_CATEGORIES:
            assert cat in counts, f"Category {cat!r} missing from corpus"
            diff = abs(counts[cat] - expected_per_cat) / expected_per_cat
            assert diff <= tolerance, (
                f"Category {cat!r}: count={counts[cat]}, "
                f"expected≈{expected_per_cat:.0f}, diff={diff:.1%}"
            )

    def test_all_categories_valid(self):
        docs = CorpusGenerator(seed=1).generate(50)
        for doc in docs:
            assert doc.category in VALID_CATEGORIES

    def test_doc_id_format(self):
        import re
        docs = CorpusGenerator(seed=5).generate(20)
        pattern = re.compile(r"^prod_\d{4}$")
        for doc in docs:
            assert pattern.match(doc.doc_id), f"Invalid doc_id: {doc.doc_id!r}"

    def test_doc_ids_unique(self):
        docs = CorpusGenerator(seed=42).generate(200)
        ids = [d.doc_id for d in docs]
        assert len(ids) == len(set(ids)), "doc_ids are not unique"

    def test_description_word_count_bounds(self):
        docs = CorpusGenerator(seed=42).generate(100)
        for doc in docs:
            wc = len(doc.description.split())
            assert 50 <= wc <= 300, (
                f"{doc.doc_id}: description has {wc} words (expected 50–300)"
            )

    def test_features_count_bounds(self):
        docs = CorpusGenerator(seed=42).generate(100)
        for doc in docs:
            assert 3 <= len(doc.features) <= 7, (
                f"{doc.doc_id}: {len(doc.features)} features (expected 3–7)"
            )

    def test_specs_non_empty(self):
        docs = CorpusGenerator(seed=42).generate(100)
        for doc in docs:
            assert doc.specs, f"{doc.doc_id}: specs is empty"

    def test_reproducibility_same_seed(self):
        a = CorpusGenerator(seed=99).generate(50)
        b = CorpusGenerator(seed=99).generate(50)
        for da, db in zip(a, b):
            assert da.doc_id == db.doc_id
            assert da.description == db.description

    def test_different_seeds_differ(self):
        a = CorpusGenerator(seed=1).generate(50)
        b = CorpusGenerator(seed=2).generate(50)
        # At least some descriptions should differ
        descriptions_a = {d.description for d in a}
        descriptions_b = {d.description for d in b}
        assert descriptions_a != descriptions_b

    def test_created_at_format(self):
        import re
        docs = CorpusGenerator(seed=7).generate(30)
        pattern = re.compile(r"^\d{4}-\d{2}-\d{2}$")
        for doc in docs:
            assert pattern.match(doc.created_at), f"Invalid created_at: {doc.created_at!r}"

    def test_minimum_n_one(self):
        docs = CorpusGenerator(seed=0).generate(1)
        assert len(docs) == 1

    def test_invalid_n_raises(self):
        with pytest.raises(ValueError):
            CorpusGenerator().generate(0)
