"""Tests for text normalization."""

import pytest

from resolve_ai.ingestion.normalizer import TextNormalizer


class TestNameNormalization:
    """Tests for company name normalization."""

    def test_lowercase_and_strip(self):
        """Should convert to lowercase and strip whitespace."""
        normalizer = TextNormalizer()
        assert normalizer.normalize_name("  Apple Inc.  ") == "apple inc"

    def test_standardize_suffixes(self):
        """Should standardize company suffixes."""
        normalizer = TextNormalizer()

        assert normalizer.normalize_name("Apple Incorporated") == "apple inc"
        assert normalizer.normalize_name("Microsoft Corporation") == "microsoft corp"
        assert normalizer.normalize_name("Example Limited") == "example ltd"
        assert normalizer.normalize_name("Test Company") == "test co"

    def test_remove_punctuation(self):
        """Should remove punctuation except useful separators."""
        normalizer = TextNormalizer()

        assert normalizer.normalize_name("Amazon.com, Inc.") == "amazon com inc"
        assert normalizer.normalize_name("AT&T Inc.") == "at&t inc"

    def test_handle_llc_variations(self):
        """Should standardize LLC variations."""
        normalizer = TextNormalizer()

        assert normalizer.normalize_name("Test L.L.C.") == "test llc"
        assert normalizer.normalize_name("Example Limited Liability Company") == "example llc"

    def test_empty_string(self):
        """Should handle empty strings."""
        normalizer = TextNormalizer()
        assert normalizer.normalize_name("") == ""

    def test_accents_removed(self):
        """Should remove accents and diacritics."""
        normalizer = TextNormalizer()

        assert normalizer.normalize_name("Café Corporation") == "cafe corp"
        assert normalizer.normalize_name("Société Générale") == "societe generale"


class TestAddressNormalization:
    """Tests for address normalization."""

    def test_lowercase_and_strip(self):
        """Should convert to lowercase and strip whitespace."""
        normalizer = TextNormalizer()
        result = normalizer.normalize_address("  123 Main Street  ")
        assert result == "123 main st"

    def test_standardize_street_types(self):
        """Should standardize street type abbreviations."""
        normalizer = TextNormalizer()

        assert "st" in normalizer.normalize_address("123 Main Street")
        assert "ave" in normalizer.normalize_address("456 Oak Avenue")
        assert "blvd" in normalizer.normalize_address("789 Broadway Boulevard")
        assert "rd" in normalizer.normalize_address("321 Country Road")

    def test_standardize_directions(self):
        """Should standardize directional prefixes."""
        normalizer = TextNormalizer()

        assert "n" in normalizer.normalize_address("100 North Main Street")
        assert "s" in normalizer.normalize_address("200 South Oak Avenue")
        assert "e" in normalizer.normalize_address("300 East First Street")
        assert "w" in normalizer.normalize_address("400 West Second Avenue")

    def test_standardize_states(self):
        """Should standardize state names to abbreviations."""
        normalizer = TextNormalizer()

        assert "ca" in normalizer.normalize_address("Los Angeles California")
        assert "ny" in normalizer.normalize_address("New York New York")
        assert "tx" in normalizer.normalize_address("Houston Texas")

    def test_empty_string(self):
        """Should handle empty strings."""
        normalizer = TextNormalizer()
        assert normalizer.normalize_address("") == ""


class TestBlockingKeys:
    """Tests for blocking key generation."""

    def test_first_3_chars(self):
        """Should return first 3 characters without spaces."""
        normalizer = TextNormalizer()

        assert normalizer.get_blocking_key("apple inc", "first_3_chars") == "app"
        assert normalizer.get_blocking_key("microsoft corp", "first_3_chars") == "mic"
        assert normalizer.get_blocking_key("a b c", "first_3_chars") == "abc"

    def test_first_word(self):
        """Should return first word."""
        normalizer = TextNormalizer()

        assert normalizer.get_blocking_key("apple inc", "first_word") == "apple"
        assert normalizer.get_blocking_key("microsoft corp", "first_word") == "microsoft"

    def test_empty_string(self):
        """Should handle empty strings."""
        normalizer = TextNormalizer()
        assert normalizer.get_blocking_key("", "first_3_chars") == ""
