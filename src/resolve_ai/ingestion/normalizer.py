"""Text normalization for entity resolution."""

import re
import unicodedata


class TextNormalizer:
    """Normalize text for consistent matching."""

    COMPANY_SUFFIXES = {
        "incorporated": "inc",
        "corporation": "corp",
        "limited": "ltd",
        "company": "co",
        "limited liability company": "llc",
        "l.l.c.": "llc",
        "l.l.c": "llc",
        "p.l.c.": "plc",
        "p.l.c": "plc",
        "& co": "co",
        "&co": "co",
    }

    STATE_ABBREVS = {
        "alabama": "al",
        "alaska": "ak",
        "arizona": "az",
        "arkansas": "ar",
        "california": "ca",
        "colorado": "co",
        "connecticut": "ct",
        "delaware": "de",
        "florida": "fl",
        "georgia": "ga",
        "hawaii": "hi",
        "idaho": "id",
        "illinois": "il",
        "indiana": "in",
        "iowa": "ia",
        "kansas": "ks",
        "kentucky": "ky",
        "louisiana": "la",
        "maine": "me",
        "maryland": "md",
        "massachusetts": "ma",
        "michigan": "mi",
        "minnesota": "mn",
        "mississippi": "ms",
        "missouri": "mo",
        "montana": "mt",
        "nebraska": "ne",
        "nevada": "nv",
        "new hampshire": "nh",
        "new jersey": "nj",
        "new mexico": "nm",
        "new york": "ny",
        "north carolina": "nc",
        "north dakota": "nd",
        "ohio": "oh",
        "oklahoma": "ok",
        "oregon": "or",
        "pennsylvania": "pa",
        "rhode island": "ri",
        "south carolina": "sc",
        "south dakota": "sd",
        "tennessee": "tn",
        "texas": "tx",
        "utah": "ut",
        "vermont": "vt",
        "virginia": "va",
        "washington": "wa",
        "west virginia": "wv",
        "wisconsin": "wi",
        "wyoming": "wy",
    }

    def normalize_name(self, name: str) -> str:
        """Normalize a company/entity name.

        Args:
            name: Raw entity name.

        Returns:
            Normalized name for matching.
        """
        if not name:
            return ""

        text = name.lower().strip()
        text = self._remove_accents(text)

        # Longer phrases first so "limited liability company" matches before "limited"
        for full, abbrev in sorted(
            self.COMPANY_SUFFIXES.items(), key=lambda x: len(x[0]), reverse=True
        ):
            text = re.sub(rf"\b{re.escape(full)}\b", abbrev, text)

        text = re.sub(r"[^\w\s&-]", " ", text)

        # Handle "L.L.C." -> "l l c" after punctuation removal
        text = re.sub(r"\bl\s*l\s*c\b", "llc", text)
        text = re.sub(r"\bp\s*l\s*c\b", "plc", text)

        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def normalize_address(self, address: str) -> str:
        """Normalize an address.

        Args:
            address: Raw address string.

        Returns:
            Normalized address for matching.
        """
        if not address:
            return ""

        text = address.lower().strip()
        text = self._remove_accents(text)

        replacements = {
            r"\bstreet\b": "st",
            r"\bavenue\b": "ave",
            r"\bboulevard\b": "blvd",
            r"\broad\b": "rd",
            r"\bdrive\b": "dr",
            r"\blane\b": "ln",
            r"\bcourt\b": "ct",
            r"\bplace\b": "pl",
            r"\bapartment\b": "apt",
            r"\bsuite\b": "ste",
            r"\bbuilding\b": "bldg",
            r"\bnorth\b": "n",
            r"\bsouth\b": "s",
            r"\beast\b": "e",
            r"\bwest\b": "w",
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        for full, abbrev in self.STATE_ABBREVS.items():
            text = re.sub(rf"\b{re.escape(full)}\b", abbrev, text)

        text = re.sub(r"[^\w\s]", " ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _remove_accents(self, text: str) -> str:
        """Remove accents and diacritics from text."""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def get_blocking_key(self, text: str, method: str = "first_3_chars") -> str:
        """Generate a blocking key for a text.

        Args:
            text: Normalized text.
            method: Blocking method to use.

        Returns:
            Blocking key string.
        """
        if not text:
            return ""

        if method == "first_3_chars":
            cleaned = text.replace(" ", "")
            return cleaned[:3].lower()

        elif method == "first_word":
            words = text.split()
            return words[0].lower() if words else ""

        elif method == "metaphone":
            vowels = set("aeiou")
            consonants = [c for c in text.lower() if c.isalpha() and c not in vowels]
            return "".join(consonants[:4])

        else:
            return text[:3].lower()
