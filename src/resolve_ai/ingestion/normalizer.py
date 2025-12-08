"""Text normalization for entity resolution.

this module does all the heavy lifting for making messy real-world data comparable.
spent way too much time on this but it makes a huge difference in match quality.

similar to what I built at my last job but cleaner - that version had too many
special cases bolted on over time.
"""

import re
import unicodedata


class TextNormalizer:
    """Normalize text for consistent matching.

    the goal here is to make "ACME Corporation" match "Acme Corp." and similar
    variations. not perfect but handles the common cases pretty well.
    """

    # Common company suffixes to standardize
    # NOTE: the order doesn't matter in the dict, but we sort by length when applying
    # (see normalize_name) so longer phrases match before shorter ones
    COMPANY_SUFFIXES = {
        "incorporated": "inc",
        "corporation": "corp",
        "limited": "ltd",
        "company": "co",
        "limited liability company": "llc",
        "l.l.c.": "llc",
        "l.l.c": "llc",  # some people don't put the trailing period, annoyingly
        "p.l.c.": "plc",
        "p.l.c": "plc",
        "& co": "co",
        "&co": "co",
        # TODO: add more international suffixes (gmbh, sarl, etc) if we expand to non-US data
    }

    # State abbreviation mappings
    # verbose but easier to read than loading from a file, and states don't change often
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

        # Convert to lowercase and strip - always do this first
        text = name.lower().strip()

        # Remove accents/diacritics early so we don't have to think about them later
        text = self._remove_accents(text)

        # Standardize company suffixes BEFORE removing punctuation
        # the suffix replacement order matters here - longer phrases first or
        # 'limited liability company' won't match before 'limited' and 'company'
        # get replaced separately (learned this the hard way)
        for full, abbrev in sorted(
            self.COMPANY_SUFFIXES.items(), key=lambda x: len(x[0]), reverse=True
        ):
            text = re.sub(rf"\b{re.escape(full)}\b", abbrev, text)

        # Remove punctuation except for useful separators
        # keeping & and - because they're meaningful ("AT&T", "Coca-Cola")
        text = re.sub(r"[^\w\s&-]", " ", text)

        # Handle patterns that might remain after punctuation removal (like "l l c")
        # this catches the case where "L.L.C." becomes "l l c" after punctuation removal
        # kinda hacky but works
        text = re.sub(r"\bl\s*l\s*c\b", "llc", text)
        text = re.sub(r"\bp\s*l\s*c\b", "plc", text)

        # Collapse multiple spaces - do this last
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def normalize_address(self, address: str) -> str:
        """Normalize an address.

        addresses are honestly harder than names because there's so many formats.
        this handles the common US patterns but might need work for international.

        Args:
            address: Raw address string.

        Returns:
            Normalized address for matching.
        """
        if not address:
            return ""

        # Convert to lowercase and strip
        text = address.lower().strip()

        # Remove accents/diacritics
        text = self._remove_accents(text)

        # Standardize common abbreviations
        # TODO: handle unicode edge cases better (like fancy quotes in "123 Main St.")
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
            # could add more but these cover like 90% of cases
        }

        for pattern, replacement in replacements.items():
            text = re.sub(pattern, replacement, text)

        # Standardize state names - do this after other replacements so we don't
        # accidentally match partial words
        for full, abbrev in self.STATE_ABBREVS.items():
            text = re.sub(rf"\b{re.escape(full)}\b", abbrev, text)

        # Remove punctuation
        text = re.sub(r"[^\w\s]", " ", text)

        # Collapse multiple spaces
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def _remove_accents(self, text: str) -> str:
        """Remove accents and diacritics from text.

        turns "cafe" into "cafe", "nino" into "nino", etc.
        """
        # using NFKD normalization because it handles more unicode edge cases than NFD
        # (like ligatures and weird spacing characters)
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    def get_blocking_key(self, text: str, method: str = "first_3_chars") -> str:
        """Generate a blocking key for a text.

        blocking is how we avoid O(n^2) comparisons - instead of comparing every
        record to every other record, we only compare records that share the same
        blocking key. huge performance win for large datasets.

        Args:
            text: Normalized text.
            method: Blocking method to use.

        Returns:
            Blocking key string.
        """
        if not text:
            return ""

        if method == "first_3_chars":
            # Remove spaces and get first 3 chars - simple but effective
            # works well for company names that start with the actual name
            cleaned = text.replace(" ", "")
            return cleaned[:3].lower()

        elif method == "first_word":
            # better for datasets where first word is usually the key identifier
            words = text.split()
            return words[0].lower() if words else ""

        elif method == "metaphone":
            # Simplified metaphone - just first consonant sounds
            # this is NOT real metaphone, just a quick approximation
            # might want to use jellyfish or something for real phonetic matching
            vowels = set("aeiou")
            consonants = [c for c in text.lower() if c.isalpha() and c not in vowels]
            return "".join(consonants[:4])

        else:
            # fallback - shouldn't really get here but just in case
            return text[:3].lower()
