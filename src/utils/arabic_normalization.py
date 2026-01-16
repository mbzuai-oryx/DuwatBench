#!/usr/bin/env python3
"""
Arabic text normalization utilities
Based on paper Section 3.1 (lines 236-244):
"We apply an Arabic-aware normalization pipeline before scoring.
This includes Unicode normalization, removal of tatweel, and unification
of character variants such as Alef forms and Alef Maqsurah"
"""

import re
import unicodedata


# Arabic diacritics (harakat/tashkeel) Unicode ranges
ARABIC_DIACRITICS = (
    '\u064B'  # FATHATAN
    '\u064C'  # DAMMATAN
    '\u064D'  # KASRATAN
    '\u064E'  # FATHA
    '\u064F'  # DAMMA
    '\u0650'  # KASRA
    '\u0651'  # SHADDA
    '\u0652'  # SUKUN
    '\u0653'  # MADDAH ABOVE
    '\u0654'  # HAMZA ABOVE
    '\u0655'  # HAMZA BELOW
    '\u0656'  # SUBSCRIPT ALEF
    '\u0657'  # INVERTED DAMMA
    '\u0658'  # MARK NOON GHUNNA
    '\u0659'  # ZWARAKAY
    '\u065A'  # VOWEL SIGN SMALL V ABOVE
    '\u065B'  # VOWEL SIGN INVERTED SMALL V ABOVE
    '\u065C'  # VOWEL SIGN DOT BELOW
    '\u065D'  # REVERSED DAMMA
    '\u065E'  # FATHA WITH TWO DOTS
    '\u065F'  # WAVY HAMZA BELOW
    '\u0670'  # SUPERSCRIPT ALEF
)

# Compile regex pattern for diacritics removal
DIACRITICS_PATTERN = re.compile(f'[{ARABIC_DIACRITICS}]')


class ArabicNormalizer:
    """
    Normalizes Arabic text following CAMeL Tools and Lucene Arabic normalizer
    standards as mentioned in the paper.
    """

    def __init__(self, remove_tatweel=True, normalize_alef=True,
                 normalize_alef_maqsurah=True, unicode_normalize=True,
                 remove_diacritics=True):
        self.remove_tatweel = remove_tatweel
        self.normalize_alef = normalize_alef
        self.normalize_alef_maqsurah = normalize_alef_maqsurah
        self.unicode_normalize = unicode_normalize
        self.remove_diacritics = remove_diacritics

    def normalize(self, text):
        """Apply all normalization steps to Arabic text"""
        if not text:
            return ""

        # Unicode normalization (NFC form)
        if self.unicode_normalize:
            text = unicodedata.normalize('NFC', text)

        # Remove Arabic diacritics (harakat/tashkeel)
        # This is critical for fair comparison as models may not include diacritics
        if self.remove_diacritics:
            text = DIACRITICS_PATTERN.sub('', text)

        # Remove tatweel (ـ)
        if self.remove_tatweel:
            text = re.sub('[ـ]', '', text)

        # Normalize Alef variants: أ إ آ ٱ → ا
        if self.normalize_alef:
            text = re.sub('[أإآٱ]', 'ا', text)

        # Normalize Alef Maqsurah: ى → ي
        if self.normalize_alef_maqsurah:
            text = re.sub('ى', 'ي', text)

        # Remove zero-width characters
        text = re.sub('[\u200b\u200c\u200d\u200e\u200f\ufeff]', '', text)

        # Normalize whitespace
        text = ' '.join(text.split())

        return text.strip()

    def normalize_for_comparison(self, text):
        """
        Normalize text specifically for metric comparison
        More aggressive than display normalization
        """
        text = self.normalize(text)

        # Additional normalization for comparison
        # Normalize different types of spaces
        text = re.sub('[\u00a0\u1680\u2000-\u200a\u202f\u205f\u3000]', ' ', text)

        # Normalize Arabic presentation forms to basic forms
        # Presentation Form A (U+FB50-U+FDFF)
        # Presentation Form B (U+FE70-U+FEFF)
        text = unicodedata.normalize('NFKC', text)

        return text.strip()


def normalize_prediction_and_reference(prediction, reference, normalizer=None):
    """
    Normalize both prediction and reference for fair comparison

    Args:
        prediction: Model output text
        reference: Ground truth text
        normalizer: ArabicNormalizer instance (creates default if None)

    Returns:
        Tuple of (normalized_prediction, normalized_reference)
    """
    if normalizer is None:
        normalizer = ArabicNormalizer()

    norm_pred = normalizer.normalize_for_comparison(prediction)
    norm_ref = normalizer.normalize_for_comparison(reference)

    return norm_pred, norm_ref


# Singleton normalizer instance
_default_normalizer = None

def get_default_normalizer():
    """Get or create the default normalizer instance"""
    global _default_normalizer
    if _default_normalizer is None:
        _default_normalizer = ArabicNormalizer()
    return _default_normalizer
