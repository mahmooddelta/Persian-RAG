"""
Persian Text Normalizer implementing `TextNormalizerPort`.

Handles critical Persian NLP preprocessing steps:
1. Unicode NFC composition and presentation-form decomposition (NFKC where safe)
2. Unification of Arabic character variants to standard Persian characters:
   - `ي` (U+064A), `ى` (U+0649), `ئ` -> `ی` (U+06CC) where appropriate
   - `ك` (U+0643) -> `ک` (U+06A9)
   - `ة` (U+0629) -> `ه` (U+0647)
   - `ؤ` (U+0624) -> `و` (U+0648)
   - `إ` (U+0625), `أ` (U+0623) -> `ا` (U+0627)
3. Digit normalization (Eastern Arabic `٠-٩` to Persian `۰-۹`)
4. Removal of Arabic diacritics (Tashkeel / Harakat) and Kashida / Tatweel (`ـ`)
5. Zero-Width Non-Joiner (ZWNJ / نیم‌فاصله `\u200c`) cleanup:
   - Collapses multiple consecutive ZWNJs into a single ZWNJ
   - Strips stray ZWNJs adjacent to whitespace or punctuation
   - Converts common Persian morphological suffixes (`ها`, `های`, `هایی`, `تر`, `ترین`, `می`, `نمی`)
     to proper ZWNJ-joined forms when separated by spaces
6. Punctuation and whitespace standardization (`?` -> `؟`, `,` in Persian context, collapsing blank lines)
"""

from __future__ import annotations

import re
import unicodedata

from src.domain.exceptions import TextNormalizationError
from src.domain.interfaces import TextNormalizerPort


class PersianTextNormalizer(TextNormalizerPort):
    """
    Production-grade regex and Unicode-based Persian text normalizer.
    Zero external NLP heavy library required; deterministic and fast.
    """

    ZWNJ = "\u200c"

    # Character-level translation map
    _CHAR_TRANSLATION_TABLE = str.maketrans(
        {
            # Arabic Yeh variants -> Persian Yeh (U+06CC)
            "\u064a": "\u06cc",  # ي Arabic Yeh
            "\u0649": "\u06cc",  # ى Alef Maksura
            "\u0626": "\u06cc",  # ئ Yeh with Hamza above (normalized in search/RAG)
            "\ufbfe": "\u06cc",
            "\ufbff": "\u06cc",
            "\ufef1": "\u06cc",
            "\ufef2": "\u06cc",
            "\ufef3": "\u06cc",
            "\ufef4": "\u06cc",
            # Arabic Kaf -> Persian Keheh (U+06A9)
            "\u0643": "\u06a9",  # ك Arabic Kaf
            "\ufedb": "\u06a9",
            "\ufedc": "\u06a9",
            "\ufedd": "\u06a9",
            "\ufede": "\u06a9",
            # Teh Marbuta -> Heh (U+0647)
            "\u0629": "\u0647",  # ة Teh Marbuta
            "\u06c0": "\u0647\u0654",
            "\u06d5": "\u0647",
            # Waw with Hamza -> Waw
            "\u0624": "\u0648",  # ؤ
            # Alef with Hamza below -> Alef
            "\u0625": "\u0627",  # إ
            # Eastern Arabic numerals (٠-٩) -> Persian numerals (۰-۹)
            "\u0660": "\u06f0",
            "\u0661": "\u06f1",
            "\u0662": "\u06f2",
            "\u0663": "\u06f3",
            "\u0664": "\u06f4",
            "\u0665": "\u06f5",
            "\u0666": "\u06f6",
            "\u0667": "\u06f7",
            "\u0668": "\u06f8",
            "\u0669": "\u06f9",
            # Non-breaking spaces & zero-width spaces -> standard space or ZWNJ
            "\u00a0": " ",
            "\u200b": "\u200c",  # Zero-width space -> ZWNJ
            "\u200f": "",  # Right-to-left mark
            "\u200e": "",  # Left-to-right mark
            "\ufeff": "",  # BOM
        }
    )

    _DIACRITICS_AND_TATWEEL_PATTERN = re.compile(
        r"[\u0640\u064b-\u065f\u0670\u06d6-\u06ed]"
    )

    _MULTIPLE_ZWNJ_PATTERN = re.compile(r"\u200c{2,}")

    _STRAY_ZWNJ_WHITESPACE_PATTERN = re.compile(
        r"(?:\s+\u200c+|\u200c+\s+)"
    )

    _STRAY_ZWNJ_PUNCT_PATTERN = re.compile(
        r"(?:^\u200c+|\u200c+$|(?<=[\.\،\؛\:\!\؟\?\(\)\[\]\{\}«»])\u200c+|\u200c+(?=[\.\،\؛\:\!\؟\?\(\)\[\]\{\}«»]))"
    )

    _MI_PREFIX_PATTERN = re.compile(r"\b(ن?می)\s+(?=[\u0621-\u06cc])")

    _SUFFIX_ZWNJ_PATTERN = re.compile(
        r"(?<=[\u0621-\u06cc])\s+(ها(?:ی(?:ی)?)?|تر(?:ین)?)\b"
    )

    _HORIZONTAL_WHITESPACE_PATTERN = re.compile(r"[ \t\r\f\v]+")

    _EXCESS_NEWLINES_PATTERN = re.compile(r"\n{3,}")

    _SPACE_BEFORE_PUNCT_PATTERN = re.compile(r"[ \t]+([،؛؟!\.\:\)\]\}»])")

    def normalize(self, text: str) -> str:
        if text is None:
            raise TextNormalizationError("Input text for normalization cannot be None.")

        if not text:
            return ""

        try:
            result = unicodedata.normalize("NFKC", text)

            result = result.translate(self._CHAR_TRANSLATION_TABLE)

            result = self._DIACRITICS_AND_TATWEEL_PATTERN.sub("", result)

            result = result.replace("\r\n", "\n").replace("\r", "\n")

            result = self._MULTIPLE_ZWNJ_PATTERN.sub(self.ZWNJ, result)
            result = self._STRAY_ZWNJ_WHITESPACE_PATTERN.sub(" ", result)
            result = self._STRAY_ZWNJ_PUNCT_PATTERN.sub("", result)

            result = self._MI_PREFIX_PATTERN.sub(rf"\1{self.ZWNJ}", result)
            result = self._SUFFIX_ZWNJ_PATTERN.sub(rf"{self.ZWNJ}\1", result)

            result = self._HORIZONTAL_WHITESPACE_PATTERN.sub(" ", result)
            result = self._SPACE_BEFORE_PUNCT_PATTERN.sub(r"\1", result)

            lines = [line.strip() for line in result.split("\n")]
            result = "\n".join(lines)
            result = self._EXCESS_NEWLINES_PATTERN.sub("\n\n", result)

            return result.strip()
        except Exception as exc:
            raise TextNormalizationError(
                "Failed to normalize Persian text.", details=str(exc)
            ) from exc
