# Copyright (c) 2025
# Vietnamese text normalization for Chatterbox TTS
# MIT License

import unicodedata
import re
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# Vietnamese character set for validation
VIETNAMESE_CHARS = set(
    'aàảãáạăằẳẵắặâầẩẫấậ'
    'eèẻẽéẹêềểễếệ'
    'iìỉĩíị'
    'oòỏõóọôồổỗốộơờởỡớợ'
    'uùủũúụưừửữứự'
    'yỳỷỹýỵ'
    'đ'
    'AÀẢÃÁẠĂẰẲẴẮẶÂẦẨẪẤẬ'
    'EÈẺẼÉẸÊỀỂỄẾỆ'
    'IÌỈĨÍỊ'
    'OÒỎÕÓỌÔỒỔỖỐỘƠỜỞỠỚỢ'
    'UÙỦŨÚỤƯỪỬỮỨỰ'
    'YỲỶỸÝỴ'
    'Đ'
)

# Common Vietnamese abbreviations and their expansions
# Only include abbreviations with periods to avoid matching partial words
ABBREVIATIONS = {
    'tp.': 'thành phố',
    'TP.': 'thành phố',
    'ths.': 'thạc sĩ',
    'ThS.': 'thạc sĩ',
    'ts.': 'tiến sĩ',
    'TS.': 'tiến sĩ',
    'gs.': 'giáo sư',
    'GS.': 'giáo sư',
    'pgs.': 'phó giáo sư',
    'PGS.': 'phó giáo sư',
    'q.': 'quận',
    'Q.': 'quận',
    'p.': 'phường',
    'P.': 'phường',
    'tx.': 'thị xã',
    'TX.': 'thị xã',
    'tt.': 'thị trấn',
    'TT.': 'thị trấn',
}

# Number words in Vietnamese
NUMBER_WORDS = {
    '0': 'không',
    '1': 'một',
    '2': 'hai',
    '3': 'ba',
    '4': 'bốn',
    '5': 'năm',
    '6': 'sáu',
    '7': 'bảy',
    '8': 'tám',
    '9': 'chín',
    '10': 'mười',
}


def vietnamese_normalize(text: str, expand_abbreviations: bool = True) -> str:
    """
    Normalize Vietnamese text for TTS processing.

    Args:
        text: Input Vietnamese text
        expand_abbreviations: Whether to expand common abbreviations

    Returns:
        Normalized text with preserved tone marks
    """
    if not text:
        return text

    # Normalize Unicode to NFC (composed form) - important for Vietnamese
    # NFC keeps characters like 'ố' as single characters
    text = unicodedata.normalize('NFC', text)

    # Expand abbreviations
    if expand_abbreviations:
        for abbr, expansion in ABBREVIATIONS.items():
            # Match at word boundaries only (start of string or after whitespace)
            # For abbreviations ending with period, look for word boundary before
            pattern = re.compile(r'(?:^|(?<=\s))' + re.escape(abbr), re.IGNORECASE)
            text = pattern.sub(expansion, text)

    # Normalize whitespace
    text = ' '.join(text.split())

    # Remove or replace problematic characters while keeping Vietnamese
    # Keep: Vietnamese chars, ASCII letters, numbers, basic punctuation
    cleaned = []
    for char in text:
        if char.isascii() or char in VIETNAMESE_CHARS or char in ' .,!?;:\'"()-':
            cleaned.append(char)
        elif unicodedata.category(char).startswith('L'):
            # Keep other letters (might be Vietnamese combining chars)
            cleaned.append(char)

    text = ''.join(cleaned)

    return text.strip()


def number_to_vietnamese(num: int) -> str:
    """Convert a number to Vietnamese words."""
    if num < 0:
        return 'âm ' + number_to_vietnamese(-num)

    if num == 0:
        return 'không'

    if num < 10:
        return NUMBER_WORDS[str(num)]

    if num < 20:
        if num == 10:
            return 'mười'
        ones = num % 10
        if ones == 5:
            return 'mười lăm'
        return 'mười ' + NUMBER_WORDS[str(ones)]

    if num < 100:
        tens = num // 10
        ones = num % 10
        result = NUMBER_WORDS[str(tens)] + ' mươi'
        if ones == 0:
            return result
        if ones == 1:
            return result + ' mốt'
        if ones == 5:
            return result + ' lăm'
        return result + ' ' + NUMBER_WORDS[str(ones)]

    if num < 1000:
        hundreds = num // 100
        remainder = num % 100
        result = NUMBER_WORDS[str(hundreds)] + ' trăm'
        if remainder == 0:
            return result
        if remainder < 10:
            return result + ' lẻ ' + NUMBER_WORDS[str(remainder)]
        return result + ' ' + number_to_vietnamese(remainder)

    if num < 1000000:
        thousands = num // 1000
        remainder = num % 1000
        result = number_to_vietnamese(thousands) + ' nghìn'
        if remainder == 0:
            return result
        if remainder < 100:
            return result + ' không trăm ' + number_to_vietnamese(remainder)
        return result + ' ' + number_to_vietnamese(remainder)

    # For larger numbers, just read digits
    return ' '.join(NUMBER_WORDS.get(d, d) for d in str(num))


def expand_numbers(text: str) -> str:
    """Expand numbers in text to Vietnamese words."""
    def replace_number(match):
        num_str = match.group(0)
        try:
            num = int(num_str)
            return number_to_vietnamese(num)
        except ValueError:
            return num_str

    # Replace standalone numbers (not part of words)
    return re.sub(r'\b\d+\b', replace_number, text)


def validate_vietnamese_text(text: str) -> bool:
    """Check if text contains valid Vietnamese characters."""
    for char in text:
        if char.isalpha() and char not in VIETNAMESE_CHARS and not char.isascii():
            return False
    return True


def preprocess_for_tts(text: str) -> str:
    """
    Full preprocessing pipeline for Vietnamese TTS.

    Args:
        text: Raw Vietnamese text

    Returns:
        Preprocessed text ready for tokenization
    """
    # Step 1: Basic normalization
    text = vietnamese_normalize(text)

    # Step 2: Expand numbers
    text = expand_numbers(text)

    # Step 3: Handle punctuation for better prosody
    # Add slight pause markers (spaces) around punctuation
    text = re.sub(r'([.,!?;:])', r' \1 ', text)
    text = ' '.join(text.split())  # Normalize whitespace again

    # Step 4: Capitalize first letter
    if text and text[0].islower():
        text = text[0].upper() + text[1:]

    return text


# Optional: Integration with underthesea for word segmentation
try:
    from underthesea import word_tokenize as vn_word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False
    vn_word_tokenize = None


def segment_words(text: str) -> str:
    """
    Segment Vietnamese text into words using underthesea.
    Falls back to original text if underthesea is not available.
    """
    if HAS_UNDERTHESEA and vn_word_tokenize:
        try:
            return vn_word_tokenize(text, format="text")
        except Exception as e:
            logger.warning(f"Word segmentation failed: {e}")
            return text
    return text
