# Vietnamese Language Support Plan for Chatterbox TTS

## Overview

This document outlines the requirements and steps to add Vietnamese language support to Chatterbox TTS. Currently, the system supports 23 languages but does not include Vietnamese.

**Estimated Complexity:** Medium-High (comparable to Japanese/Korean implementation)

---

## Current Architecture Summary

Chatterbox TTS consists of three main components:
1. **T3 Model** - Text → Speech tokens (Llama 3 backbone, 520M params)
2. **S3Gen** - Speech tokens → Audio waveform (Flow-matching + HiFTNet vocoder)
3. **VoiceEncoder** - Reference audio → Speaker embeddings (256-dim)

Language support is handled through:
- Language-specific text preprocessing in `tokenizer.py`
- Language token prefix `[{lang_id}]` prepended to text
- BPE tokenizer with multilingual vocabulary (~2454 tokens)

---

## Vietnamese Language Characteristics

### Key Challenges

| Feature | Description | Impact |
|---------|-------------|--------|
| **Tonal System** | 6 tones (level, rising, falling, question, tumbling, heavy) | Critical for meaning - "ma" has 6+ different meanings |
| **Diacritics** | Heavy use of diacritical marks (à, á, ả, ã, ạ, etc.) | Must be preserved exactly |
| **Word Boundaries** | No spaces between syllables in compound words | May need segmentation |
| **Latin Script** | Uses Latin alphabet with modifications | Potentially compatible with existing BPE |

### Tone Marks

Vietnamese tone marks are essential for correct pronunciation:
- `a` (ngang - level)
- `á` (sắc - rising)
- `à` (huyền - falling)
- `ả` (hỏi - question)
- `ã` (ngã - tumbling)
- `ạ` (nặng - heavy)

---

## Implementation Requirements

### Phase 1: Text Processing (Inference Only)

#### 1.1 Create Vietnamese Normalizer

**File:** `src/chatterbox/models/tokenizers/vietnamese.py`

```python
import unicodedata
from typing import Optional

# Optional: Install underthesea for word segmentation
try:
    from underthesea import word_tokenize
    HAS_UNDERTHESEA = True
except ImportError:
    HAS_UNDERTHESEA = False


def vietnamese_normalize(text: str, use_segmentation: bool = False) -> str:
    """
    Normalize Vietnamese text for TTS processing.

    Args:
        text: Input Vietnamese text
        use_segmentation: Whether to apply word segmentation

    Returns:
        Normalized text with preserved tone marks
    """
    # Normalize Unicode (NFC to keep composed characters)
    text = unicodedata.normalize('NFC', text)

    # Optional: Word segmentation for compound words
    if use_segmentation and HAS_UNDERTHESEA:
        words = word_tokenize(text, format="text")
        text = words

    return text


def validate_vietnamese_text(text: str) -> bool:
    """Check if text contains valid Vietnamese characters."""
    vietnamese_chars = set(
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
    for char in text:
        if char.isalpha() and char not in vietnamese_chars and not char.isascii():
            return False
    return True
```

#### 1.2 Integrate into MTLTokenizer

**File:** `src/chatterbox/models/tokenizers/tokenizer.py`

Add import:
```python
from .vietnamese import vietnamese_normalize
```

In `encode()` method, add Vietnamese handling:
```python
elif language_id == 'vi':
    txt = vietnamese_normalize(txt)
```

#### 1.3 Register Vietnamese in Supported Languages

**File:** `src/chatterbox/mtl_tts.py`

```python
SUPPORTED_LANGUAGES = {
    # ... existing languages ...
    "vi": "Vietnamese",
}
```

#### 1.4 Add Dependencies

**File:** `pyproject.toml`

```toml
[project.optional-dependencies]
vietnamese = ["underthesea>=6.0.0"]
```

---

### Phase 2: UI Integration

#### 2.1 Add Vietnamese Sample

**File:** `multilingual_app.py`

```python
LANGUAGE_CONFIG = {
    # ... existing ...
    "vi": {
        "audio": "https://path-to-vietnamese-sample.flac",
        "text": "Xin chào, đây là hệ thống tổng hợp giọng nói tiếng Việt của Chatterbox.",
    },
}
```

---

### Phase 3: Tokenizer Vocabulary (Required for Quality)

The current multilingual tokenizer vocabulary may not fully cover Vietnamese diacritics combinations. Options:

#### Option A: Test Existing Tokenizer (Quick)
- Test if current BPE tokenizer handles Vietnamese characters
- Risk: May produce many `[UNK]` tokens for uncommon diacritic combinations

#### Option B: Expand Vocabulary (Recommended)
- Collect Vietnamese text corpus
- Train new BPE tokenizer including Vietnamese
- Merge with existing vocabulary
- Update `grapheme_mtl_merged_expanded_v1.json`

**Vocabulary Estimation:**
- Additional Vietnamese-specific tokens needed: ~100-200
- Total vocabulary would increase from 2454 to ~2600

---

### Phase 4: Model Training (Required for Production)

#### 4.1 Data Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| Audio Duration | 100 hours | 500+ hours |
| Speakers | 50+ | 200+ |
| Audio Quality | 16kHz mono | 24kHz+ |
| Transcription | Word-level | Phoneme-level |

#### 4.2 Data Sources

Potential Vietnamese speech datasets:

| Dataset | Description | Size | License |
|---------|-------------|------|---------|
| VIVOS | Vietnamese speech corpus | ~15 hours | CC BY-NC-SA |
| VietTTS | Vietnamese TTS dataset | ~30 hours | Research |
| Common Voice | Crowdsourced | ~100+ hours | CC0 |
| VnSpeechCorpus | Multi-speaker | ~70 hours | Research |
| FPT OpenSpeech | Vietnamese ASR | Large | Commercial |

#### 4.3 Training Strategy

**Option A: Full Retrain (Best Quality)**
- Retrain T3 model with Vietnamese added to 23-language dataset
- Requires: Full training infrastructure, GPU cluster
- Timeline: Weeks to months

**Option B: Fine-tune (Practical)**
1. Start from multilingual T3 checkpoint (`t3_mtl23ls_v2.safetensors`)
2. Freeze early transformer layers (language-agnostic)
3. Fine-tune:
   - Text embedding layer (add Vietnamese tokens)
   - Later transformer layers
   - Output projection
4. Joint fine-tune with small amount of other language data (prevent forgetting)

**Training Configuration (estimated):**
```python
training_config = {
    "learning_rate": 1e-5,  # Low LR for fine-tuning
    "batch_size": 32,
    "epochs": 50-100,
    "warmup_steps": 1000,
    "freeze_layers": [0, 1, 2, 3, 4, 5],  # Freeze first 6 layers
    "gradient_accumulation": 4,
}
```

#### 4.4 S3Gen Considerations

The S3Gen model (speech token → audio) may need minimal fine-tuning:
- Vietnamese phonetics share many sounds with existing languages
- Tone generation might require additional attention
- VoiceEncoder should work for Vietnamese speakers without changes

---

### Phase 5: Evaluation

#### 5.1 Objective Metrics

| Metric | Target | Tool |
|--------|--------|------|
| MOS (Mean Opinion Score) | > 3.5 | Human evaluation |
| Character Error Rate | < 5% | ASR-based |
| Tone Accuracy | > 90% | Manual annotation |
| Speaker Similarity | > 0.7 | Cosine similarity |

#### 5.2 Test Cases

```python
test_sentences = [
    # Basic
    "Xin chào, tôi là trợ lý ảo.",

    # Tone minimal pairs
    "Ma, má, mà, mả, mã, mạ.",

    # Complex tones
    "Ông ấy đã đến đây hôm qua.",

    # Numbers
    "Một, hai, ba, bốn, năm.",

    # Question
    "Bạn có khỏe không?",

    # Long sentence
    "Việt Nam là một quốc gia nằm ở phía đông bán đảo Đông Dương.",
]
```

---

## Implementation Steps Summary

### Quick Start (Inference Only, Limited Quality)

1. [ ] Create `src/chatterbox/models/tokenizers/vietnamese.py`
2. [ ] Update `tokenizer.py` with Vietnamese handling
3. [ ] Add `"vi": "Vietnamese"` to `SUPPORTED_LANGUAGES`
4. [ ] Test with existing tokenizer vocabulary
5. [ ] Add Vietnamese sample to `multilingual_app.py`

### Production Quality

6. [ ] Collect Vietnamese speech dataset (100+ hours)
7. [ ] Preprocess and validate audio quality
8. [ ] Train/expand BPE tokenizer vocabulary
9. [ ] Fine-tune T3 model on Vietnamese data
10. [ ] Evaluate tone accuracy and MOS
11. [ ] Optimize for edge cases (compound words, foreign names)

---

## Dependencies

### Required
- `unicodedata` (stdlib)

### Optional (Recommended)
```bash
uv add underthesea  # Vietnamese NLP toolkit
# Or install with optional extra:
uv sync --extra vietnamese
```

### Alternative Libraries
- `pyvi` - Vietnamese text processing
- `vncorenlp` - Vietnamese NLP (Java-based)
- `g2p-vi` - Grapheme-to-phoneme for Vietnamese

To add alternative libraries:
```bash
uv add pyvi
# or
uv add g2p-vi
```

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| Tone confusion | High | High | Fine-tune with tone-annotated data |
| Vocabulary gaps | Medium | Medium | Expand tokenizer vocabulary |
| Regional accent variation | Medium | Low | Include diverse speaker data |
| Compound word boundaries | Medium | Low | Use segmentation tool |

---

## Timeline Estimate

| Phase | Duration | Dependencies |
|-------|----------|--------------|
| Phase 1: Text Processing | 1-2 days | None |
| Phase 2: UI Integration | 0.5 days | Phase 1 |
| Phase 3: Vocabulary Expansion | 1 week | Vietnamese text corpus |
| Phase 4: Model Training | 2-4 weeks | GPU resources, speech data |
| Phase 5: Evaluation | 1 week | Trained model |

**Total:** 1-2 months for production-quality Vietnamese TTS

---

## References

- [Vietnamese phonology](https://en.wikipedia.org/wiki/Vietnamese_phonology)
- [Underthesea - Vietnamese NLP](https://github.com/undertheseanlp/underthesea)
- [Common Voice Vietnamese](https://commonvoice.mozilla.org/vi)
- [VIVOS Dataset](https://ailab.hcmus.edu.vn/vivos)
