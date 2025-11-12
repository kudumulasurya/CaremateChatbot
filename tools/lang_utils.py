from langdetect import detect, DetectorFactory

# langdetect can be non-deterministic; seed for reproducibility
DetectorFactory.seed = 0

SUPPORTED_LANGUAGES = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'hi': 'Hindi',
    'zh-cn': 'Chinese',
    'zh-tw': 'Chinese',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ar': 'Arabic'
}


def detect_language(text: str) -> str:
    """Detect the language of the provided text. Returns an ISO 639-1 code.

    Falls back to 'en' when detection fails or text is empty.
    """
    try:
        if not text or not text.strip():
            return 'en'
        lang = detect(text)
        # langdetect returns codes like 'zh-cn' sometimes; normalize to primary
        lang = lang.lower()
        if lang in SUPPORTED_LANGUAGES:
            return lang
        # Try primary subtag
        if '-' in lang:
            primary = lang.split('-')[0]
            return primary
        return lang
    except Exception:
        return 'en'
