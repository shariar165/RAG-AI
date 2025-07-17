import re

def clean_text(text):
    """
    Cleans extracted text by removing extra spaces and newlines.
    """
    text = re.sub(r'\s+', ' ', text)        # Replace multiple spaces with one
    text = re.sub(r'\n+', '\n', text)       # Normalize newlines
    return text.strip()
