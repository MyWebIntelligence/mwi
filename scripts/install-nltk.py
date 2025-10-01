import nltk

# Ensure both resources are present for NLTK>=3.8
for res in ("punkt", "punkt_tab"):
    try:
        nltk.data.find(f"tokenizers/{res}")
    except LookupError:
        nltk.download(res)
