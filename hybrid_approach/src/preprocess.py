import re
import spacy

nlp = spacy.load("en_core_web_sm")

NEUTRAL_WORDS = {
    "for","with","on","at","to","by","of","in","upto","and"
}

GENERIC_TERMS = {
    "service","services",
    "center","centers","centre","centres",
    "clinic","clinics",
    "shop","shops",
    "store","stores",
    "class","classes",
    "company","companies",
    "hub","hubs","station",
    "cafe","stations","repair"
}

import re

def linguistic_normalize(text):

    # ---------------------------------------
    # 1. Basic cleaning
    # ---------------------------------------
    text = text.lower()
    text = re.sub(r"\d+", "", text)              # remove numbers
    text = re.sub(r"[^a-zA-Z\s]", " ", text)     # remove symbols
    text = re.sub(r"\s+", " ", text).strip()

    doc = nlp(text)

    tokens = []

    for token in doc:

        lemma = token.lemma_.lower()

        # ---------------------------------------
        # 2. Skip useless tokens
        # ---------------------------------------
        if lemma in NEUTRAL_WORDS or lemma in GENERIC_TERMS:
            continue

        if token.is_stop:
            continue

        # ---------------------------------------
        # 3. Keep meaningful POS
        # ---------------------------------------
        if token.pos_ in {"NOUN", "ADJ", "PROPN", "VERB"}:
            tokens.append(lemma)

    # ---------------------------------------
    # 4. Phrase normalization (VERY IMPORTANT)
    # ---------------------------------------
    text = " ".join(tokens)

    return text