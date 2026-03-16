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


def normalize_text(text):

    text = str(text).lower()
    text = re.sub(r"[^a-z0-9\s]"," ",text)
    text = re.sub(r"\s+"," ",text).strip()

    return text


def linguistic_normalize(text):

    doc = nlp(text)

    tokens = []

    for token in doc:

        lemma = token.lemma_.lower()

        if lemma in NEUTRAL_WORDS or lemma in GENERIC_TERMS:
            continue

        if token.pos_ in {"NOUN","ADJ","PROPN"}:
            tokens.append(lemma)

    return " ".join(tokens)