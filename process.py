import argparse
import re

import pandas as pd
import spacy
from spacy.matcher import Matcher
from tqdm import tqdm


def process(df: pd.DataFrame):
    nlp = spacy.load("en")
    matcher = Matcher(nlp.vocab)

    rows = []

    not_proper_noun = [
        {"POS": {"NOT_IN": ["PROPN", "PUNCT", "SPACE", "SYM", "NUM", "X"]}},
        {"POS": {"NOT_IN": ["PROPN", "PUNCT", "SPACE", "SYM", "NUM", "X"]}},
        {"POS": {"NOT_IN": ["PROPN", "PUNCT", "SPACE", "SYM", "NUM", "X"]}},
        {"POS": {"NOT_IN": ["PROPN", "PUNCT", "SPACE", "SYM", "NUM", "X"]}, "OP": "+"}
    ]

    matcher.add(f"NoProperNouns", None, not_proper_noun)

    for i, opinion in tqdm(df.iterrows(), total=len(df)):
        doc = nlp(re.sub(r"\s?\n\s+", " ", opinion.text))

        for sent in doc.sents:

            # skip direct quotations
            if '"' in sent[0].text and '"' in sent[-1].text:
                continue

            sent_doc = sent.as_doc()

            # print(f'\n{sent}'.replace("\n", " "))

            for match_id, start, end in matcher(sent_doc):
                # print(f"phrase: {sent_doc[start:end].text}")

                rows.append({
                    "filename": opinion.filename,
                    "phrase": sent_doc[start:end].text.lower(),
                    "lemmas": " ".join([t.lemma_ for t in sent_doc[start:end]]),
                    "length": end - start
                })

    phrases = pd.DataFrame(rows)

    phrases.to_csv("data/phrases.csv", index=False)
    phrases.to_pickle("data/phrases.pkl")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    if args.input_file.endswith("pkl"):
        df = pd.read_pickle(args.input_file)
    elif args.input_file.endswith("csv"):
        df = pd.read_csv(args.input_file)
    else:
        raise ValueError("unknown input file type", args.input_file)

    # majority opinions only for now
    df = df[df.opinion_type == "MO"]

    if args.debug:
        df = df[df.year > "2000"]
        df = df.head(500)

    process(df)


if __name__ == '__main__':
    main()
