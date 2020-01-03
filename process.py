import argparse
import multiprocessing as mp

import numpy as np
import pandas as pd
import spacy
from spacy.matcher import Matcher


def process(df: pd.DataFrame):
    # disable named entity recognition and text categorization, since we aren't using them
    nlp = spacy.load("en", disable=["ner", "textcat"])

    matcher = Matcher(nlp.vocab)

    # the basic pattern matches tokens that are not punctuation, spaces, symbols, numbers, or unrecognized tokens
    unigram = [
        {"POS": {"NOT_IN": ["PUNCT", "SPACE", "SYM", "NUM", "X"]}}
    ]

    # optional comma allows for comma-delimited phrases
    another = [
        {"TEXT": ",", "OP": "?"},
        {"POS": {"NOT_IN": ["PUNCT", "SPACE", "SYM", "NUM", "X"]}}
    ]

    # add patterns for unigrams, bigrams, trigrams
    for length in range(0, 3):
        matcher.add(f"{length}", None, unigram + another * length)

    # use multiprocessing for speed, one batch per CPU
    # but at most use only as many CPUs as we have rows
    print("processing")
    with mp.Pool(min(mp.cpu_count(), len(df))) as pool:
        results = [pool.apply_async(worker, (sub_df, nlp, matcher)) for sub_df in np.array_split(df, mp.cpu_count())]
        results = [r.get() for r in results]

    print("combining results")
    phrases = pd.concat(results)

    return phrases


def worker(df, nlp, matcher):
    rows = []

    # for each opinion in the dataframe
    for i, opinion in df.iterrows():

        print("processing", opinion.filename)

        # parse with spaCy
        doc = nlp(opinion.text)

        # for each sentence in the opinion
        for sent in doc.sents:

            # skip direct quotations
            if '"' in sent[0].text and '"' in sent[-1].text:
                continue

            # convert the sentence span into a doc
            sent_doc = sent.as_doc()

            # add the entire sentence, assuming it has a verb
            # this avoids false positive "sentences" caused by citations
            if any([t.pos_ in ["VERB", "AUX"] for t in sent_doc]):
                rows.append({
                    "filename": opinion.filename,
                    "phrase": sent_doc.text.lower().strip(),
                    "length": "sentence"
                })

            # for each ngram match
            for match_id, start, end in matcher(sent_doc):
                phrase = sent_doc[start:end].text.lower().strip()

                # special case for underscores, which spaCy wrongly tags as nouns
                if "_" in phrase:
                    continue

                # add to the results
                rows.append({
                    "filename": opinion.filename,
                    "phrase": phrase,
                    # we use this instead of `end - start` so that we don't count commas as part of the length
                    "length": sent_doc.vocab.strings[match_id]
                })

    result = pd.DataFrame(rows)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    print("loading input file")
    df = pd.read_csv(args.input_file)

    if args.debug:
        print("subsetting input for debugging")
        df = df[df.opinion_type == "MO"]
        df = df[df.year > 2000]
        df = df.head(mp.cpu_count() * 100)

    phrases = process(df)

    print("writing out results")
    phrases.to_csv("data/phrases.csv", index=False)


if __name__ == '__main__':
    main()
