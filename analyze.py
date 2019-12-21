import argparse

import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lemmas", "-l", action="store_true")
    args = parser.parse_args()

    print("loading opinion data")
    opinions = pd.read_csv("data/opinions.csv", usecols=["filename", "year", "lexis_cite", "opinion_type", "author"])

    if args.lemmas:
        print("loading lemmatized phrases")
        phrases = pd.read_csv("data/phrases.csv", usecols=["filename", "length", "lemmas"]).rename(
            columns={"lemmas": "phrase"})
    else:
        print("loading verbatim phrases")
        phrases = pd.read_csv("data/phrases.csv", usecols=["filename", "length", "phrase"])

    print("merging phrases with opinion data")
    df = phrases.merge(opinions)

    # drop phrases that occur in the first 10 years on the assumption that they are commonplace
    print("dropping old phrases")
    old_phrases = df[df.year <= (df.year.min() + 10)].phrase
    df = df[~df.phrase.isin(old_phrases)]

    print("calculating phrase counts")
    phrase_counts = df.groupby(["phrase", "length"]).filename.count().reset_index().rename(
        columns={"filename": "count"}).sort_values(["count", "length"], ascending=[False, False])

    print("writing out phrase count data")
    if args.lemmas:
        outfilename = "lemmatized_phrase_counts.csv"
    else:
        outfilename = "phrase_counts.csv"
    phrase_counts.to_csv(f"data/{outfilename}", index=False)


if __name__ == '__main__':
    main()
