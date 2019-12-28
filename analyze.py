import argparse

import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    args = parser.parse_args()

    print("loading opinion data")
    opinions = pd.read_csv("data/opinions.csv", usecols=["filename", "year", "lexis_cite", "opinion_type", "author"])

    print("loading phrases")
    phrases = pd.read_csv("data/phrases.csv", usecols=["filename", "length", "phrase"])

    print("merging phrases with opinion data")
    df = phrases.merge(opinions)

    # sort df by the phrases, since that will be the most intensive search
    print("sorting by phrase")
    df = df.set_index("phrase").sort_index()

    antecedent_counts = []

    print("finding antecedents")

    # TODO parallelize this?  could be memory intensive given the size of the phrase df, but still
    #  worth it even for a few chunks

    # for each opinion in the most recent year
    for filename in tqdm(df[df.year == opinions.year.max()].filename.unique()):
        # get the opinion author
        author = opinions[opinions.filename == filename].author.iloc[0]

        # TODO generalize this to producing data for all phrase lengths

        # get this opinion's unique unigrams
        unigrams = df[(df.filename == filename)
                      & (df["length"] == 1)].index.unique().values

        # count how many also occur in opinions before this year and written by others
        antecedents = df[(df.index.isin(unigrams))
                         & (df.author != author)
                         & (df.year < opinions.year.max())
                         & (df["length"] == 1)].index.unique().values

        antecedent_counts.append({
            "filename": filename,
            "unique_unigrams": len(unigrams),
            "antecedents": len(antecedents)
        })

    antecedent_counts = pd.DataFrame(antecedent_counts)

    antecedent_counts.to_csv("data/antecedent_counts.csv", index=False)

    # # drop phrases that occur in the first 10 years on the assumption that they are commonplace
    # print("dropping old phrases")
    # old_phrases = df[df.year <= (df.year.min() + 10)].phrase
    # df = df[~df.phrase.isin(old_phrases)]


if __name__ == '__main__':
    main()
