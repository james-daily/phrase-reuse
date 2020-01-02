import argparse

import numpy as np
import pandas as pd
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true")
    args = parser.parse_args()

    print("loading opinion data")
    opinions = pd.read_csv("data/opinions.csv", usecols=["filename", "year", "lexis_cite", "opinion_type", "author"], )

    if args.debug:
        print("subsetting to 10% sample for debugging")
        opinion_sample = np.random.choice(opinions.filename, int(len(opinions) * 0.1), False)
        opinions = opinions[opinions.filename.isin(opinion_sample)]

    print("loading phrases")
    phrases = pd.read_csv("data/phrases.csv", usecols=["filename", "length", "phrase"])

    print("merging phrases with opinion data")
    df = phrases.merge(opinions)

    print("dropping phrases duplicated within the same opinion")
    df = df.drop_duplicates(subset=["filename", "phrase"])

    # sort df by the phrases, since that will be the most intensive search
    print("indexing and sorting by phrase")
    df = df.set_index("phrase").sort_index()

    # pre-compute which phrases occur within the first 5 years of the data
    print("pre-computing phrases from the first 5 years of data")
    old_phrases = df[df.year <= df.year.min() + 5]
    # only keep a single instance of each old phrase, since that's sufficient for this purpose
    old_phrases = old_phrases[~old_phrases.index.duplicated(keep="first")]

    # initialize results container
    results = []

    # TODO parallelize this?  could be memory intensive given the size of the phrase df, but still
    #  worth it even for a few chunks; runtime is around 4.5 hours right now

    # TODO should also profile this code to identify the slow spots

    # for each opinion in the most recent year
    for filename in tqdm(df[df.year == opinions.year.max()].filename.unique(), desc="finding antecedents"):
        # get the opinion author
        author = opinions[opinions.filename == filename].author.iloc[0]

        # for each phrase length
        for length in df["length"].unique():
            # get this opinion's unique phrases of that length
            phrases = df[(df.filename == filename)
                         & (df["length"] == length)]

            # count how many also occur in opinions before this year and written by others
            antecedents = df[(df.index.isin(phrases.index))
                             & (df.author != author)
                             & (df.year < opinions.year.max())
                             & (df["length"] == length)]

            # count how many of those phrases are not found in the first 5 years of data
            # (i.e. trying to exclude generic legal phrasing that would, in some sense, be "out of copyright")
            modern_antecedents = antecedents[~antecedents.index.isin(old_phrases.index)]

            phrase_count = phrases.index.nunique()
            antecedent_count = antecedents.index.nunique()
            modern_antecedent_count = modern_antecedents.index.nunique()

            results.append({
                "filename": filename,
                "phrase_length": length,
                "n_phrases": phrase_count,
                "n_antecedents": antecedent_count,
                "n_modern_antecedents": modern_antecedent_count
            })

    results = pd.DataFrame(results)
    results["antecedent_fraction"] = results.n_antecedents / results.n_phrases
    results["modern_antecedent_fraction"] = results.n_modern_antecedents / results.n_phrases

    results.to_csv("data/antecedent_counts.csv", index=False)


if __name__ == '__main__':
    main()
