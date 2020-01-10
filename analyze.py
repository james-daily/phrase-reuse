import argparse
import re

import numpy as np
import pandas as pd
from tqdm import tqdm


def mentions_mobile_phones(text):
    if re.search(r"(mobile|cellular|cell|smart)[ \-]?(tele)?phone", text, flags=re.I):
        return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", "-d", action="store_true")
    parser.add_argument("--chunks", type=int)
    parser.add_argument("--chunk_num", type=int)
    args = parser.parse_args()

    print("loading opinion data")
    opinions = pd.read_csv("data/opinions.csv")

    # calculate majority opinion or not
    opinions["majority_opinion"] = opinions.opinion_type == "MO"

    # calculate whether the opinion mentions mobile phones
    opinions["mobile_phone"] = opinions.text.apply(mentions_mobile_phones)

    # drop no longer needed text column
    del opinions["text"]

    # calculate the most frequent author, JPS
    most_frequent_authors = opinions.groupby("author").filename.count().sort_values(ascending=False)
    most_frequent_author = most_frequent_authors.reset_index().author.iloc[0]
    assert most_frequent_author == "JPS"

    if args.debug:
        print("subsetting to 5% sample for debugging")
        opinion_sample = np.random.choice(opinions.filename, int(len(opinions) * 0.05), False)
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

    # get opinions in the most recent year
    opinions_to_analyze = df[df.year == opinions.year.max()].filename.unique()
    if args.chunks is not None and args.chunk_num is not None:
        print(f"processing chunk {args.chunk_num} of {args.chunks}")
        opinions_to_analyze = np.array_split(opinions_to_analyze, args.chunks)[args.chunk_num]

    # for each opinion to analyze
    for filename in tqdm(opinions_to_analyze, desc="finding antecedents"):
        # get the opinion author
        author = opinions[opinions.filename == filename].author.iloc[0]

        # get whether this is a majority opinion or not
        majority_opinion = opinions[opinions.filename == filename].majority_opinion.iloc[0]

        # get whether this is an opinion that mentions mobile phones or not
        mobile_phone = opinions[opinions.filename == filename].mobile_phone.iloc[0]

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

            # count how many occur in opinions written by a single other author
            most_frequent_author_antecedents = df[(df.index.isin(phrases.index)
                                                   & (df.author == most_frequent_author))
                                                  & (df.year < opinions.year.max())
                                                  & (df["length"] == length)]

            # count how many occur in opinions not of this type
            other_opinion_type_antecedents = df[(df.index.isin(phrases.index)
                                                 & (df.majority_opinion != majority_opinion)
                                                 & (df.author != author))
                                                & (df.year < opinions.year.max())
                                                & (df["length"] == length)]

            # count how many occur in opinions that do/don't mention cell phones
            mobile_phone_antecedents = df[(df.index.isin(phrases.index)
                                           & (df.majority_opinion != mobile_phone)
                                           & (df.author != author))
                                          & (df.year < opinions.year.max())
                                          & (df["length"] == length)]

            results.append({
                "filename": filename,
                "author": author,
                "majority_opinion": majority_opinion,
                "mentions_mobile_phones": mobile_phone,
                "phrase_length": length,
                "n_phrases": phrases.index.nunique(),
                "n_antecedents": antecedents.index.nunique(),
                "n_modern_antecedents": modern_antecedents.index.nunique(),
                "n_most_frequent_author_antecedents": most_frequent_author_antecedents.index.nunique(),
                "n_other_opinion_type_antecedents": other_opinion_type_antecedents.index.nunique(),
                "n_mobile_phone_antecedents": mobile_phone_antecedents.index.nunique()
            })

    results = pd.DataFrame(results)
    results["antecedent_fraction"] = results.n_antecedents / results.n_phrases
    results["modern_antecedent_fraction"] = results.n_modern_antecedents / results.n_phrases
    results["most_frequent_author_antecedent_fraction"] = results.n_most_frequent_author_antecedents / results.n_phrases
    results["other_opinion_type_antedecedent_fraction"] = results.n_other_opinion_type_antecedents / results.n_phrases
    results["mobile_phone_antecedent_fraction"] = results.n_mobile_phone_antecedents / results.n_phrases

    results.to_csv(f"data/antecedent_counts_{args.chunk_num}_of_{args.chunks}.csv", index=False)


if __name__ == '__main__':
    main()
