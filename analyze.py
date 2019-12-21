import pandas as pd


def main():
    opinions = pd.read_csv("data/opinions.csv", usecols=["filename", "year", "lexis_cite", "opinion_type", "author"])
    phrases = pd.read_pickle("data/phrases.pkl")

    df = phrases.merge(opinions)

    df = df.sort_values(["lexis_cite", "author"])

    phrase_counts = df.groupby(["phrase", "length"]).filename.count().reset_index().rename(columns={"filename": "count"})

    pass


if __name__ == '__main__':
    main()
