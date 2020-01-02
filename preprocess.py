import argparse
import os
import re

import pandas as pd
from tqdm import tqdm


def preprocess(input_dir):
    rows = []

    for root, directories, files in os.walk(input_dir):
        for filename in tqdm(files, desc=root):
            path = os.path.join(root, filename)

            m = re.search(r"(\d{4})-(\d+)-(\w+)_([A-Z]+)", filename)

            if not m:
                continue

            with open(path, "r", encoding="latin-1") as f:
                text = f.read()

                # remove repeated spaces
                text = re.sub(r" +", " ", text)

                # remove markup such as
                # "- - - - - - - - - - - - - - - Footnotes - - - - - - - - - - - - - - -"
                text = re.sub(r"(?:- ){3,}(-)?[\w ]+?(?:- ){3,}(-)?", "", text)

                # remove footnote numbers that run into the text
                text = re.sub(r"^\d(?=[A-Z])", "", text, flags=re.MULTILINE)

                # remove anything between square brackets, such as [HN1] or [**LEdHR1]
                text = re.sub(r"\[.*?\]", "", text)

                # remove all newlines
                text = re.sub(r"\n+", " ", text)

                # tidy up extraneous spaces
                text = re.sub(r" +", " ", text)

                rows.append({
                    "filename": filename,
                    "text": text,
                    "year": m.group(1),
                    "lexis_cite": f"{m.group(1)} U.S. LEXIS {m.group(2)}",
                    "opinion_type": m.group(3),
                    "author": m.group(4)
                })

    df = pd.DataFrame(rows)

    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir")
    args = parser.parse_args()

    df = preprocess(args.input_dir)

    print("writing out csv")
    df.to_csv("data/opinions.csv", index=False)


if __name__ == '__main__':
    main()
