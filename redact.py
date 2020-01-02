import argparse
import re

import pandas as pd

from process import process


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("input_file")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    # load the input json file into a minimal dataframe
    print("loading", args.input_file)
    df = pd.read_json(args.input_file)
    original_text = df.text.iloc[0]
    lexis_cite = df.lexis_cite.iloc[0]
    author = df.author.iloc[0]
    year = df.year.iloc[0]

    df = process(df)

    print("loading opinion data")
    opinions = pd.read_csv("data/opinions.csv", usecols=["filename", "year", "lexis_cite", "opinion_type", "author"], )

    # load the phrase data
    print("loading phrase data")
    if args.debug:
        nrows = 200000
    else:
        nrows = None

    phrases = pd.read_csv("data/phrases.csv", usecols=["filename", "length", "phrase"], nrows=nrows)

    # join phrase and opinion data
    print("merging phrase and opinion data")
    phrases = phrases.merge(opinions)

    # subset to remove anything newer than the input file and anything by the same author
    print("removing phrases from opinions on or after", year, "or written by", author)
    phrases = phrases[(phrases.year < year) & (phrases.author != author)]

    print("indexing phrases")
    phrases = phrases.set_index("phrase").sort_index()

    # load spacy
    print("loading spacy")

    # for each phrase length
    for length in ["1", "2", "3", "sentence"]:

        text = original_text

        phrase_subset = phrases[phrases.length == length]

        # for each phrase in the document of that length
        for phrase in df[df.length == length].phrase:
            if not phrase_subset[phrase_subset.index == phrase].empty:
                print("found:\t\t", phrase)

                text = redact(text, phrase)

            else:
                print("NOT found:\t", phrase)

        text = cleanup(text)

        html = make_html(text)

        html_filename = f"{lexis_cite.replace(' ', '_')}_{length}.html"

        print("writing out", html_filename)
        with open(f"data/redactions/{html_filename}", "w") as f:
            f.write(html)


def make_html(text):
    html = f"""
        <html lang="en">
            <style>
                p {{
                    text-indent: 5em;
                }}
                span.redacted {{
                    background-color: #000000;
                }}
            </style>
        
            <body>
                {text}
            </body>
        </html>
        """

    return html


def cleanup(text):
    cleaned = re.sub("\n\n", "</p><p>", text)

    cleaned = f"<p>{cleaned}</p>"

    return cleaned


def redact(text, phrase):
    print(f"redacting '{phrase}'")

    redacted_text = re.sub(phrase, r'<span class="redacted">\g<0></span>', text, flags=re.I)

    return redacted_text


if __name__ == '__main__':
    main()
