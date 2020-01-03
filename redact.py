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
        # this is roughly a 10% sample
        nrows = 12 * 1000 * 1000
    else:
        nrows = None

    phrases = pd.read_csv("data/phrases.csv", usecols=["filename", "length", "phrase"], nrows=nrows)

    # join phrase and opinion data
    print("merging phrase and opinion data")
    phrases = phrases.merge(opinions)

    # subset to remove anything newer than the input file and anything by the same author
    print("removing phrases from opinions on or after", year, "or written by", author)
    phrases = phrases[(phrases.year < year) & (phrases.author != author)]

    print("dropping phrases duplicated within the same opinion")
    df = df.drop_duplicates(subset=["filename", "phrase"])

    print("indexing phrases")
    phrases = phrases.set_index("phrase").sort_index()

    # load spacy
    print("loading spacy")

    # for each phrase length
    for length in ["1", "2", "3", "sentence"]:

        # HACK: we add a space to the beginning of the text so that the first word can be matched
        # this is necessary because of the redaction regex: fr"(?<=[^A-Z])({phrase})(?=[^A-Z])"
        text = " " + original_text

        phrase_subset = phrases[phrases.length == length]

        # for each unique phrase in the document of that length
        for phrase in df[df.length == length].phrase.unique():

            # skip pure underlines, numbers, and spaces
            if re.search("^_+$", phrase) or phrase.isnumeric() or phrase.isspace():
                continue

            if not phrase_subset[phrase_subset.index == phrase].empty:
                text = redact(text, phrase)

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
                    font-family: "New Century Schoolbook", Times, serif;
                }}
                span.redact {{
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
    # turn double newlines into paragraphs
    cleaned = re.sub("\n\n", "</p><p>", text)

    # wrap the whole text in a paragraph
    cleaned = f"<p>{cleaned}</p>"

    return cleaned


def redact(text, phrase):
    if phrase in '<span class="redacted':
        print("PROBLEMATIC PHRASE DETECTED:", phrase, "occurs in the html")

    # try to match the phrase only if it's surrounded by characters that aren't letters
    redacted_text = re.sub(fr"(?<=[^A-Z])({phrase})(?=[^A-Z])", r'<span class="redact">\g<1></span>', text, flags=re.I)

    return redacted_text


if __name__ == '__main__':
    main()
