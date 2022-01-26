from tqdm import tqdm
import re
from collections import defaultdict
import os
from bs4 import BeautifulSoup
import pickle
import difflib
import scipy

# import spacy
# from benepar.spacy_plugin import BeneparComponent
from nltk.translate.bleu_score import sentence_bleu


def read_incr_dump(d):
    edit_titles = {}
    for folder in tqdm(os.listdir(d)):
        for file in os.listdir(os.path.join(d, folder)):
            f = os.path.join(d, folder, file)
            data = open(f, "r").read()

            # parse
            soup = BeautifulSoup(data, "html.parser")

            for doc in soup.find_all("doc"):
                id = doc.get("id")
                title = doc.get("title")
                url = doc.get("url")
                text = doc.get_text()
                assert title not in edit_titles
                edit_titles[title] = {
                    "id": id,
                    "title": title,
                    "url": url,
                    "text": text,
                }
    return edit_titles


def read_dump(d, edits):
    documents = []
    for file in tqdm(os.listdir(d)):
        data = pickle.loads(open(os.path.join(d, file), "rb").read())
        for doc in data:
            title = doc["title"]
            if title in edits and edits[title]["id"] == doc["id"]:
                documents.append(doc)
    return documents


def get_spans(sent_diffs):
    spans = []
    start = None
    for i in range(len(sent_diffs)):
        r = sent_diffs[i]
        if r[:2] == "+ " or r[:2] == "- ":
            if start is None:
                start = i

        if start is not None:
            if (r[:2] != "+ " and r[:2] != "- ") or i == len(sent_diffs) - 1:
                spans.append((start, i))
                start = None
    return spans


def get_diffs(sent_a, sent_b):
    d = difflib.Differ()
    result = list(d.compare(sent_a, sent_b))
    sent_a_diffs = [r for r in result if "+ " not in r]
    sent_b_diffs = [r for r in result if "- " not in r]
    return sent_a_diffs, sent_b_diffs


def split_sentences(text):
    rtext = text.replace(".\n", ".\n<SPLITHERE>")
    rtext = rtext.replace(". ", ". <SPLITHERE>")
    sentences = rtext.split("<SPLITHERE>")
    assert len(text) == sum(
        [len(s) for s in sentences]
    ), f"Invalid length {len(text)}, {sum([len(s) for s in sentences])}"

    index_to_sent = [-1] * len(text)
    sent_to_index = [-1] * len(sentences)
    index = 0
    for i in range(len(sentences)):
        sent_to_index[i] = index
        for j in range(len(sentences[i])):
            # map character index to sentence index
            index_to_sent[index] = i
            index += 1

    assert index == len(text), f"changed text len {index}, {len(text)}"
    for i in index_to_sent:
        assert i >= 0

    return sentences, index_to_sent, sent_to_index


def get_diff_spans(doc, doc_diffs_raw, nlp):

    # get spans
    doc_spans = get_spans(doc_diffs_raw)

    # get sentences from
    sentences, index_to_sent, sent_to_index = split_sentences(doc)

    sent_diffs = []

    for span in doc_spans:

        # print('DIFF:', doc[span[0]:span[1]])

        # get sentence indices
        start_i = index_to_sent[span[0]]
        end_i = index_to_sent[span[1]]
        sent_ind = range(start_i, end_i + 1, 1) if end_i > start_i else [start_i]

        # offset char indices
        offset = sent_to_index[start_i]
        diff_span = (span[0] - offset, span[1] - offset)

        # parse sentence
        sent_comb = " ".join([sentences[i] for i in sent_ind])
        if len(sent_comb) > 10000:
            print("sentence too long", len(sent_comb))
            continue

        try:

            parsed = nlp(sent_comb)
            csent_all = list(parsed.sents)

            # generate word spans
            word_spans = []
            words = []
            for csent in csent_all:
                for constituent in csent._.constituents:

                    word_spans.append((constituent.start, constituent.end))
                    if DEBUG:
                        print("C:", const_offset, constituent)
                    if constituent.start + 1 == constituent.end:
                        words.append((constituent.start, str(constituent)))
        except Exception as e:
            print(e)
            continue

        # map word indices to character indices
        index = 0
        word_to_char_index = {len(words): len(sent_comb)}
        # TODO: make sure to sort words
        for word_index, word in words:
            csize = len(word)
            while str(sent_comb[index : index + csize]) != str(word):
                index += 1
            word_to_char_index[word_index] = index

        # convert word spans to char spans
        char_spans = [
            (word_to_char_index[s[0]], word_to_char_index[s[1]]) for s in word_spans
        ]

        # find minimal length span
        min_i = None
        min_length = len(sent_comb)
        for i in range(len(char_spans)):
            span_len = char_spans[i][1] - char_spans[i][0]
            if (
                char_spans[i][0] <= diff_span[0]
                and char_spans[i][1] >= diff_span[1]
                and span_len <= min_length
            ):
                min_i = i
                min_length = span_len

        if min_i is None:
            span_text = sent_comb
            # print("COULD NOT DETERMINE SPAN")
            # print(doc[span[0]:span[1]])
        else:
            span_text = sent_comb[char_spans[min_i][0] : char_spans[min_i][1]]

        # generate span text
        diff_text = doc[span[0] : span[1]]
        sent_diffs.append((diff_text, span_text))

        if DEBUG:
            print("WORD SPANS", word_spans)
            print("CHAR SPANS", char_spans)
            print("DIFF SPAN", diff_span)
            print("DIFF", diff_text)
            print(char_spans[min_i], span_text)
            print()

    return sent_diffs


def get_diffs(sent_a, sent_b):
    d = difflib.Differ()
    result = list(d.compare(sent_a, sent_b))
    sent_a_diffs = [r for r in result if "+ " not in r]
    sent_b_diffs = [r for r in result if "- " not in r]
    return sent_a_diffs, sent_b_diffs


def get_diffs(sent_a, sent_b):
    d = difflib.Differ()
    result = list(d.compare(sent_a, sent_b))
    sent_a_diffs = [r for r in result if "+ " not in r]
    sent_b_diffs = [r for r in result if "- " not in r]
    return sent_a_diffs, sent_b_diffs


def get_diffs(sent_a, sent_b):
    d = difflib.Differ()
    result = list(d.compare(sent_a, sent_b))
    sent_a_diffs = [r for r in result if "+ " not in r]
    sent_b_diffs = [r for r in result if "- " not in r]
    return sent_a_diffs, sent_b_diffs


def get_diffs(sent_a, sent_b):
    d = difflib.Differ()
    result = list(d.compare(sent_a, sent_b))
    sent_a_diffs = [r for r in result if "+ " not in r]
    sent_b_diffs = [r for r in result if "- " not in r]
    return sent_a_diffs, sent_b_diffs


def get_sentence_diff(sent_a, sent_b, nlp=None):

    # get spans from differ
    sent_a_diffs_raw, sent_b_diffs_raw = get_diffs(sent_a, sent_b)

    if nlp is None:
        diffs_a = sent_a_diffs_raw
        diffs_b = sent_b_diffs_raw
    else:
        diffs_a = list(set(get_diff_spans(sent_a, sent_a_diffs_raw, nlp)))
        diffs_b = list(set(get_diff_spans(sent_b, sent_b_diffs_raw, nlp)))

    span_diffs_a = [d[1] for d in diffs_a]
    span_diffs_b = [d[1] for d in diffs_b]
    raw_diffs_a = [d[0] for d in diffs_a]
    raw_diffs_b = [d[0] for d in diffs_b]

    return {
        "sent_a": sent_a,
        "sent_b": sent_b,
        "sent_a_diffs": span_diffs_a,
        "sent_b_diffs": span_diffs_b,
        "sent_a_raw_diffs": raw_diffs_a,
        "sent_b_raw_diffs": raw_diffs_b,
    }


def generate_diffs(documents, edits):
    all_diffs = []
    i = 0
    for article in tqdm(documents):
        title = article["title"]
        edit = edits[title]

        sent_a = article["text"]
        sent_b = edit["text"]

        doc_id = article["id"]
        assert (
            article["id"] == edit["id"]
        ), f"Mismatch article - title: {title}, {edit['title']}, id: {article['id']}, {edit['id']}"

        DEBUG = False
        # run: python -m spacy download en
        # nlp = spacy.load("en")
        ## nlp = spacy.load("en_core_web_sm")
        # nlp.add_pipe(BeneparComponent("benepar_en3"))

        diff = get_sentence_diff(sent_a, sent_b, nlp)
        diff["title"] = (title,)
        diff["doc_id"] = doc_id
        all_diffs.append(diff)

        if len(all_diffs) > 1000:
            print("Writing", i)
            pickle.dump(all_diffs, open(f"output/diffs_{i}.pkl", "wb"))
            all_diffs = []

            i += 1


def check_alphanumeric(s):
    return re.match("(?s).*[a-zA-Z0-9]+(?s).*$", s) is not None


def generate_sentence_level_diffs(documents, edits):
    all_diffs = []
    count = 0

    # for article in tqdm(documents):
    for article in documents:
        title = article["title"]
        edit = edits[title]
        sent_a = article["text"]
        sent_b = edit["text"]

        splits_a, index_to_sent_a, sent_to_index_a = split_sentences(sent_a)
        splits_b, index_to_sent_b, sent_to_index_b = split_sentences(sent_b)

        d = difflib.Differ(
            linejunk=lambda x: x in " \n", charjunk=lambda x: x in " \n \t"
        )
        diff = list(d.compare(splits_a, splits_b))

        index = 0
        last_match = 0
        options = defaultdict(list)
        for i in range(len(diff)):

            code = diff[i][:2]
            if code == "? ":
                continue
            elif code == "+ ":
                options[last_match].append(diff[i])
            elif code == "- ":
                options[last_match].append(diff[i])
            else:
                options[index] = diff[i][2:]
                last_match = index + 1
            index += 1

        diff_data = []

        has_diff = False
        for key, value in options.items():
            # print(key, value)
            if not isinstance(value, list):
                diff_data.append(
                    {
                        "sent_a": value,
                        "sent_b": value,
                        "sent_a_diffs": [],
                        "sent_b_diffs": [],
                        "diff_type": None,
                    }
                )
                continue

            diff_a = [d[2:] for d in value if "- " in d]
            diff_b = [d[2:] for d in value if "+ " in d]

            has_diff = True

            # nlp = spacy.load("en")
            # nlp.add_pipe(BeneparComponent("benepar_en3"))

            for da in diff_a:
                match = False
                for i in range(len(diff_b) - 1, -1, -1):
                    db = diff_b[i]
                    score = sentence_bleu([da.split()], db.split())
                    # print(score)
                    if score > 0.1:
                        # local_a, local_b = get_diffs(da, db)
                        diff = get_sentence_diff(da, db, nlp=None)
                        diff["diff_type"] = "EDIT"
                        diff["score"] = score

                        # filter alphanumeric
                        # orig_a = list( diff["sent_a_diffs"])
                        # orig_b = list( diff["sent_b_diffs"])
                        diff["sent_a_diffs"] = [
                            d for d in diff["sent_a_diffs"] if check_alphanumeric(d)
                        ]
                        diff["sent_b_diffs"] = [
                            d for d in diff["sent_b_diffs"] if check_alphanumeric(d)
                        ]
                        if (
                            len(diff["sent_a_diffs"]) == 0
                            and len(diff["sent_b_diffs"]) == 0
                            and da == db
                        ):
                            diff["diff_type"] = None
                            # print("CONVERT EDIT TO NONE")
                            # print(diff["sent_a_raw_diffs"])
                            # print(diff["sent_b_raw_diffs"])
                            # print(orig_a)
                            # print(orig_b)

                        # pprint(diff)
                        diff_data.append(diff)
                        del diff_b[i]  # avoid double counting
                        match = True
                        break

                if not match:
                    diff_data.append(
                        {
                            "sent_a": da,
                            "sent_b": "",
                            "sent_a_diffs": [da],
                            "sent_b_diffs": [],
                            "diff_type": "DELETE",
                        }
                    )

            for db in diff_b:
                diff_data.append(
                    {
                        "sent_a": "",
                        "sent_b": db,
                        "sent_a_diffs": [],
                        "sent_b_diffs": [db],
                        "diff_type": "INSERT",
                    }
                )

        # pprint([d for d in diff_data if d['diff_type'] is not None])
        all_diffs.append(diff_data)
        count += 1

        # if len(all_diffs) > 1000:
        #    print("Writing", count)
        #    pickle.dump(all_diffs, open(f"output/sent_diffs_{count}.pkl", "wb"))
        #    all_diffs = []

        return all_diffs, has_diff


def main():
    edits = read_incr_dump("/home/ubuntu/incr-enwiki-20190206/text/")
    print("finished reading edits", len(edits.keys()))
    documents = read_dump("/home/ubuntu/enwiki-20190201/tmp/parsed", edits)
    print("finished reading docs", len(documents))

    print("generating diffs...")
    # generate_diffs(documents, edits)
    generate_sentence_level_diffs(documents, edits)


if __name__ == "__main__":
    main()
