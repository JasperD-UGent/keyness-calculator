from .counters import multiple_250
from .process_JSONs import dump_json
import copy
import csv
import os
import random
import sys
from typing import Dict, Tuple, Union


def d_freq(
        corpus_name: str,
        corpus_input: Union[str, Tuple],
        mapping_custom_to_ud: Dict,
        mapping_ud_to_custom: Dict,
        desired_pos: Tuple,
        lem_or_tok: str,
        maintain_subcorpora: bool,
        div_n_docs_by: int
) -> Tuple[Dict, Dict, Dict]:
    """Construct frequency dictionaries (per item).
    :param corpus_name: name of the corpus.
    :param corpus_input: provided input data for the corpus.
    :param mapping_custom_to_ud: if you work with custom POS tags, dictionary which maps custom tags to UD counterparts.
    :param mapping_ud_to_custom: if you work with custom POS tags, dictionary which maps UD tags to custom counterparts.
    :param desired_pos: tuple of UD tags which should be taken into account in the keyness calculations.
    :param lem_or_tok: defines whether to calculate frequencies on token or lemma level.
    :param maintain_subcorpora: when working with adjusted frequencies, boolean value which defines whether dispersion
        is based on existing subcorpora, or whether all documents are merged and randomly split into new subcorpora.
    :param div_n_docs_by: when working with adjusted frequencies, number by which the total number of documents is
        divided to arrive at the number of new randomly generated subcorpora.
    :return: a tuple containing three dictionaries: a frequency dictionary of the entire corpus, a frequency dictionary
        per corpus part, and a dictionary in which the corpus part names are linked to their IDs.
    """

    # check input type
    if type(corpus_input) == str:
        input_type = "3-column_delimited"
    elif type(corpus_input) == tuple and len(corpus_input) == 2:
        input_type = "tuples"
    else:
        raise ValueError("Input in invalid format.")

    l_pos = []

    for pos in desired_pos:

        for tag in mapping_ud_to_custom[pos]:
            l_pos.append(tag)

    if input_type == "3-column_delimited":
        d_tuples_corpus = {}
        l_docs_all = []

        for subcorpus in os.listdir(os.path.join(corpus_input)):
            print(f"Number of files in {subcorpus}: {len(os.listdir(os.path.join(corpus_input, subcorpus)))}.")
            l_tuples_subcorpus = []
            id_doc = 0
            counter = 0

            for doc in os.listdir(os.path.join(corpus_input, subcorpus)):
                id_doc += 1
                docname = f"{subcorpus}_{id_doc}"
                l_docs_all.append(docname)

                if doc.endswith(".csv"):
                    delim = ","
                elif doc.endswith(".tsv"):
                    delim = "\t"
                else:
                    raise ValueError("Delimiter is not recognised.")

                with open(os.path.join(corpus_input, subcorpus, doc), mode="r") as f_delimited:
                    reader = csv.reader(f_delimited, delimiter=delim)

                    for row in reader:
                        tok = row[0]
                        pos = row[1]
                        lem = row[2]

                        if lem_or_tok == "lemma":

                            if pos in l_pos:

                                if maintain_subcorpora:
                                    l_tuples_subcorpus.append((lem, mapping_custom_to_ud[pos]))
                                else:
                                    l_tuples_subcorpus.append((lem, mapping_custom_to_ud[pos], docname))

                        elif lem_or_tok == "token":

                            if pos in l_pos:

                                if maintain_subcorpora:
                                    l_tuples_subcorpus.append((tok, mapping_custom_to_ud[pos]))
                                else:
                                    l_tuples_subcorpus.append((tok, mapping_custom_to_ud[pos], docname))

                        else:
                            raise ValueError("`lemma_or_token` is not correctly defined.")

                f_delimited.close()

                counter += 1
                multiple_250(counter)

            d_tuples_corpus[subcorpus] = l_tuples_subcorpus

    elif input_type == "tuples":
        d_tuples_corpus = {}
        l_docs_all = []

        for subcorpus in corpus_input[1]:
            print(f"Number of files in {subcorpus}: {len(corpus_input[1][subcorpus])}.")
            l_tuples_subcorpus = []
            id_doc = 0
            counter = 0

            for doc in corpus_input[1][subcorpus]:
                id_doc += 1
                docname = f"{subcorpus}_{id_doc}"
                l_docs_all.append(docname)

                for tup in doc:
                    tok = tup[0]
                    pos = tup[1]
                    lem = tup[2]

                    if lem_or_tok == "lemma":

                        if pos in l_pos:

                            if maintain_subcorpora:
                                l_tuples_subcorpus.append((lem, mapping_custom_to_ud[pos]))
                            else:
                                l_tuples_subcorpus.append((lem, mapping_custom_to_ud[pos], docname))

                    elif lem_or_tok == "token":

                        if pos in l_pos:

                            if maintain_subcorpora:
                                l_tuples_subcorpus.append((tok, mapping_custom_to_ud[pos]))
                            else:
                                l_tuples_subcorpus.append((tok, mapping_custom_to_ud[pos], docname))

                    else:
                        raise ValueError("`lemma_or_token` is not correctly defined.")

                counter += 1
                multiple_250(counter)

            d_tuples_corpus[subcorpus] = l_tuples_subcorpus

    else:
        raise ValueError("`input_type` is not correctly defined.")

    d_freq_corpus = {"subcorpora": {}}
    d_freq_corpus_json = {"subcorpora": {}}
    d_freq_all = {}

    for subcorpus in d_tuples_corpus:
        d_freq_subcorpus = {}

        for tup in d_tuples_corpus[subcorpus]:
            tup_d_freq = (tup[0], tup[1])

            if tup_d_freq not in d_freq_subcorpus:
                d_freq_subcorpus[tup_d_freq] = 1
            else:
                d_freq_subcorpus[tup_d_freq] += 1

            if tup_d_freq not in d_freq_all:
                d_freq_all[tup_d_freq] = 1
            else:
                d_freq_all[tup_d_freq] += 1

        d_freq_subcorpus_json = {}

        for tup in d_freq_subcorpus:
            d_freq_subcorpus_json[str(tup)] = d_freq_subcorpus[tup]

        d_freq_corpus["subcorpora"][subcorpus] = d_freq_subcorpus
        d_freq_corpus_json["subcorpora"][subcorpus] = d_freq_subcorpus_json

    d_freq_all_json = {}

    for tup in d_freq_all:
        d_freq_all_json[str(tup)] = d_freq_all[tup]

    d_freq_corpus["corpus"] = d_freq_all
    d_freq_corpus_json["corpus"] = d_freq_all_json

    fn_d_freq = f"{corpus_name}_d_freq.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_freq, d_freq_corpus_json)

    # d_freq corpus parts
    l_docs = list(dict.fromkeys(l_docs_all))
    d_freq_corpus_parts = {}
    d_cps = {}

    if maintain_subcorpora:

        for subcorpus in d_freq_corpus["subcorpora"]:

            for tup in d_freq_corpus["subcorpora"][subcorpus]:
                item = tup[0]
                pos = tup[1]
                new_tup = (item, pos, subcorpus)
                d_freq_corpus_parts[new_tup] = d_freq_corpus["subcorpora"][subcorpus][tup]

    else:
        l_docs_shuffled = copy.deepcopy(l_docs)
        random.shuffle(l_docs_shuffled)
        n_docs = len(l_docs_shuffled)
        n_cps = int(round(n_docs / div_n_docs_by))

        if n_cps < n_docs:
            assert 0 < n_cps <= n_docs
            quot, remain = divmod(n_docs, n_cps)
            size_large = quot + 1
            l_docs_div = (
                    [l_docs_shuffled[part:part + size_large] for part in range(0, remain * size_large, size_large)]
                    + [l_docs_shuffled[part:part + quot] for part in range(remain * size_large, n_docs, quot)]
            )

            id_cp = 1

            for part in l_docs_div:
                cp_name = f"corpus_part_{id_cp}"
                d_cps[cp_name] = part
                id_cp += 1

        else:
            id_cp = 1

            for part in l_docs:
                cp_name = f"corpus_part_{id_cp}"
                d_cps[cp_name] = part
                id_cp += 1

        d_cp_map = {}

        for cp in d_cps:

            for doc in d_cps[cp]:
                d_cp_map[doc] = cp

        for subcorpus in d_tuples_corpus:

            for tup in d_tuples_corpus[subcorpus]:
                item = tup[0]
                pos = tup[1]
                docname = tup[2]
                cp = d_cp_map[docname]
                new_tup = (item, pos, cp)

                if new_tup not in d_freq_corpus_parts:
                    d_freq_corpus_parts[new_tup] = 1
                else:
                    d_freq_corpus_parts[new_tup] += 1

    d_freq_corpus_parts_json = {}

    for tup in d_freq_corpus_parts:
        d_freq_corpus_parts_json[str(tup)] = d_freq_corpus_parts[tup]

    fn_d_freq_cps = f"{corpus_name}_d_freq_corpus_parts.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_freq_cps, d_freq_corpus_parts_json)

    return d_freq_corpus, d_freq_corpus_parts, d_cps


def sum_words_desired_pos(
        corpus_name: str, d_freq_corpus: Dict, desired_pos: Tuple, d_freq_cps: Dict, d_cps: Dict,
        maintain_subcorpora: bool
) -> Dict:
    """Construct frequency dictionary (totals).
    :param corpus_name: name of the corpus.
    :param d_freq_corpus: frequency dictionary of the entire corpus.
    :param desired_pos: tuple of UD tags which should be taken into account in the keyness calculations.
    :param d_freq_cps: frequency dictionary per corpus part.
    :param d_cps: dictionary in which the corpus part names are linked to their IDs.
    :param maintain_subcorpora: when working with adjusted frequencies, boolean value which defines whether dispersion
        is based on existing subcorpora, or whether all documents are merged and randomly split into new subcorpora.
    :return: a dictionary containing the sum of the words per corpus part.
    """

    # sum corpus
    d_sum_corpus = {"corpus": {"all": {"total": 0, "unique": 0}}, "subcorpora": {}}

    for subcorpus in d_freq_corpus["subcorpora"]:
        d_sum_corpus["subcorpora"][subcorpus] = {"all": {"total": 0, "unique": 0}}

    for pos in desired_pos:
        d_sum_corpus["corpus"][pos] = {"total": 0, "unique": 0}

        for subcorpus in d_freq_corpus["subcorpora"]:
            d_sum_corpus["subcorpora"][subcorpus][pos] = {"total": 0, "unique": 0}

    for tup in d_freq_corpus["corpus"]:
        pos = tup[1]
        d_sum_corpus["corpus"]["all"]["total"] += d_freq_corpus["corpus"][tup]
        d_sum_corpus["corpus"]["all"]["unique"] += 1
        d_sum_corpus["corpus"][pos]["total"] += d_freq_corpus["corpus"][tup]
        d_sum_corpus["corpus"][pos]["unique"] += 1

    for subcorpus in d_freq_corpus["subcorpora"]:

        for tup in d_freq_corpus["subcorpora"][subcorpus]:
            pos = tup[1]
            d_sum_corpus["subcorpora"][subcorpus]["all"]["total"] += d_freq_corpus["subcorpora"][subcorpus][tup]
            d_sum_corpus["subcorpora"][subcorpus]["all"]["unique"] += 1
            d_sum_corpus["subcorpora"][subcorpus][pos]["total"] += d_freq_corpus["subcorpora"][subcorpus][tup]
            d_sum_corpus["subcorpora"][subcorpus][pos]["unique"] += 1

    fn_d_sum_corpus = f"{corpus_name}_sum_words_desired_pos.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_sum_corpus, d_sum_corpus)

    # sum corpus parts
    d_sum_cps = {}

    if maintain_subcorpora:

        for subcorpus in d_sum_corpus["subcorpora"]:
            d_sum_cps[subcorpus] = {}

        for subcorpus in d_sum_corpus["subcorpora"]:
            d_sum_cps[subcorpus]["total_all"] = d_sum_corpus["subcorpora"][subcorpus]["all"]["total"]
            d_sum_cps[subcorpus]["normalised_total_all"] = \
                d_sum_corpus["subcorpora"][subcorpus]["all"]["total"] / d_sum_corpus["corpus"]["all"]["total"]

            for pos in desired_pos:
                entry = f"total_{pos}"
                d_sum_cps[subcorpus][entry] = d_sum_corpus["subcorpora"][subcorpus][pos]["total"]

    else:

        for part in d_cps:
            d_sum_cps[part] = {"total_all": 0}

            for pos in desired_pos:
                d_sum_cps[part][f"total_{pos}"] = 0

        for tup in d_freq_cps:
            pos = tup[1]
            part = tup[2]
            d_sum_cps[part]["total_all"] += d_freq_cps[tup]
            d_sum_cps[part][f"total_{pos}"] += d_freq_cps[tup]

        for part in d_cps:
            d_sum_cps[part]["normalised_total_all"] = \
                d_sum_cps[part]["total_all"] / d_sum_corpus["corpus"]["all"]["total"]

    fn_d_sum_cps = f"{corpus_name}_sum_words_desired_pos_corpus_parts.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_sum_cps, d_sum_cps)

    return d_sum_cps


def corpora_to_d_freq(
        corpus_name: str,
        input_corpus: Union[str, Tuple],
        mapping_custom_to_ud: Dict,
        mapping_ud_to_custom: Dict,
        desired_pos: Tuple,
        lem_or_tok: str,
        maintain_subcorpora: bool,
        div_n_docs_by: int
) -> Tuple[Dict, Dict, Dict]:
    """STEP_1: convert corpora into frequency dictionaries (data stored per corpus in "prep" folder).
    :param corpus_name: name of the corpus
    :param input_corpus: provided input data for the corpus.
    :param mapping_custom_to_ud: if you work with custom POS tags, dictionary which maps custom tags to UD counterparts.
    :param mapping_ud_to_custom: if you work with custom POS tags, dictionary which maps UD tags to custom counterparts.
    :param desired_pos: tuple of UD tags which should be taken into account in the keyness calculations.
    :param lem_or_tok: defines whether to calculate frequencies on token or lemma level.
    :param maintain_subcorpora: when working with adjusted frequencies, boolean value which defines whether dispersion
        is based on existing subcorpora, or whether all documents are merged and randomly split into new subcorpora.
    :param div_n_docs_by: when working with adjusted frequencies, number by which the total number of documents is
        divided to arrive at the number of new randomly generated subcorpora.
    :return: a tuple containing three dictionaries: a frequency dictionary of the entire corpus, a frequency dictionary
        per corpus part, and a dictionary containing the sum of the words per corpus part.
    """
    d_freq_corpus, d_freq_cps, d_cps = d_freq(
        corpus_name, input_corpus, mapping_custom_to_ud, mapping_ud_to_custom, desired_pos, lem_or_tok,
        maintain_subcorpora, div_n_docs_by
    )
    d_sum_cps = sum_words_desired_pos(
        corpus_name, d_freq_corpus, desired_pos, d_freq_cps, d_cps, maintain_subcorpora
    )

    return d_freq_corpus, d_freq_cps, d_sum_cps
