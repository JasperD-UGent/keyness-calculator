from .counters import multiple_250
from .process_JSONs import dump_json, load_json
import copy
import csv
import numpy as np
import operator
import os
import random
from sklearn.cluster import AgglomerativeClustering
import xlsxwriter


def check_meta(corpus_name, desired_pos, lem_or_tok, maintain_subcorpora, div_n_docs_by):
    """Check if information in meta file corresponds to current query."""

    if os.path.exists(os.path.join("prep", corpus_name, "meta.json")):
        d_meta_corpus = load_json(os.path.join("prep", corpus_name, "meta.json"))

        if tuple(d_meta_corpus["desired_pos"]) == desired_pos \
                and d_meta_corpus["lemma_or_token"] == lem_or_tok \
                and d_meta_corpus["maintain_subcorpora"] == maintain_subcorpora \
                and d_meta_corpus["divide_number_docs_by"] == div_n_docs_by:
            return True
        else:
            return False

    else:
        return False


def define_additional_variables(corpus_input):
    """Define additional variables based on the criteria enterded in the `keynCalc.py` script."""

    if type(corpus_input) == str:
        corpus_name = os.path.basename(corpus_input)
    elif type(corpus_input) == tuple and len(corpus_input) == 2:
        corpus_name = corpus_input[0]
    else:
        raise ValueError("Input in invalid format.")

    return corpus_name


def d_freq(
        corpus_name, corpus_input, mapping_custom_to_ud, mapping_ud_to_custom, desired_pos, lem_or_tok,
        maintain_subcorpora, div_n_docs_by
):
    """Construct frequency dictionary (per item)."""

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
                docname = subcorpus + "_" + str(id_doc)
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
                docname = subcorpus + "_" + str(id_doc)
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

    fn_d_freq = corpus_name + "_d_freq.json"
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
                cp_name = "corpus_part_" + str(id_cp)
                d_cps[cp_name] = part
                id_cp += 1

        else:
            id_cp = 1

            for part in l_docs:
                cp_name = "corpus_part_" + str(id_cp)
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

    fn_d_freq_cps = corpus_name + "_d_freq_corpus_parts.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_freq_cps, d_freq_corpus_parts_json)

    return d_freq_corpus, d_freq_corpus_parts, d_cps


def sum_words_desired_pos(corpus_name, d_freq_corpus, desired_pos, d_freq_cps, d_cps, maintain_subcorpora):
    """Construct frequency dictionary (totals)."""

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

    fn_d_sum_corpus = corpus_name + "_sum_words_desired_pos.json"
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
                entry = "total_" + pos
                d_sum_cps[subcorpus][entry] = d_sum_corpus["subcorpora"][subcorpus][pos]["total"]

    else:

        for part in d_cps:
            d_sum_cps[part] = {"total_all": 0}

            for pos in desired_pos:
                d_sum_cps[part][pos] = 0

        for tup in d_freq_cps:
            pos = tup[1]
            part = tup[2]
            d_sum_cps[part]["total_all"] += d_freq_cps[tup]
            d_sum_cps[part][pos] += d_freq_cps[tup]

        for part in d_cps:
            d_sum_cps[part]["normalised_total_all"] = \
                d_sum_cps[part]["total_all"] / d_sum_corpus["corpus"]["all"]["total"]

    fn_d_sum_cps = corpus_name + "_sum_words_desired_pos_corpus_parts.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_sum_cps, d_sum_cps)

    return d_sum_cps


def dp(corpus_name, d_freq_corpus, d_freq_cps, d_sum_cps):
    """Calculate dispersion values (DPnorm)."""

    if len(d_sum_cps) == 1:
        d_dp_norm = {}

        for tup in d_freq_corpus["corpus"]:
            d_dp_norm[tup] = 0

    else:
        l_norm_sum_cps = [d_sum_cps[part]["normalised_total_all"] for part in d_sum_cps]
        smallest_cp = min(l_norm_sum_cps)

        d_dp = {}

        for part in d_sum_cps:

            for tup in d_freq_corpus["corpus"]:
                expected = d_sum_cps[part]["normalised_total_all"]
                entry = (tup[0], tup[1], part)

                if entry in d_freq_cps:
                    freq_part = d_freq_cps[entry]
                else:
                    freq_part = 0

                freq_corpus = d_freq_corpus["corpus"][tup]
                observed = freq_part / freq_corpus
                abs_diff = abs(expected - observed)

                if tup not in d_dp:
                    d_dp[tup] = abs_diff * 0.5
                else:
                    d_dp[tup] += abs_diff * 0.5

        d_dp_norm = {}

        for tup in d_dp:
            d_dp_norm[tup] = d_dp[tup] / (1 - smallest_cp)

    d_dp_norm_json = {}

    for tup in d_dp_norm:
        d_dp_norm_json[str(tup)] = d_dp_norm[tup]

    fn_d_dp = corpus_name + "_DP.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_dp, d_dp_norm_json)

    return d_dp_norm


def d_freq_abs_adj(corpus_name, d_freq_corpus, d_dp):
    """Add adjusted frequencies to frequency dictionary (per item)."""
    d_abs_adj = {}

    for tup in d_freq_corpus["corpus"]:
        d_tup = {}
        dp_score = d_dp[tup]
        adj_freq = d_freq_corpus["corpus"][tup] * (1 - dp_score)
        abs_freq_lapl = d_freq_corpus["corpus"][tup] + 1
        adj_freq_lapl = (d_freq_corpus["corpus"][tup] * (1 - dp_score)) + 1

        d_tup["DP"] = dp_score
        d_tup["abs_freq"] = d_freq_corpus["corpus"][tup]
        d_tup["adj_freq"] = adj_freq
        d_tup["abs_freq_Lapl"] = abs_freq_lapl
        d_tup["adj_freq_Lapl"] = adj_freq_lapl

        d_abs_adj[tup] = d_tup

    d_abs_adj_json = {}

    for tup in d_abs_adj:
        d_abs_adj_json[str(tup)] = d_abs_adj[tup]

    fn_d_abs_adj = corpus_name + "_d_freq_abs_adj.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_abs_adj, d_abs_adj_json)

    return d_abs_adj


def sum_words_desired_pos_abs_adj(corpus_name, d_abs_adj, desired_pos):
    """Add adjusted frequencies to frequency dictionary (per item)."""
    d_sum_abs_adj = {"all": {"abs_freq": 0, "adj_freq": 0, "abs_freq_Lapl": 0, "adj_freq_Lapl": 0, "unique": 0}}

    for pos in desired_pos:
        d_sum_abs_adj[pos] = {"abs_freq": 0, "adj_freq": 0, "abs_freq_Lapl": 0, "adj_freq_Lapl": 0, "unique": 0}

    for tup in d_abs_adj:
        pos = tup[1]
        d_sum_abs_adj["all"]["abs_freq"] += d_abs_adj[tup]["abs_freq"]
        d_sum_abs_adj["all"]["adj_freq"] += d_abs_adj[tup]["adj_freq"]
        d_sum_abs_adj["all"]["abs_freq_Lapl"] += d_abs_adj[tup]["abs_freq_Lapl"]
        d_sum_abs_adj["all"]["adj_freq_Lapl"] += d_abs_adj[tup]["adj_freq_Lapl"]
        d_sum_abs_adj["all"]["unique"] += 1

        d_sum_abs_adj[pos]["abs_freq"] += d_abs_adj[tup]["abs_freq"]
        d_sum_abs_adj[pos]["adj_freq"] += d_abs_adj[tup]["adj_freq"]
        d_sum_abs_adj[pos]["abs_freq_Lapl"] += d_abs_adj[tup]["abs_freq_Lapl"]
        d_sum_abs_adj[pos]["adj_freq_Lapl"] += d_abs_adj[tup]["adj_freq_Lapl"]
        d_sum_abs_adj[pos]["unique"] += 1

    fn_d_sum_abs_adj = corpus_name + "_sum_words_desired_pos_abs_adj.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_sum_abs_adj, d_sum_abs_adj)

    return d_sum_abs_adj


def keyness_calculation(
        corpus_name_sc, corpus_name_rc, lem_or_tok, approx, stat_sign_thresh, degrs_of_freed, keyn_thresh, freq_type,
        keyn_metric, n_ckis_want_analyse, sel_items, d_abs_adj_sc, d_abs_adj_rc, d_sum_abs_adj_sc, d_sum_abs_adj_rc
):
    """Calculate keyness."""
    output_direc = corpus_name_sc + "_VS_" + corpus_name_rc
    fn_keyn = corpus_name_sc + "_keyness_" + keyn_metric + "_" + freq_type
    l_d_keyn = []
    l_d_keyn_top_n = []
    l_d_keyn_sel_items = []
    l_freq_diff = []

    sum_sc = d_sum_abs_adj_sc["all"][freq_type]
    sum_rc = d_sum_abs_adj_rc["all"][freq_type]

    for tup in d_abs_adj_sc:
        freq_sc = d_abs_adj_sc[tup][freq_type]

        if tup in d_abs_adj_rc:
            freq_rc = d_abs_adj_rc[tup][freq_type]
        else:

            if freq_type in ["abs_freq", "adj_freq"]:
                freq_rc = approx
            elif freq_type in ["abs_freq_Lapl", "adj_freq_Lapl"]:
                freq_rc = 1
            else:
                raise ValueError("`frequency_type` is not correctly defined.")

        exp_freq_sc = sum_sc * (freq_sc + freq_rc) / (sum_sc + sum_rc)
        exp_freq_rc = sum_rc * (freq_sc + freq_rc) / (sum_sc + sum_rc)

        norm_freq_1000_sc = freq_sc / sum_sc * 1000
        norm_freq_1000_rc = freq_rc / sum_rc * 1000

        if keyn_metric == "DIFF":
            keyn_score_sc = ((norm_freq_1000_sc - norm_freq_1000_rc) * 100) / norm_freq_1000_rc
            keyn_score_rc = ((norm_freq_1000_rc - norm_freq_1000_sc) * 100) / norm_freq_1000_sc
        elif keyn_metric == "Ratio":
            keyn_score_sc = norm_freq_1000_sc / norm_freq_1000_rc
            keyn_score_rc = norm_freq_1000_rc / norm_freq_1000_sc
        elif keyn_metric == "OddsRatio":
            keyn_score_sc = (freq_sc / (sum_sc - freq_sc)) / (freq_rc / (sum_rc - freq_rc))
            keyn_score_rc = (freq_rc / (sum_rc - freq_rc)) / (freq_sc / (sum_sc - freq_sc))
        elif keyn_metric == "LogRatio":
            keyn_score_sc = np.log2(norm_freq_1000_sc / norm_freq_1000_rc)
            keyn_score_rc = np.log2(norm_freq_1000_rc / norm_freq_1000_sc)
        elif keyn_metric == "DiffCoefficient":
            keyn_score_sc = (norm_freq_1000_sc - norm_freq_1000_rc) / (norm_freq_1000_sc + norm_freq_1000_rc)
            keyn_score_rc = (norm_freq_1000_rc - norm_freq_1000_sc) / (norm_freq_1000_rc + norm_freq_1000_sc)
        else:
            raise ValueError("`keyness_metric` is not correctly defined.")

        if freq_sc == 0 or exp_freq_sc == 0:

            if freq_rc == 0 or exp_freq_rc == 0:
                log_lik = 0
            else:
                log_lik = 2 * (freq_rc * np.log(freq_rc / exp_freq_rc))

        else:

            if freq_rc == 0 or exp_freq_rc == 0:
                log_lik = 2 * (freq_sc * np.log(freq_sc / exp_freq_sc))
            else:
                log_lik = 2 * ((freq_sc * np.log(freq_sc / exp_freq_sc)) + (freq_rc * np.log(freq_rc / exp_freq_rc)))

        bic = log_lik - (degrs_of_freed * np.log(sum_sc + sum_rc))

        d_keyn = {
            "item": tup,
            "keyness": keyn_score_sc,
            "BIC": bic,
            "DP_SC": d_abs_adj_sc[tup]["DP"],
            "abs_freq_SC": d_abs_adj_sc[tup]["abs_freq"],
            "norm_abs_freq_1000_SC": d_abs_adj_sc[tup]["abs_freq"] / d_sum_abs_adj_sc["all"]["abs_freq"] * 1000,
            "adj_freq_SC": d_abs_adj_sc[tup]["adj_freq"],
            "norm_adj_freq_1000_SC": d_abs_adj_sc[tup]["adj_freq"] / d_sum_abs_adj_sc["all"]["adj_freq"] * 1000,
            "abs_freq_Lapl_SC": d_abs_adj_sc[tup]["abs_freq_Lapl"],
            "norm_abs_freq_Lapl_1000_SC":
                d_abs_adj_sc[tup]["abs_freq_Lapl"] / d_sum_abs_adj_sc["all"]["abs_freq_Lapl"] * 1000,
            "adj_freq_Lapl_SC": d_abs_adj_sc[tup]["adj_freq_Lapl"],
            "norm_adj_freq_Lapl_1000_SC":
                d_abs_adj_sc[tup]["adj_freq_Lapl"] / d_sum_abs_adj_sc["all"]["adj_freq_Lapl"] * 1000
        }

        if tup in d_abs_adj_rc:
            d_keyn["DP_RC"] = d_abs_adj_rc[tup]["DP"]
            d_keyn["abs_freq_RC"] = d_abs_adj_rc[tup]["abs_freq"]
            d_keyn["norm_abs_freq_1000_RC"] = \
                d_abs_adj_rc[tup]["abs_freq"] / d_sum_abs_adj_rc["all"]["abs_freq"] * 1000
            d_keyn["adj_freq_RC"] = d_abs_adj_rc[tup]["adj_freq"]
            d_keyn["norm_adj_freq_1000_RC"] = \
                d_abs_adj_rc[tup]["adj_freq"] / d_sum_abs_adj_rc["all"]["adj_freq"] * 1000
            d_keyn["abs_freq_Lapl_RC"] = d_abs_adj_rc[tup]["abs_freq_Lapl"]
            d_keyn["norm_abs_freq_Lapl_1000_RC"] = \
                d_abs_adj_rc[tup]["abs_freq_Lapl"] / d_sum_abs_adj_rc["all"]["abs_freq_Lapl"] * 1000
            d_keyn["adj_freq_Lapl_RC"] = d_abs_adj_rc[tup]["adj_freq_Lapl"]
            d_keyn["norm_adj_freq_Lapl_1000_RC"] = \
                d_abs_adj_rc[tup]["adj_freq_Lapl"] / d_sum_abs_adj_rc["all"]["adj_freq_Lapl"] * 1000
        else:
            d_keyn["DP_RC"] = "NA"
            d_keyn["abs_freq_RC"] = approx
            d_keyn["norm_abs_freq_1000_RC"] = approx / d_sum_abs_adj_rc["all"]["abs_freq"] * 1000
            d_keyn["adj_freq_RC"] = approx
            d_keyn["norm_adj_freq_1000_RC"] = approx / d_sum_abs_adj_rc["all"]["adj_freq"] * 1000
            d_keyn["abs_freq_Lapl_RC"] = 1
            d_keyn["norm_abs_freq_Lapl_1000_RC"] = 1 / d_sum_abs_adj_rc["all"]["abs_freq_Lapl"] * 1000
            d_keyn["adj_freq_Lapl_RC"] = 1
            d_keyn["norm_adj_freq_Lapl_1000_RC"] = 1 / d_sum_abs_adj_rc["all"]["adj_freq_Lapl"] * 1000

        l_d_keyn.append(d_keyn)

        d_keyn_top_n = {}

        if bic >= stat_sign_thresh and keyn_score_sc > keyn_thresh:
            d_keyn_top_n["item"] = tup
            d_keyn_top_n["DP_SC"] = d_abs_adj_sc[tup]["DP"]

            if tup in d_abs_adj_rc:
                d_keyn_top_n["DP_RC"] = d_abs_adj_rc[tup]["DP"]
            else:
                d_keyn_top_n["DP_RC"] = "NA"

            d_keyn_top_n["freq_SC"] = freq_sc
            d_keyn_top_n["freq_RC"] = freq_rc
            d_keyn_top_n["exp_freq_SC"] = exp_freq_sc
            d_keyn_top_n["exp_freq_RC"] = exp_freq_rc
            d_keyn_top_n["norm_freq_1000_SC"] = norm_freq_1000_sc
            d_keyn_top_n["norm_freq_1000_RC"] = norm_freq_1000_rc
            d_keyn_top_n["BIC"] = bic
            d_keyn_top_n["LL"] = log_lik
            d_keyn_top_n["keyn_SC"] = keyn_score_sc
            d_keyn_top_n["keyn_RC"] = keyn_score_rc

            l_d_keyn_top_n.append(d_keyn_top_n)
            l_freq_diff.append([keyn_score_sc, keyn_score_rc])

        d_keyn_sel_items = {}

        if tup in sel_items:
            d_keyn_sel_items["item"] = tup
            d_keyn_sel_items["keyness"] = keyn_score_sc
            d_keyn_sel_items["BIC"] = bic
            d_keyn_sel_items["DP_SC"] = d_abs_adj_sc[tup]["DP"]
            d_keyn_sel_items["abs_freq_SC"] = d_abs_adj_sc[tup]["abs_freq"]
            d_keyn_sel_items["norm_abs_freq_1000_SC"] = \
                d_abs_adj_sc[tup]["abs_freq"] / d_sum_abs_adj_sc["all"]["abs_freq"] * 1000
            d_keyn_sel_items["adj_freq_SC"] = d_abs_adj_sc[tup]["adj_freq"]
            d_keyn_sel_items["norm_adj_freq_1000_SC"] = \
                d_abs_adj_sc[tup]["adj_freq"] / d_sum_abs_adj_sc["all"]["adj_freq"] * 1000
            d_keyn_sel_items["abs_freq_Lapl_SC"] = d_abs_adj_sc[tup]["abs_freq_Lapl"]
            d_keyn_sel_items["norm_abs_freq_Lapl_1000_SC"] = \
                d_abs_adj_sc[tup]["abs_freq_Lapl"] / d_sum_abs_adj_sc["all"]["abs_freq_Lapl"] * 1000
            d_keyn_sel_items["adj_freq_Lapl_SC"] = d_abs_adj_sc[tup]["adj_freq_Lapl"]
            d_keyn_sel_items["norm_adj_freq_Lapl_1000_SC"] = \
                d_abs_adj_sc[tup]["adj_freq_Lapl"] / d_sum_abs_adj_sc["all"]["adj_freq_Lapl"] * 1000

            if tup in d_abs_adj_rc:
                d_keyn_sel_items["DP_RC"] = d_abs_adj_rc[tup]["DP"]
                d_keyn_sel_items["abs_freq_RC"] = d_abs_adj_rc[tup]["abs_freq"]
                d_keyn_sel_items["norm_abs_freq_1000_RC"] = \
                    d_abs_adj_rc[tup]["abs_freq"] / d_sum_abs_adj_rc["all"]["abs_freq"] * 1000
                d_keyn_sel_items["adj_freq_RC"] = d_abs_adj_rc[tup]["adj_freq"]
                d_keyn_sel_items["norm_adj_freq_1000_RC"] = \
                    d_abs_adj_rc[tup]["adj_freq"] / d_sum_abs_adj_rc["all"]["adj_freq"] * 1000
                d_keyn_sel_items["abs_freq_Lapl_RC"] = d_abs_adj_rc[tup]["abs_freq_Lapl"]
                d_keyn_sel_items["norm_abs_freq_Lapl_1000_RC"] = \
                    d_abs_adj_rc[tup]["abs_freq_Lapl"] / d_sum_abs_adj_rc["all"]["abs_freq_Lapl"] * 1000
                d_keyn_sel_items["adj_freq_Lapl_RC"] = d_abs_adj_rc[tup]["adj_freq_Lapl"]
                d_keyn_sel_items["norm_adj_freq_Lapl_1000_RC"] = \
                    d_abs_adj_rc[tup]["adj_freq_Lapl"] / d_sum_abs_adj_rc["all"]["adj_freq_Lapl"] * 1000

            else:
                d_keyn_sel_items["DP_RC"] = "NA"
                d_keyn_sel_items["abs_freq_RC"] = approx
                d_keyn_sel_items["norm_abs_freq_1000_RC"] = approx / d_sum_abs_adj_rc["all"]["abs_freq"] * 1000
                d_keyn_sel_items["adj_freq_RC"] = approx
                d_keyn_sel_items["norm_adj_freq_1000_RC"] = approx / d_sum_abs_adj_rc["all"]["adj_freq"] * 1000
                d_keyn_sel_items["abs_freq_Lapl_RC"] = 1
                d_keyn_sel_items["norm_abs_freq_Lapl_1000_RC"] = 1 / d_sum_abs_adj_rc["all"]["abs_freq_Lapl"] * 1000
                d_keyn_sel_items["adj_freq_Lapl_RC"] = 1
                d_keyn_sel_items["norm_adj_freq_Lapl_1000_RC"] = 1 / d_sum_abs_adj_rc["all"]["adj_freq_Lapl"] * 1000

            l_d_keyn_sel_items.append(d_keyn_sel_items)

    dump_json(os.path.join("output", output_direc), fn_keyn + "_all.json", l_d_keyn)
    dump_json(os.path.join("output", output_direc), fn_keyn + "_selection.json", l_d_keyn_sel_items)

    # clustering
    if len(l_d_keyn_top_n) >= n_ckis_want_analyse:
        array_freq_differences = np.array(l_freq_diff)
        n_ckis = len(l_d_keyn_top_n)
        n_clusters = int(round(n_ckis / n_ckis_want_analyse))
        cluster = AgglomerativeClustering(n_clusters=n_clusters, affinity='euclidean', linkage='average')
        l_cluster_labels = list(cluster.fit_predict(array_freq_differences))

        # add cluster to l_d_keyn_top-N
        for dic, label in zip(l_d_keyn_top_n, l_cluster_labels):
            dic["cluster"] = int(label)

        counter_n_clusters = 0
        l_average_keyn_cluster = []  # list of tuples with average keyness for each cluster (because clusters not properly ranked according to effect size)

        for cluster in range(n_clusters):
            sum_keyn_cluster = 0
            label = counter_n_clusters
            label_name = "cluster_" + str(label)
            counter_item = 0

            for dic in l_d_keyn_top_n:

                if dic["cluster"] == label:
                    counter_item += 1
                    sum_keyn_cluster += dic["keyn_SC"]

            average_keyn_cluster = sum_keyn_cluster / counter_item
            l_average_keyn_cluster.append((label_name, average_keyn_cluster))
            counter_n_clusters += 1

        sorted_l_sum_keyn_cluster = sorted(l_average_keyn_cluster, key=operator.itemgetter(1), reverse=True)  # sort l_average_keyness_cluster by average keyness (descending)

        counter_n_clusters_2 = 0
        cluster_rank = 1

        for cluster in range(n_clusters):
            label = counter_n_clusters_2
            label_name_sorted_l_sum_keyn_cluster = sorted_l_sum_keyn_cluster[label][0]

            for dic in l_d_keyn_top_n:
                label_name_d_keyn = "cluster_" + str(dic["cluster"])

                if label_name_d_keyn == label_name_sorted_l_sum_keyn_cluster:
                    dic["cluster_rank"] = int(cluster_rank)

            counter_n_clusters_2 += 1
            cluster_rank += 1

        sorted_l_d_keyn_top_n = sorted(sorted(sorted(sorted(
            l_d_keyn_top_n,
            key=lambda i: i["item"]),
            key=lambda i: i["freq_SC"], reverse=True),
            key=lambda i: i["keyn_SC"], reverse=True),
            key=lambda i: i["cluster_rank"]
        )  # sort key items by 1) cluster_rank (ascending); 2) keyness (descending); 3) freq_SC (descending); 4) pos_lem_or_tok (ascending)

        last_cluster = sorted_l_d_keyn_top_n[n_ckis_want_analyse]["cluster_rank"]

        l_d_keyn_top_n = []

        for dic in sorted_l_d_keyn_top_n:

            if dic["cluster_rank"] <= last_cluster:
                l_d_keyn_top_n.append(dic)

    else:

        for dic in l_d_keyn_top_n:
            dic["cluster"] = 0
            dic["cluster_rank"] = 0

        sorted_l_d_keyn_top_n = sorted(sorted(sorted(
            l_d_keyn_top_n,
            key=lambda i: i["item"]),
            key=lambda i: i["freq_SC"], reverse=True),
            key=lambda i: i["keyn_SC"], reverse=True
        )  # sort key items by 1) keyness (descending); 2) freq_SC (descending); 3) pos_lem_or_tok (ascending)

        l_d_keyn_top_n = []

        for dic in sorted_l_d_keyn_top_n:
            l_d_keyn_top_n.append(dic)

    dump_json(os.path.join("output", output_direc), fn_keyn + "_top-N.json", l_d_keyn_top_n)

    # write final results into XLSX

    headers_ws1_and_3 = [
        lem_or_tok, "pos", "keyness", "BIC", "DP_SC", "DP_RC", "AbF_SC", "NAbF_SC", "AbF_RC", "NAbF_RC", "AdF_SC",
        "NAdF_SC", "AdF_RC", "NAdF_RC", "AbFLpl_SC", "NAbFLpl_SC", "AbFLpl_RC", "NAbFLpl_RC", "AdFLpl_SC", "NAdFLpl_SC",
        "AdFLpl_RC", "NAdFLpl_RC"
    ]
    headers_ws2 = [
        lem_or_tok, "pos", "keyness_SC", "keyness_RC", "BIC", "LL", "DP_SC", "DP_RC", "freq_SC", "freq_RC",
        "exp_freq_SC", "exp_freq_RC", "norm_freq_1000_SC", "norm_freq_1000_RC", "cluster", "cluster_rank"
    ]

    wb = xlsxwriter.Workbook(os.path.join("output", output_direc, fn_keyn + ".xlsx"))
    ws1 = wb.add_worksheet("all")
    ws2 = wb.add_worksheet("top-N")
    ws3 = wb.add_worksheet("selection")

    column_ws1_and_3 = 0

    for header in headers_ws1_and_3:
        ws1.write(0, column_ws1_and_3, header)
        ws3.write(0, column_ws1_and_3, header)
        column_ws1_and_3 += 1

    column_ws2 = 0

    for header in headers_ws2:
        ws2.write(0, column_ws2, header)
        column_ws2 += 1

    row_ws1 = 1
    row_ws2 = 1
    row_ws3 = 1

    for dic in l_d_keyn:
        ws1.write(row_ws1, 0, str(dic["item"][0]))
        ws1.write(row_ws1, 1, str(dic["item"][1]))
        ws1.write(row_ws1, 2, round(dic["keyness"], 2))
        ws1.write(row_ws1, 3, round(dic["BIC"], 2))
        ws1.write(row_ws1, 4, round(dic["DP_SC"], 2))

        if isinstance(dic["DP_RC"], float):
            ws1.write(row_ws1, 5, round(dic["DP_RC"], 2))
        else:
            ws1.write(row_ws1, 5, dic["DP_RC"])

        ws1.write(row_ws1, 6, round(dic["abs_freq_SC"], 2))
        ws1.write(row_ws1, 7, round(dic["norm_abs_freq_1000_SC"], 2))
        ws1.write(row_ws1, 8, round(dic["abs_freq_RC"], 2))
        ws1.write(row_ws1, 9, round(dic["norm_abs_freq_1000_RC"], 2))
        ws1.write(row_ws1, 10, round(dic["adj_freq_SC"], 2))
        ws1.write(row_ws1, 11, round(dic["norm_adj_freq_1000_SC"], 2))
        ws1.write(row_ws1, 12, round(dic["adj_freq_RC"], 2))
        ws1.write(row_ws1, 13, round(dic["norm_adj_freq_1000_RC"], 2))
        ws1.write(row_ws1, 14, round(dic["abs_freq_Lapl_SC"], 2))
        ws1.write(row_ws1, 15, round(dic["norm_abs_freq_Lapl_1000_SC"], 2))
        ws1.write(row_ws1, 16, round(dic["abs_freq_Lapl_RC"], 2))
        ws1.write(row_ws1, 17, round(dic["norm_abs_freq_Lapl_1000_RC"], 2))
        ws1.write(row_ws1, 18, round(dic["adj_freq_Lapl_SC"], 2))
        ws1.write(row_ws1, 19, round(dic["norm_adj_freq_Lapl_1000_SC"], 2))
        ws1.write(row_ws1, 20, round(dic["adj_freq_Lapl_RC"], 2))
        ws1.write(row_ws1, 21, round(dic["norm_adj_freq_Lapl_1000_RC"], 2))

        row_ws1 += 1

    for dic in l_d_keyn_top_n:
        ws2.write(row_ws2, 0, str(dic["item"][0]))
        ws2.write(row_ws2, 1, str(dic["item"][1]))
        ws2.write(row_ws2, 2, round(dic["keyn_SC"], 2))
        ws2.write(row_ws2, 3, round(dic["keyn_RC"], 2))
        ws2.write(row_ws2, 4, round(dic["LL"], 2))
        ws2.write(row_ws2, 5, round(dic["BIC"], 2))
        ws2.write(row_ws2, 6, round(dic["DP_SC"], 2))

        if isinstance(dic["DP_RC"], float):
            ws2.write(row_ws2, 7, round(dic["DP_RC"], 2))
        else:
            ws2.write(row_ws2, 7, dic["DP_RC"])

        ws2.write(row_ws2, 8, round(dic["freq_SC"], 2))
        ws2.write(row_ws2, 9, round(dic["freq_RC"], 2))
        ws2.write(row_ws2, 10, round(dic["exp_freq_SC"], 2))
        ws2.write(row_ws2, 11, round(dic["exp_freq_RC"], 2))
        ws2.write(row_ws2, 12, round(dic["norm_freq_1000_SC"], 2))
        ws2.write(row_ws2, 13, round(dic["norm_freq_1000_RC"], 2))
        ws2.write(row_ws2, 14, round(dic["cluster"], 2))
        ws2.write(row_ws2, 15, round(dic["cluster_rank"], 2))

        row_ws2 += 1

    for dic in l_d_keyn_sel_items:
        ws3.write(row_ws3, 0, str(dic["item"][0]))
        ws3.write(row_ws3, 1, str(dic["item"][1]))
        ws3.write(row_ws3, 2, round(dic["keyness"], 2))
        ws3.write(row_ws3, 3, round(dic["BIC"], 2))
        ws3.write(row_ws3, 4, round(dic["DP_SC"], 2))

        if isinstance(dic["DP_RC"], float):
            ws3.write(row_ws3, 5, round(dic["DP_RC"], 2))
        else:
            ws3.write(row_ws3, 5, dic["DP_RC"])

        ws3.write(row_ws3, 6, round(dic["abs_freq_SC"], 2))
        ws3.write(row_ws3, 7, round(dic["norm_abs_freq_1000_SC"], 2))
        ws3.write(row_ws3, 8, round(dic["abs_freq_RC"], 2))
        ws3.write(row_ws3, 9, round(dic["norm_abs_freq_1000_RC"], 2))
        ws3.write(row_ws3, 10, round(dic["adj_freq_SC"], 2))
        ws3.write(row_ws3, 11, round(dic["norm_adj_freq_1000_SC"], 2))
        ws3.write(row_ws3, 12, round(dic["adj_freq_RC"], 2))
        ws3.write(row_ws3, 13, round(dic["norm_adj_freq_1000_RC"], 2))
        ws3.write(row_ws3, 14, round(dic["abs_freq_Lapl_SC"], 2))
        ws3.write(row_ws3, 15, round(dic["norm_abs_freq_Lapl_1000_SC"], 2))
        ws3.write(row_ws3, 16, round(dic["abs_freq_Lapl_RC"], 2))
        ws3.write(row_ws3, 17, round(dic["norm_abs_freq_Lapl_1000_RC"], 2))
        ws3.write(row_ws3, 18, round(dic["adj_freq_Lapl_SC"], 2))
        ws3.write(row_ws3, 19, round(dic["norm_adj_freq_Lapl_1000_SC"], 2))
        ws3.write(row_ws3, 20, round(dic["adj_freq_Lapl_RC"], 2))
        ws3.write(row_ws3, 21, round(dic["norm_adj_freq_Lapl_1000_RC"], 2))

        row_ws3 += 1

    wb.close()

    return l_d_keyn, l_d_keyn_top_n, l_d_keyn_sel_items


def d_meta(corpus_name, desired_pos, lem_or_tok, maintain_subcorpora, div_n_docs_by):
    """Construct meta file."""
    d_meta_corpus = {
        "desired_pos": desired_pos,
        "lemma_or_token": lem_or_tok,
        "maintain_subcorpora": maintain_subcorpora,
        "divide_number_docs_by": div_n_docs_by
    }

    dump_json(os.path.join("prep", corpus_name), "meta.json", d_meta_corpus)


def corpora_to_d_freq(
        corpus_name, input_corpus, mapping_custom_to_ud, mapping_ud_to_custom, desired_pos, lem_or_tok,
        maintain_subcorpora, div_n_docs_by
):
    """STEP_1: convert corpora into frequency dictionaries (data stored per corpus in "prep" folder)."""
    d_freq_corpus, d_freq_cps, d_cps = d_freq(
        corpus_name, input_corpus, mapping_custom_to_ud, mapping_ud_to_custom, desired_pos, lem_or_tok,
        maintain_subcorpora, div_n_docs_by
    )
    d_sum_cps = sum_words_desired_pos(
        corpus_name, d_freq_corpus, desired_pos, d_freq_cps, d_cps, maintain_subcorpora
    )

    return d_freq_corpus, d_freq_cps, d_sum_cps


def dispersion(corpus_name, d_freq_corpus, d_freq_cps, d_sum_cps, desired_pos):
    """STEP_2: apply dispersion metric (DPnorm; Gries, 2008; Lijffijt & Gries, 2012), calculate adjusted frequencies and
    update frequency dictionaries (data stored per corpus in "prep" folder)."""
    d_dp = dp(corpus_name, d_freq_corpus, d_freq_cps, d_sum_cps)
    d_freq_abs_adj_corpus = d_freq_abs_adj(corpus_name, d_freq_corpus, d_dp)
    d_sum_abs_adj = sum_words_desired_pos_abs_adj(corpus_name, d_freq_abs_adj_corpus, desired_pos)

    return d_freq_abs_adj_corpus, d_sum_abs_adj


def keyness(
        name_sc, name_rc, lem_or_tok, approx, stat_sign_thresh, degrs_of_freed, keyn_thresh, freq_type, keyn_metric,
        n_ckis_want_analyse, sel_items, d_freq_abs_adj_sc, d_sum_abs_adj_sc, d_freq_abs_adj_rc, d_sum_abs_adj_rc
):
    """STEP_3: calculate keyness (data stored in "[SC]_VS_[RC]" folder)."""
    l_d_keyn, l_d_keyn_top_n, l_d_keyn_sel_items = keyness_calculation(
        name_sc, name_rc, lem_or_tok, approx, stat_sign_thresh, degrs_of_freed, keyn_thresh, freq_type, keyn_metric,
        n_ckis_want_analyse, sel_items, d_freq_abs_adj_sc, d_freq_abs_adj_rc, d_sum_abs_adj_sc, d_sum_abs_adj_rc
    )

    return l_d_keyn, l_d_keyn_top_n, l_d_keyn_sel_items


def meta(corpus_name, desired_pos, lem_or_tok, maintain_subcorpora, div_n_docs_by):
    """STEP_4: store information of last query in meta file (data stored per corpus in "prep" folder)."""
    d_meta(corpus_name, desired_pos, lem_or_tok, maintain_subcorpora, div_n_docs_by)
