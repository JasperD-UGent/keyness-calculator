from .process_JSONs import dump_json
import os
import sys
from typing import Dict, Tuple


def dp(corpus_name: str, d_freq_corpus: Dict, d_freq_cps: Dict, d_sum_cps: Dict) -> Dict:
    """Calculate dispersion values (DPnorm).
    :param corpus_name: name of the corpus.
    :param d_freq_corpus: frequency dictionary of the entire corpus.
    :param d_freq_cps: frequency dictionary per corpus part.
    :param d_sum_cps: dictionary containing the sum of the words per corpus part.
    :return: a dictionary containing the dispersion values.
    """

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

    fn_d_dp = f"{corpus_name}_DP.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_dp, d_dp_norm_json)

    return d_dp_norm


def d_freq_abs_adj(corpus_name: str, d_freq_corpus: Dict, d_dp: Dict) -> Dict:
    """Add adjusted frequencies to frequency dictionary (per item).
    :param corpus_name: name of the corpus.
    :param d_freq_corpus: frequency dictionary of the entire corpus.
    :param d_dp: dictionary containing the dispersion values.
    :return: the frequency dictionary enriched with adjusted frequency values.
    """
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

    fn_d_abs_adj = f"{corpus_name}_d_freq_abs_adj.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_abs_adj, d_abs_adj_json)

    return d_abs_adj


def sum_words_desired_pos_abs_adj(corpus_name: str, d_abs_adj: Dict, desired_pos: Tuple) -> Dict:
    """Add adjusted frequencies to frequency dictionary (totals).
    :param corpus_name: name of the corpus.
    :param d_abs_adj: frequency dictionary enriched with adjusted frequency values.
    :param desired_pos: tuple of UD tags which should be taken into account in the keyness calculations.
    :return: the frequency dictionary enriched with adjusted frequency values.
    """
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

    fn_d_sum_abs_adj = f"{corpus_name}_sum_words_desired_pos_abs_adj.json"
    dump_json(os.path.join("prep", corpus_name), fn_d_sum_abs_adj, d_sum_abs_adj)

    return d_sum_abs_adj


def dispersion(
        corpus_name: str, d_freq_corpus: Dict, d_freq_cps: Dict, d_sum_cps: Dict, desired_pos: Tuple
) -> Tuple[Dict, Dict]:
    """STEP_2: apply dispersion metric (DPnorm; Gries, 2008; Lijffijt & Gries, 2012), calculate adjusted frequencies and
    update frequency dictionaries (data stored per corpus in "prep" folder).
    :param corpus_name: name of the corpus.
    :param d_freq_corpus: frequency dictionary of the entire corpus
    :param d_freq_cps: frequency dictionary per corpus part
    :param d_sum_cps: dictionary containing the sum of the words per corpus part
    :param desired_pos: tuple of UD tags which should be taken into account in the keyness calculations.
    :return: a tuple containing a frequency dictionary enriched with adjusted frequency values (per item) and a
        frequency dictionary enriched with adjusted frequency values (totals).
    """
    d_dp = dp(corpus_name, d_freq_corpus, d_freq_cps, d_sum_cps)
    d_freq_abs_adj_corpus = d_freq_abs_adj(corpus_name, d_freq_corpus, d_dp)
    d_sum_abs_adj = sum_words_desired_pos_abs_adj(corpus_name, d_freq_abs_adj_corpus, desired_pos)

    return d_freq_abs_adj_corpus, d_sum_abs_adj
