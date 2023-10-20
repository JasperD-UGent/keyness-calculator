from .process_JSONs import dump_json
import numpy as np
import operator
import os
from sklearn.cluster import AgglomerativeClustering
import sys
from typing import Dict, List, Tuple
import xlsxwriter


def keyness(
        corpus_name_sc: str,
        corpus_name_rc: str,
        lem_or_tok: str,
        approx: float,
        stat_sign_thresh: int,
        degrs_of_freed: int,
        keyn_thresh: float,
        freq_type: str,
        keyn_metric: str,
        n_ckis_want_analyse: int,
        sel_items: List,
        d_abs_adj_sc: Dict,
        d_sum_abs_adj_sc: Dict,
        d_abs_adj_rc: Dict,
        d_sum_abs_adj_rc: Dict
) -> Tuple[List, List, List]:
    """STEP_3: calculate keyness (data stored in "[SC]_VS_[RC]" folder).
    :param corpus_name_sc: name of the study corpus.
    :param corpus_name_rc: name of the reference corpus.
    :param lem_or_tok: defines whether to calculate frequencies on token or lemma level.
    :param approx: float by which zero frequencies are approximated.
    :param stat_sign_thresh: statistical significance threshold for BIC values.
    :param degrs_of_freed: degrees of freedom used to calculate log likelihood values.
    :param keyn_thresh: keyness value threshold for an item to be considered key to the study corpus.
    :param freq_type: frequency type based on which keyness values are calculated.
    :param keyn_metric: keyness metric used to perform the keyness calculations.
    :param n_ckis_want_analyse: number of candidate key items (CKIs) you wish to analyse.
    :param sel_items: list of items you wish to analyse.
    :param d_abs_adj_sc: frequency dictionary enriched with adjusted frequency values (per item) for the study corpus.
    :param d_sum_abs_adj_sc: frequency dictionary enriched with adjusted frequency values (totals) for the study corpus.
    :param d_abs_adj_rc: frequency dictionary enriched with adjusted frequency values (per item) for the reference
        corpus.
    :param d_sum_abs_adj_rc: frequency dictionary enriched with adjusted frequency values (totals) for the reference
        corpus.
    :return: a tuple containing three lists: a list containing the keyness analysis for all items, a list containing the
        keyness analysis for the top-N CKIs, and a list containing the keyness analysis for the selected items.
    """
    output_direc = f"{corpus_name_sc}_VS_{corpus_name_rc}"
    fn_keyn = f"{corpus_name_sc}_keyness_{keyn_metric}_{freq_type}"
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

        if keyn_metric == "%DIFF":
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

    dump_json(os.path.join("output", output_direc), f"{fn_keyn}_all.json", l_d_keyn)
    dump_json(os.path.join("output", output_direc), f"{fn_keyn}_selection.json", l_d_keyn_sel_items)

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
                label_name_d_keyn = f"cluster_{dic['cluster']}"

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

    dump_json(os.path.join("output", output_direc), f"{fn_keyn}_top-N.json", l_d_keyn_top_n)

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

    wb = xlsxwriter.Workbook(os.path.join("output", output_direc, f"{fn_keyn}.xlsx"))
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
