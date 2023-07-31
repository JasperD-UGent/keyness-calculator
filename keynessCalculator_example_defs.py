from utils.STEP_1 import corpora_to_d_freq
from utils.STEP_2 import dispersion
from utils.STEP_3 import keyness
from utils.STEP_4 import meta
from utils.keynessCalculator_support import extract_corpus_name, check_meta
from utils.process_JSONs import load_json, load_json_str_to_obj
import os
import sys
from typing import Dict, List, Optional, Tuple, Union


def init_keyness_calculator(
        input_sc: Union[str, Tuple],
        input_rc: Union[str, Tuple],
        *,
        mapping_custom_to_ud: Optional[Dict] = None,
        mapping_ud_to_custom: Optional[Dict] = None,
        desired_pos: Tuple = ("NOUN", "ADJ", "VERB", "ADV"),
        lemma_or_token: str = "lemma",
        maintain_subcorpora: bool = True,
        divide_number_docs_by: int = 10,
        approximation: float = 0.000000000000000001,
        statistical_significance_threshold_bic: int = 2,
        degrees_of_freedom: int = 1,
        keyness_threshold: float = 0.0,
        frequency_type: str = "adj_freq_Lapl",
        keyness_metric: str = "LogRatio",
        number_ckis_want_analyse: int = 100,
        selection_items: Optional[List] = None,
) -> Dict:
    """Initialise the keyness calculator.
    :param input_sc: study corpus. Either a string to a folder (root = corpus; subdirectories = subcorpora;
        files in subdirectories = documents) or a tuple containing the corpus name and the corpus in dictionary format
        (keys = subcorpora; values = a list of lists [= documents] containing token-POS-lemma tuples).
    :param input_rc: reference corpus. Either a path to a folder (root = corpus; subdirectories = subcorpora;
        files in subdirectories = documents) or a tuple containing the corpus name and the corpus in dictionary format
        (keys = subcorpora; values = a list of lists [= documents] containing token-POS-lemma tuples).
    :param mapping_custom_to_ud: if you work with custom POS tags, dictionary which maps custom tags to UD counterparts.
    :param mapping_ud_to_custom: if you work with custom POS tags, dictionary which maps UD tags to custom counterparts.
    :param desired_pos: tuple of UD tags which should be taken into account in the keyness calculations.
        Defaults to ("NOUN", "ADJ", "VERB", "ADV").
    :param lemma_or_token: defines whether to calculate frequencies on token or lemma level. Choose between: "lemma",
        "token". Defaults to "lemma".
    :param maintain_subcorpora: when working with adjusted frequencies, boolean value which defines whether dispersion
        is based on existing subcorpora, or whether all documents are merged and randomly split into new subcorpora.
        Defaults to True.
    :param divide_number_docs_by: when working with adjusted frequencies, number by which the total number of documents
        is divided to arrive at the number of new randomly generated subcorpora. Defaults to 10.
    :param approximation: float by which zero frequencies are approximated. Defaults to 0.000000000000000001.
    :param statistical_significance_threshold_bic: statistical significance threshold for BIC values. Defaults to 2
        (see also Gabrielatos [2018] and Wilson [2013]).
    :param degrees_of_freedom: degrees of freedom used to calculate log likelihood values. Defaults to 1 (which is
        the default number of degrees of freedom for keyness calculations).
    :param keyness_threshold: keyness value threshold for an item to be considered key to the study corpus. Defaults to
        0.
    :param frequency_type: frequency type based on which keyness values are calculated. Choose between: "abs_freq"
        (absolute frequency), "adj_freq" (adjusted frequency), "abs_freq_Lapl" (absolute frequency + Laplace smoothing),
        "adj_freq_Lapl" (adjusted frequency + Laplace smoothing). Defaults to "adj_freq_Lapl".
    :param keyness_metric: keyness metric used to perform the keyness calculations. Choose between: "DIFF" (Gabrielatos
        & Marchi, 2011), "Ratio" (Kilgarriff, 2009), "OddsRatio" (Everitt, 2002; Pojanapunya & Watson Todd, 2016),
        "LogRatio" (Hardie, 2014), "DiffCoefficient" (Hofland & Johansson, 1982). Defaults to "LogRatio".
    :param number_ckis_want_analyse: number of candidate key items (CKIs) you wish to analyse (results for these items
        are saved in "top-N" sheet in Excel file and in separate JSON file). Defaults to 100.
    :param selection_items: list of items you wish to analyse (results for these items are saved in "selection" sheet
        in Excel file and in separate JSON file). Format: tuple of lemma/token and its POS tag.
    :return: dictionary containing results (key "keyness_all" = results for all items; key "keyness_top-N = results
        for top N CKIs; key "keyness_selection" = results for custom selection of items)
    """
    mapping_custom_to_ud = {
        "ADJ": "ADJ", "ADV": "ADV", "INTJ": "INTJ", "NOUN": "NOUN", "PROPN": "PROPN", "VERB": "VERB", "ADP": "ADP",
        "AUX": "AUX", "CCONJ": "CCONJ", "DET": "DET", "NUM": "NUM", "PART": "PART", "PRON": "PRON", "SCONJ": "SCONJ",
        "PUNCT": "PUNCT", "SYM": "SYM", "X": "X"
    } if mapping_custom_to_ud is None else mapping_custom_to_ud
    mapping_ud_to_custom = {
        "ADJ": ["ADJ"], "ADV": ["ADV"], "INTJ": ["INTJ"], "NOUN": ["NOUN"], "PROPN": ["PROPN"], "VERB": ["VERB"],
        "ADP": ["ADP"], "AUX": ["AUX"], "CCONJ": ["CCONJ"], "DET": ["DET"], "NUM": ["NUM"], "PART": ["PART"],
        "PRON": ["PRON"], "SCONJ": ["SCONJ"], "PUNCT": ["PUNCT"], "SYM": ["SYM"], "X": ["X"]
    } if mapping_ud_to_custom is None else mapping_ud_to_custom
    selection_items = [] if selection_items is None else selection_items
    
    # define additional variables
    name_sc = extract_corpus_name(input_sc)
    name_rc = extract_corpus_name(input_rc)

    # check meta file last keyness calculation with selected corpora
    load_from_files_sc = check_meta(
        name_sc, desired_pos, lemma_or_token, maintain_subcorpora, divide_number_docs_by
    )
    load_from_files_rc = check_meta(
        name_rc, desired_pos, lemma_or_token, maintain_subcorpora, divide_number_docs_by
    )

    # perform steps

    #   - STEP_1 and STEP_2: convert corpora into frequency dictionaries (data stored per corpus in "prep" folder) and
    #   apply dispersion metric (DPnorm; Gries, 2008; Lijffijt & Gries, 2012), calculate adjusted frequencies and update
    #   frequency dictionaries (data stored per corpus in "prep" folder)
    print("Performing STEP_1 and STEP_2.")

    if load_from_files_sc:
        d_freq_abs_adj_sc = load_json_str_to_obj(os.path.join("prep", name_sc, f"{name_sc}_d_freq_abs_adj.json"))
        d_sum_abs_adj_sc = load_json(os.path.join("prep", name_sc, f"{name_sc}_sum_words_desired_pos_abs_adj.json"))
    else:
        d_freq_sc, d_freq_cps_sc, d_sum_cps_sc = corpora_to_d_freq(
            name_sc, input_sc, mapping_custom_to_ud, mapping_ud_to_custom, desired_pos, lemma_or_token,
            maintain_subcorpora, divide_number_docs_by
        )
        d_freq_abs_adj_sc, d_sum_abs_adj_sc = dispersion(name_sc, d_freq_sc, d_freq_cps_sc, d_sum_cps_sc, desired_pos)

    if load_from_files_rc:
        d_freq_abs_adj_rc = load_json_str_to_obj(os.path.join("prep", name_rc, f"{name_rc}_d_freq_abs_adj.json"))
        d_sum_abs_adj_rc = load_json(os.path.join("prep", name_rc, f"{name_rc}_sum_words_desired_pos_abs_adj.json"))
    else:
        d_freq_rc, d_freq_cps_rc, d_sum_cps_rc = corpora_to_d_freq(
            name_rc, input_rc, mapping_custom_to_ud, mapping_ud_to_custom, desired_pos, lemma_or_token,
            maintain_subcorpora, divide_number_docs_by
        )
        d_freq_abs_adj_rc, d_sum_abs_adj_rc = dispersion(name_rc, d_freq_rc, d_freq_cps_rc, d_sum_cps_rc, desired_pos)

    #   - STEP_3: calculate keyness (data stored in "[SC]_VS_[RC]" folder)
    print("Performing STEP_3.")

    l_d_keyn_sc, l_d_keyn_top_n_sc, l_d_keyn_sel_items_sc = keyness(
        name_sc, name_rc, lemma_or_token, approximation, statistical_significance_threshold_bic, degrees_of_freedom,
        keyness_threshold, frequency_type, keyness_metric, number_ckis_want_analyse, selection_items, d_freq_abs_adj_sc,
        d_sum_abs_adj_sc, d_freq_abs_adj_rc, d_sum_abs_adj_rc
    )

    #   - STEP_4: store information of last query in meta file (data stored per corpus in "prep" folder)
    print("Performing STEP_4.")

    meta(name_sc, desired_pos, lemma_or_token, maintain_subcorpora, divide_number_docs_by)
    meta(name_rc, desired_pos, lemma_or_token, maintain_subcorpora, divide_number_docs_by)

    return {"all": l_d_keyn_sc, "top-N": l_d_keyn_top_n_sc, "selection": l_d_keyn_sel_items_sc}
