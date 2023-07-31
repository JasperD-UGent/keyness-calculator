from .process_JSONs import load_json
import os
import sys
from typing import Tuple, Union


def check_meta(
        corpus_name: str, desired_pos: Tuple, lem_or_tok: str, maintain_subcorpora: bool, div_n_docs_by: int
) -> bool:
    """Check if information in meta file corresponds to current query.
    :param corpus_name: name of the corpus.
    :param desired_pos: tuple of UD tags which should be taken into account in the keyness calculations.
    :param lem_or_tok: defines whether to calculate frequencies on token or lemma level.
    :param maintain_subcorpora: when working with adjusted frequencies, boolean value which defines whether dispersion
        is based on existing subcorpora, or whether all documents are merged and randomly split into new subcorpora.
    :param div_n_docs_by: when working with adjusted frequencies, number by which the total number of documents is
        divided to arrive at the number of new randomly generated subcorpora.
    :return: `True` if corresponds, `False` if not.
    """

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


def extract_corpus_name(corpus_input: Union[str, Tuple]) -> str:
    """Extract corpus name based on provided input data.
    :param corpus_input: provided input data for the corpus.
    :return: the name of the corpus.
    """

    if type(corpus_input) == str:
        corpus_name = os.path.basename(corpus_input)
    elif type(corpus_input) == tuple and len(corpus_input) == 2:
        corpus_name = corpus_input[0]
    else:
        raise ValueError("Input in invalid format.")

    return corpus_name
