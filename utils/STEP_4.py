from .process_JSONs import dump_json
import os
import sys
from typing import Tuple


def meta(
        corpus_name: str, desired_pos: Tuple, lem_or_tok: str, maintain_subcorpora: bool, div_n_docs_by: int
) -> None:
    """STEP_4: store information of last query in meta file (data stored per corpus in "prep" folder).
    :param corpus_name: name of the corpus
    :param desired_pos: tuple of UD tags which should be taken into account in the keyness calculations.
    :param lem_or_tok: defines whether to calculate frequencies on token or lemma level.
    :param maintain_subcorpora: when working with adjusted frequencies, boolean value which defines whether dispersion
        is based on existing subcorpora, or whether all documents are merged and randomly split into new subcorpora.
    :param div_n_docs_by: when working with adjusted frequencies, number by which the total number of documents
        is divided to arrive at the number of new randomly generated subcorpora.
    :return: `None`
    """
    d_meta_corpus = {
        "desired_pos": desired_pos,
        "lemma_or_token": lem_or_tok,
        "maintain_subcorpora": maintain_subcorpora,
        "divide_number_docs_by": div_n_docs_by
    }
    dump_json(os.path.join("prep", corpus_name), f"{corpus_name}_meta.json", d_meta_corpus)
