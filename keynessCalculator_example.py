from keynessCalculator_example_defs import init_keyness_calculator
import numpy as np
import os
import random
import sys


seed = 42
np.random.seed(seed)
random.seed(seed)


def main():
    # CSV/TSV files as input
    input_sc = os.path.join("exampleCorpora", "SC_singleSubc_1")
    input_rc = os.path.join("exampleCorpora", "RC_multSubc_1")
    keyness_dictionary_1 = init_keyness_calculator(input_sc, input_rc)
    
    # Python dictionary as input
    input_sc = ("SC_singleSubc_2", {
        "SC_subcorpus1": [[("tok1", "NOUN", "lem1"), ("tok2", "NOUN", "lem1")],
                          [("tok1", "NOUN", "lem1"), ("tok3", "VERB", "lem2")],
                          [("tok4", "NOUN", "lem3"), ("tok3", "VERB", "lem2")],
                          [("tok4", "NOUN", "lem3"), ("tok3", "VERB", "lem2")]]
    })
    input_rc = ("RC_multSubc_2", {
        "RC_subcorpus1": [[("tok1", "NOUN", "lem1"), ("tok2", "NOUN", "lem1")],
                          [("tok1", "NOUN", "lem1"), ("tok3", "VERB", "lem2")],
                          [("tok1", "NOUN", "lem1"), ("tok2", "NOUN", "lem1")],
                          [("tok1", "NOUN", "lem1"), ("tok3", "VERB", "lem2")]],
        "RC_subcorpus2": [[("tok1", "NOUN", "lem1"), ("tok2", "NOUN", "lem1")],
                          [("tok1", "NOUN", "lem1"), ("tok3", "VERB", "lem2")],
                          [("tok1", "NOUN", "lem1"), ("tok2", "NOUN", "lem1")],
                          [("tok1", "NOUN", "lem1"), ("tok3", "VERB", "lem2")],
                          [("tok5", "NOUN", "lem4"), ("tok5", "NOUN", "lem4")],
                          [("tok5", "NOUN", "lem4"), ("tok6", "VERB", "lem5")],
                          [("tok1", "NOUN", "lem1"), ("tok2", "NOUN", "lem1")],
                          [("tok1", "NOUN", "lem1"), ("tok3", "VERB", "lem2")]]
    })
    keyness_dictionary_2 = init_keyness_calculator(input_sc, input_rc)


if __name__ == "__main__":
    main()
