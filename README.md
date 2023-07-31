# keyness-calculator
This module allows you to analyse the keyness of items in a study corpus compared to a reference corpus. The keyness calculator takes 3-tuples consisting of a token, part-of-speech tag and lemma as input, meaning that you need to have your corpora tokenised, part-of-speech tagged an lemmatised beforehand. The tuples can be introduced into the keyness calculator as CSV or TSV files (one line per tuple; one file = one corpus document; one folder of files = one subcorpus; one folder of subcorpora = one corpus), or they can also be organised in a Python dictionary (keys = subcorpora; values = list of lists \[one list = one corpus document] of tuples) and directly passed into the calculator. Below you can find a concrete usage example for both input types and an overview of the main steps performed by the underlying script.


**NOTE**: the example corpus used for the CSV/TSV input type is included in the <code>exampleCorpora</code> folder of this GitHub repository. This dummy corpus was created based on the [UD Spanish AnCora treebank](https://universaldependencies.org/treebanks/es_ancora/index.html). The treebank sentences were randomly divided over six documents, which were, at their turn, equally divided over three subcorpora (one subcorpus for the study corpus, and two for the reference corpus). The corpus adheres to the required folder structure: <code>corpus_folder/subcorpus_folders/document_files</code>.

## Usage example
### Input
The usage example is presented in the <code>keynessCalculator_example.py</code> file. It contains a usage example for both input types (CSV/TSV files or Python dictionary). The <code>init_keyness_calculator</code> function used to perform the keyness calculations only requires two arguments, namely the study corpus (passed to the first-position <code>input_sc</code> argument) and the reference corpus (passed to the second-position <code>input_rc</code> argument). For CSV/TSV files as input type, the argument is simply the path to the corpus folder; for the Python dictionary as input, you need to construct a 2-tuple of the corpus name followed by the Python dictionary in second position. To learn more about all the possible other arguments which can be passed to the <code>init_keyness_calculator</code> function, have a look at the [source code](https://github.com/JasperD-UGent/keyness-calculator/blob/main/keynessCalculator_example_defs.py).
```python
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
```

### Output
The output of intermediary steps (frequency dictionaries \[per item and totals] and dispersion values) are saved per corpus into an automatically created <code>prep</code> folder. The final results are stored in the automatically created <code>output</code> folder, in a subdirectory named <code>[study_corpus]\_VS_[reference_corpus]</code>. Four output files are created:
- An Excel file containing three sheets:
  - "all", in which the values for each item are visualised
  - "top-N", in which the results for the top-N CKIs (the value for N can be changed in the <code>number_ckis_want_analyse</code> argument) are presented
  - "selection", in which the results for the custom selection of items (which can be passed to the function through the <code>selection_items</code> argument) are presented
- The content of those three Excel sheets in three separate JSON files

**NOTE**: the <code>init_keyness_calculator</code> function also returns those three types of results as a Python dictionary (keys: "all", "top-N" and "selection").

## Method
### Step_1
1. Convert corpora into frequency dictionaries (per item and totals).
2. Store this intermediate output in <code>prep</code> folder.

### Step_2
1. Apply dispersion metric (DPnorm; Gries, 2008; Lijffijt & Gries, 2012), calculate adjusted frequencies and update frequency dictionaries. The dispersion values are based on the frequency distribution across subcorpora. If the <code>maintain_subcorpora</code> argument is set to <code>True</code> (which is the default value), the formula takes the original subcorpora as input (which means that if there is only one subcorpus, the adjusted frequencies will be equal to the absolute ones). However, if the value is set to <code>False</code>, all documents in all subcorpora (also if there is only one subcorpus) are randomly assigned to new subcorpora (the number of subcorpora is calculated by dividing the total number of documents in the corpus by the the value passed to the <code>divide_number_docs_by</code> argument, which defaults to 10).
2. Store this intermediate output in <code>prep</code> folder.

### Step_3
1. Calculate the keyness values. Parameters such as the type of frequencies used to perform the calculations (absolute or adjusted, with or without Laplace smoothing) and the keyness threshold can all be passed to the <code>init_keyness_calculator</code> function as additional keyword arguments. As for the type of metric, the five following methods are offered:
- DIFF (Gabrielatos & Marchi, 2011);
- Ratio (Kilgarriff, 2009);
- OddsRatio (Everitt, 2002; Pojanapunya & Watson Todd, 2016);
- LogRatio (Hardie, 2014);
- DiffCoefficient (Hofland & Johansson, 1982).
2. Store the results of the keyness analysis in the <code>output</code> folder.

### Step_4
1. Construct meta file containing the information of the last query.
2. Save this meta file into the <code>prep</code> folder (when the keyness calculator is initialised, it first checks this meta file, and when the query criteria are identical, the calculator will immediately load the intermediate output for the corpus in question in the <code>prep</code> folder, instead of again calculating the frequencies from scratch).

## Required Python modules
The keyness calculator uses the Python modules mentioned below, so you need to have them installed for the script to work.
- [numpy](https://pypi.org/project/numpy/) (~=1.18.2)
- [scikit-learn](https://pypi.org/project/scikit-learn/) (~=0.24.1)
- [Xlsxwriter](https://pypi.org/project/XlsxWriter/) (~=1.2.8)

## References
- Everitt, B.S. (2002). The Cambridge Dictionary of Statistics (2nd ed.). Cambridge University Press
- Gabrielatos, C. (2018). Keyness Analysis: nature, metrics and techniques. In C. Taylor & A. Marchi (Eds.), Corpus Approaches to Discourse: A Critical Review. Routledge.
- Gabrielatos, C., & Marchi, A. (2011). Keyness Matching metrics to definitions. November, 1–28.
- Gries, S. T. (2008). Dispersions and adjusted frequencies in corpora. International Journal of Corpus Linguistics, 13(4), 403–437. https://doi.org/10.1075/ijcl.13.4.02gri
- Hardie, A. (2014). Log Ratio - an informal introduction. http://cass.lancs.ac.uk/log-ratio-an-informal-introduction/
- Hofland, K., & Johansson, S. (1982). Word Frequencies in British and American English. Longman.
- Kilgarriff, A. (2009). Simple maths for keywords. In M. Mahlberg, V. González-Díaz & C. Smith (Eds.), Proceedings of the Corpus Linguistics Conference, CL2009. University of Liverpool
- Lijffijt, J., & Gries, S. T. (2012). Review of ((2008)): International Journal of Corpus Linguistics. International Journal of Corpus Linguistics, 17(1), 147–149. https://doi.org/10.1075/ijcl.17.1.08lij
- Pojanapunya, P., & Watson Todd, R. (2016). Log-likelihood and odds ratio: Keyness statistics for different purposes of keyword analysis. Corpus Linguistics and Linguistic Theory.
- Wilson, A. (2013). Embracing Bayes factors for key item analysis in corpus linguistics. In M. Bieswanger & A. Koll-Stobbe (Eds.), New Approaches to the Study of Linguistic Variability (pp. 3–11). Peter Lang.
