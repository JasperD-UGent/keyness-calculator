# keyness-calculator
<p>This module allows you to analyse the keyness of items in a study corpus compared to a reference corpus. The keyness calculator takes 3-tuples consisting of the token, part-of-speech tag and lemma as input, meaning that you need to have your corpora tokenised, part-of-speech tagged an lemmatised beforehand. The tuples can be introduced into the keyness calculator as CSV or TSV files (one line per tuple; one file = one corpus document; one folder of files = one subcorpus; one folder of subcorpora = one corpus), or they can also be organised in a Python dictionary (keys = subcorpora; values = list of lists one list = one corpus document of tuples) and directly passed into the calculator. Below you can find a concrete usage example for both input types and an overview of the keyness calculation methodology, which consists of four main steps.</p>
<p>**NOTE**: the example corpus used for the CSV/TSV input type is included in the <code>exampleCorpus</code> folder of this GitHub repository. This dummy corpus was created based on the [UD Spanish AnCora treebank](https://universaldependencies.org/treebanks/es_ancora/index.html). The treebank sentences were randomly divided over six documents, which were, at their turn, equally divided over three subcorpora (one subcorpora for the study corpus, and two for the reference corpus). The corpus adheres to the required folder structure: <code>corpus_folder/subcorpus_folders/document_files</code>.</p>
## Usage example
### Input
<p>The usage example is presented in the <code>keynessCalculator_example.py</code> file. It contains a usage example for both input types (CSV/TSV files or Python dictionary). It only requires two arguments, namely the study corpus (passed to the first-position <code>input_sc</code> argument) and the reference corpus (passed to the second-position <code>input_rc</code> argument). For CSV/TSV files as input type, the argument is simply the path to the corpus folder; for the Python dictionary as input, you need to construct a 2-tuple of the corpus name followed by the Python dictionary in second position. To learn more about all the possible other arguments which can be passed to the <code>init_keyness_calculator</code> function, have a look at the [source code](https://github.com/JasperD-UGent/keyness-calculator/blob/main/utils.py).</p>

### Output
$
## Keyness calculation methodology
### Step_1
$
### Step_2
$
### Step_3
$
### Step_4
$
## Required Python modules
<p>The keyness calculator uses the Python modules mentioned below, so you need to have them installed for the script to work.</p>
- [numpy](https://pypi.org/project/numpy/)
- [scikit-learn](https://pypi.org/project/scikit-learn/)
- [Xlsxwriter](https://pypi.org/project/XlsxWriter/)
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
