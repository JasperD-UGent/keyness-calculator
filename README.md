# keyness-calculator
This module allows you to analyse the keyness of items in a study corpus compared to a reference corpus. The keyness calculator takes comma- or tab-separated files as input, or you can also directly pass your corpus as a Python dictionary into the calculator. Below you can find a concrete usage example (the example corpus used for this example is also included in the GitHub repository) and an overview of the keyness calculation methodology, which consists of four main steps.
## Usage example
### Input
The file <code>keynessCalculator_example.py</code> contains a usage example for both input types (CSV/TSV files or Python dictionary). In the case of the CSV/TSV file, a dummy example corpus was created based on the [UD Spanish AnCora treebank](https://universaldependencies.org/treebanks/es_ancora/index.html). The treebank sentences were randomly divided over six documents, which were, at their turn, equally divided over three subcorpora (one subcorpora for the study corpus, and two for the reference corpus). All files and folders are gathered in the <code>exampleCorpus<code> folder, according to the required folder structure: <code>corpus_folder/subcorpus_folders/document_files</code>.
  
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
The keyness calculator uses the Python modules mentioned below, so you need to have them installed for the script to work.
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
