# L2AWE (LearningToAdapt with Word Embeddings)

While most of the existing Named Entity Recognition (NER) systems make use of generic entity type classification schemas, the comparison and integration of different entity types among different NER systems is a complex problem even for human experts.
**L2AWE** (Learning To Adapt with Word Embeddings) aims at adapting a NER system trained on a source classification schema to a given target one.

## 1 Usage

(lines starting with '$' denote command line input)

### Shell script:

`$python3 main.py -i <TSV input file> [options] `

### Options:
```
	--path_input, -i <STR>:   Path of input file which comprises text, entities and probability distribution over source classification schema.
  
	--path_output, -o <STR>:   Path of output files (prediction and performance). Default: '../output/'
  
	--path_we, -w <STR>:   Path of Word Embeddings files bi.gz file. Default: ''
  
	--target_label_column, -t <STR>:   Target Label Column in input file. Default: 'Target-Label'
  
	--entity_column, -e <STR>:   Entity Column in input file. Default: 'entity'
  
	--id_column, -d <STR>:   ID Column in input file. Default: 'id_text'
  
	--n_distribution_column, -n <INT>:   ID Column in input file. Default: 10
  
	--n_fold, -f <INT>:   Number of folds for Cross-Validation evaluation. Default: 3
  
```

## 2 Inputs

### Input File
The **input file** has to be a tab-separated file, where each line represents a different entity recognized in a text. The file should contain a header, a column with the ID of the text, the entity, the probability distribution over source classification schema, and the label in the target schema. See *data* folder for more details.

## 3 Outputs

L2AWE provides two output files:
* The **performance.csv** file which reports Accuracy, Precision, Recall, and F-measure of the adaptation model.
* The **prediction.csv** file which reports predictions for each entity.


## 4 Examples

`$python main.py -i "../data/data_example.tsv"`

Run L2AWE on the input example file with default parameters


`$python main.py -i "../data/data_example.tsv" -o "../output/" -f 3`

Run L2AWE on the input example file, save results in the output folder of a 3-fold cross-validation evaluation


`$python main.py -i "../data/data_example.tsv" -w "GoogleNews-vectors-negative300.bin.gz"`

Run L2AWE on the input example file and add the word embedding representation of input entities with GoogleNews Word2Vec embeddings (please note that embeddings should be downloaded in advance)


[![licensebuttons by-nc-sa](https://licensebuttons.net/l/by-nc-sa/3.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0)
