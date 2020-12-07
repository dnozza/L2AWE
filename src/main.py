import pandas as pd
import utils_word_embeddings as utils
import arff
from pathlib import Path

path_input = "../data/data_example.tsv"
path_we = "/Volumes/GoogleDrive/Il mio Drive/Post-Doc MIND/workspaceNER/LearningToAdapt/data/raw/GoogleNews-vectors-negative300.bin.gz"
path_output = "../output/"
target_label_column = "Target-Label"
entity_column = "entity"
id_column = "id_text"
n_distribution_column = 10



Path(path_output).mkdir(parents=True, exist_ok=True)

## import data

data = pd.read_csv(path_input, sep = "\t")
distribution_columns = list(data.columns[-n_distribution_column:])
data_distribution = data[[id_column,entity_column]+distribution_columns]

##
if (path_we != ""):
    print("Computing Word Embedding representation")
    wv = utils.load_word_embeddings(path_we)
    embeddings_df = utils.extract_we_representation(data, entity_column, id_column, wv)
    data_distribution[id_column] = data_distribution[id_column].astype(int)
    embeddings_data = pd.merge(embeddings_df, data_distribution, on = [id_column,entity_column])

    embeddings_data[target_label_column] = data[target_label_column].astype('category')
    embeddings_data = embeddings_data.drop(columns=[id_column,entity_column])
else:
    print("Considering only probability distribution as input")
    embeddings_data = data[distribution_columns+[target_label_column]]


    # Transform labels into int
    #embeddings_data[target_label_column] = embeddings_data[target_label_column].cat.codes

# Save ARFF files

arff.dump(path_output + "temp.arff"
      , embeddings_data.values
      , relation='temp'
      , names=embeddings_data.columns)

## Substitute String with Nominal for the Attribute Class


class_values = ','.join(embeddings_data[target_label_column].unique())

with open(path_output + "temp.arff", "r") as prev:
    new_file_lines = []
    for line in prev:
        if target_label_column in line:
            line = "@attribute "+ target_label_column + " " + '{'+class_values+'}\n'
        elif not('@attribute' in line) and (line != "") and not('@data' in line):
            line = line.replace("'","")
        new_file_lines.append(line)

with open(path_output + "temp.arff", "w") as after:
        after.writelines(new_file_lines)



###

arff_writer = arff.Writer(path_output, relation='temp')#, header_names=list(embeddings_data.columns))
arff_writer.pytypes[type(myZipCodeObject)] = '{'+class_values+'}'
arff_writer.write([arff.nominal(target_label_column)])



class ZipCode(str):
    """Use this class to wrap strings which are intended to be nominals
    and shouldn't have enclosing quote signs."""
    def __repr__(self):
        return self

myZipCodeObject = ZipCode()