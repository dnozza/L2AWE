import pandas as pd
import utils_word_embeddings as utils
import arff
from pathlib import Path
import subprocess
import argparse


def get_parser():
    """
    Creates a new argument parser.
    """
    parser = argparse.ArgumentParser('L2A')
    parser.add_argument('--path_input', '-i', action='store', type=str, help='Path Input.')
    parser.add_argument('--path_output', '-o', action='store', type=str, help='Path Output.', default='../output/')
    parser.add_argument('--path_we', '-w', action='store', type=str, help='Path Word Embeddings.', default='')

    parser.add_argument('--target_label_column', '-t', action='store', type=str, help='Target Label Column.',
                        default='Target-Label')
    parser.add_argument('--entity_column', '-e', action='store', type=str, help='Entity Column.',
                        default='entity')
    parser.add_argument('--id_column', '-d', action='store', type=str, help='ID Column.',
                        default='id_text')

    parser.add_argument('--n_distribution_column', '-n', type=int,
                        help='Number of columns that are associated to distribution', default=10)
    parser.add_argument('--n_fold', '-f', type=int,
                        help='Number of folds for Cross-Validation evaluation', default=3)
    return parser


def main(args=None):
    parser = get_parser()
    args = parser.parse_args(args)

    Path(args.path_output).mkdir(parents=True, exist_ok=True)

    ## Import TSV data

    data = pd.read_csv(args.path_input, sep="\t")
    distribution_columns = list(data.columns[-args.n_distribution_column:])
    data_distribution = data[[args.id_column, args.entity_column] + distribution_columns]

    ##
    if (args.path_we != ""):
        print("Computing Word Embedding representation")
        wv = utils.load_word_embeddings(args.path_we)
        embeddings_df = utils.extract_we_representation(data, args.entity_column, args.id_column, wv)
        data_distribution[args.id_column] = data_distribution[args.id_column].astype(int)
        embeddings_data = pd.merge(embeddings_df, data_distribution, on=[args.id_column, args.entity_column])

        embeddings_data[args.target_label_column] = data[args.target_label_column].astype('category')
        embeddings_data = embeddings_data.drop(columns=[args.id_column, args.entity_column])
    else:
        print("Considering only probability distribution as input")
        embeddings_data = data[distribution_columns + [args.target_label_column]]

    # Save ARFF files
    print("Save temporary ARFF file")

    arff.dump(args.path_output + "temp.arff"
              , embeddings_data.values
              , relation='temp'
              , names=embeddings_data.columns)

    # Substitute String with Nominal for the Attribute Class

    class_values = ','.join(embeddings_data[args.target_label_column].unique())

    with open(args.path_output + "temp.arff", "r") as prev:
        new_file_lines = []
        for line in prev:
            if args.target_label_column in line:
                line = "@attribute " + args.target_label_column + " " + '{' + class_values + '}\n'
            elif not ('@attribute' in line) and (line != "") and not ('@data' in line):
                line = line.replace("'", "")
            new_file_lines.append(line)

    with open(args.path_output + "temp.arff", "w") as after:
        after.writelines(new_file_lines)

    # Run WEKA Classification
    print("Run WEKA Classification")

    path_jar = '../src_java/L2A/target/LearningToMap-0.0.1-SNAPSHOT-jar-with-dependencies.jar'

    subprocess.call(['java', '-jar', path_jar, args.path_output + "temp.arff", args.path_output, str(args.n_fold)])


if __name__ == '__main__':
    main()
