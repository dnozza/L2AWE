from gensim.models import KeyedVectors
import numpy as np
import pandas as pd
import string
import re
from tqdm import tqdm

def load_word_embeddings(path_we):
    wv = KeyedVectors.load_word2vec_format(path_we, binary=True)
    return wv

def get_embedding_word(word, wv):
    if (word in wv):
        return wv[word]
    else:
        if not (word.islower()):
            lower_emb = get_embedding_word(word.lower(),wv)
            if not (lower_emb is None):
                return lower_emb
            else:
                nopunct_word = word.translate(str.maketrans('', '', string.punctuation))
                if not (word == nopunct_word):
                    nopunct_emb = get_embedding_word(nopunct_word,wv)
                    if not (nopunct_emb is None):
                        return nopunct_emb
                    else:
                        return None
                else:
                    return None
        else:
            nopunct_word = word.translate(str.maketrans('', '', string.punctuation))
            if not (word == nopunct_word):
                nopunct_emb = get_embedding_word(nopunct_word,wv)
                if not (nopunct_emb is None):
                    return nopunct_emb
                else:
                    return None
            else:
                return None


def get_embedding_entity(entity, wv):
    if " " in entity:
        embs = []
        for token in entity.split(" "):
            notpunct_token = token.translate(str.maketrans('', '', string.punctuation))
            if not (token in string.punctuation) and not (token.isnumeric()) and not (notpunct_token.isnumeric()):
                embs.append(get_embedding_word(token, wv))
        embs_notnone = [e for e in embs if not (e is None)]
        if (len(embs_notnone) == 0):
            emb = None
        else:
            emb = np.mean(embs_notnone, axis=0)

        return emb
    else:
        emb = get_embedding_word(entity, wv)

        is_upper = (any(l.isupper() for l in entity[1:]))

        if (emb is None) and is_upper:
            entity_uppersplit = ' '.join(re.sub(r"([A-Z])", r" \1", entity).split())
            return get_embedding_entity(entity_uppersplit, wv)
        else:
            return emb


def extract_we_representation(data, entity_column, id_column, wv):
    # Extract word embedding representation for each entity. Entities which are not in the embeddings are printed

    embeddings = {}
    count_not_find = 0
    print("Extracting Word Embedding representation")
    for index, row in tqdm(data.iterrows(), total=data.shape[0]):
        e = row[entity_column]
        emb_e = get_embedding_entity(e, wv)
        if (emb_e is None):
            count_not_find = count_not_find + 1
        else:
            id_entity = str(row[id_column])
            if not (id_entity in embeddings):
                embeddings[id_entity] = {}
            embeddings[id_entity][e] = (emb_e)
    print("N. of not found entities: ", count_not_find)

    # Convert embedding array to pandas

    embeddings_df_dict = {}

    for i in embeddings.keys():
        embeddings_df_dict[i] = pd.DataFrame.from_dict(embeddings[i], orient="index")
        embeddings_df_dict[i][id_column] = i
        embeddings_df_dict[i] = embeddings_df_dict[i].reset_index()

    embeddings_df = pd.concat(embeddings_df_dict.values())
    embeddings_df = embeddings_df.rename(columns={"index": entity_column})
    embeddings_df[id_column] = embeddings_df[id_column].apply(lambda x: x.split("_")[0])
    embeddings_df[id_column] = embeddings_df[id_column].astype(int)

    return embeddings_df