from json import dumps
from typing import Any, Dict, List


def find_vocab_words_by_indexes(vocab: Dict[Any, Any], indexes: List[int]):
    keys = list(vocab.keys())
    values = list(vocab.values())
    return [keys[values.index(i)] for i in indexes]


def retrieve_index_and_tf_idf_for_word(
    tf_idf_matrix, index: int, name: str, file_path: str
):
    documents = tf_idf_matrix[:, index].toarray()
    indexes = [i for i, value in enumerate(documents) if value and value[0]]
    tf_idf = [value[0] for value in documents if value and value[0]]

    with open(file_path, "w+") as file:
        file.write(
            dumps(
                {"name": name, "indexes": indexes, "tf_idf": tf_idf},
                sort_keys=True,
                indent=2,
            )
        )
    return indexes, tf_idf
