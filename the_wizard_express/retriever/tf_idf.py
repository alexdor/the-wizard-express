from collections import Counter
from math import log10
from multiprocessing import Pool
from os import remove
from os.path import lexists
from pickle import load
from typing import Counter as CounterType
from typing import Dict, Iterator, Tuple

from numpy import argpartition, empty, hstack, ndarray, zeros
from scipy.sparse import csc_matrix, csr_matrix, load_npz, save_npz
from tokenizers import Encoding
from tqdm import tqdm

from ..config import Config
from ..utils import pickle_and_save_to_file
from . import Retriever


class TFIDFRetriever(Retriever):
    __slots__ = ("tokenizer", "corpus", "tf_idf", "retriever_path")
    friendly_name = "tfidf"
    _file_ending = ".npz"

    def retrieve_docs(self, question: str, number_of_docs: int) -> Tuple[str]:
        encoded_question = self.tokenizer.encode(question)

        # Get ids for special tokens
        special_token = {
            self.tokenizer.vocab[special_token]
            for special_token in Config.special_tokens_list
        }

        # Drop extra tokens (sep, unknown etc)
        encoded_question = [
            id for id in encoded_question.ids if id not in special_token
        ]

        # Get the tf-idf for the encoded tokens and sum it per row
        sum_tf_idf = self.tf_idf[:, encoded_question].sum(axis=1)

        # get the documents with the top sum
        doc_indexes = argpartition(sum_tf_idf, -number_of_docs, axis=0)[
            -number_of_docs:
        ]

        # return the corpuses with the proper index
        return tuple(
            doc.as_py()
            for doc in self.corpus.get_docs_by_index(doc_indexes.flatten().tolist()[0])
        )

    def _load_from_file(self) -> None:
        self.tf_idf = load_npz(self.retriever_path)

    def _build(self) -> None:
        vocab = self.tokenizer.vocab

        encoded_special_tokens = {vocab[token] for token in Config.special_tokens_list}
        vocab_values = tuple(
            value for value in vocab.values() if value not in encoded_special_tokens
        )

        vocab_length = len(vocab_values)
        corpus_length = len(self.corpus.corpus)
        tf = empty((vocab_length, 0))
        batch_size = 250
        documents_with_word: CounterType[int] = Counter()

        corpus_iterator = self._get_iterator_for_corpus(
            vocab_values, batch_size=batch_size
        )

        # Compute tf-idf for the full corpus
        with Pool(Config.max_proc_to_use) as pool:
            first_loop_cache = f"{self.retriever_path}_tf_first_loop"
            # compute tf and word document frequency
            if lexists(first_loop_cache):
                if Config.debug:
                    print("Loading tf and document counter from cache")
                (tf, documents_with_word) = load(open(first_loop_cache, "rb"))
            else:
                if Config.debug:
                    print("Computing tf and document counter")

                with tqdm(total=corpus_length) as pbar:
                    for (tf_res, partial_documents_with_word) in pool.imap(
                        func=TFIDFRetriever._create_tf_and_document_counter,
                        iterable=corpus_iterator,
                    ):
                        documents_with_word += partial_documents_with_word
                        tf = hstack((tf, tf_res))
                        pbar.update(batch_size)

                del tf_res, partial_documents_with_word
                # this matrix is iterated mainly row wise, that's why csr is used
                tf = csr_matrix(tf)

                # temporary store tf and documents_with_word to disk
                try:
                    pickle_and_save_to_file((tf, documents_with_word), first_loop_cache)
                except Exception as e:
                    print("Failed to save first part of tf_idf to disk, error:", e)

            # Init tf-idf matrix and create iterator for computing it
            tf_idf = empty((corpus_length, vocab_length))
            tf_iterator = (
                (doc_counter, tf[doc_counter[0]], corpus_length)
                for doc_counter in documents_with_word.most_common()
            )

            if Config.debug:
                print("Computing tf-idf sparse matrix")

            # compute tf-idf
            with tqdm(total=len(documents_with_word)) as pbar:
                for (word_tf_idf, word_id) in pool.imap_unordered(
                    func=TFIDFRetriever._compute_tf_idf,
                    iterable=tf_iterator,
                    chunksize=50,
                ):
                    tf_idf[:, word_id] = word_tf_idf.todense()
                    pbar.update()
        del tf, documents_with_word, word_tf_idf

        # this matrix is iterated mainly column wise, that's why csc is used
        tf_idf = csc_matrix(tf_idf)
        save_npz(self.retriever_path, tf_idf)
        self.tf_idf = tf_idf
        remove(first_loop_cache)
        if Config.debug:
            print("Building tf-idf done")

    @staticmethod
    def _create_tf_and_document_counter(
        params: Tuple[Tuple[Encoding, ...], Tuple[int, ...]]
    ) -> Tuple[ndarray, CounterType[int]]:
        (encoded_corpus, vocab_values) = params
        document_counter: Dict[int, int] = {}
        tf = zeros((len(vocab_values), len(encoded_corpus)))
        for corpus_index, doc in enumerate(encoded_corpus):
            counter: CounterType[int] = Counter(doc.ids)
            for word_id in vocab_values:
                if word_id in counter:
                    document_counter[word_id] = document_counter.get(word_id, 0) + 1
                tf[word_id, corpus_index] = counter[word_id] / float(len(doc))

        return (tf, Counter(document_counter))

    @staticmethod
    def _compute_tf_idf(
        params: Tuple[Tuple[int, int], ndarray, int]
    ) -> Tuple[ndarray, int]:
        ((word_id, document_count), tf, corpus_length) = params

        if document_count == 0:
            return (zeros((corpus_length)), word_id)

        idf = log10(corpus_length / float(document_count) + 1)
        return (idf * tf, word_id)

    def _get_iterator_for_corpus(
        self,
        vocab_values,
        batch_size=100,
    ) -> Iterator[Tuple[Tuple[Encoding, ...], Tuple[int, ...]]]:
        for current_batch in self.corpus.corpus_iterator(batch_size):
            yield (
                self.tokenizer.encode_batch(current_batch),
                vocab_values,
            )
