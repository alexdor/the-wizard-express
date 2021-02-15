import gc
from datetime import datetime, timedelta
from functools import partial
from json import dumps
from multiprocessing import Pool
from timeit import default_timer as timer

from the_wizard_express.config import Config
from the_wizard_express.corpus import Squad, TriviaQA
from the_wizard_express.retriever import TFIDFRetriever
from the_wizard_express.tokenizer import (
    WordTokenizer,
    WordTokenizerWithoutStopWords,
    WordTokenizerWithoutStopWordsAndNotAlpha,
    WordTokenizerWithSimpleSplit,
)
from tqdm import tqdm


def check_question(retriever, number_of_docs, current_question):
    return current_question["context"] in retriever.retrieve_docs(
        current_question["question"], number_of_docs
    )


def run_retriever_test(
    corpus_class,
    retriever_class,
    tokenizer_class,
    number_of_docs=5,
    data_to_run_on="test_data",
):
    prep_time_start = timer()
    corpus = corpus_class()
    retriever = retriever_class(corpus=corpus, tokenizer=tokenizer_class(corpus))
    data_to_function_call = {
        "test_data": corpus.get_test_data,
        "train_data": corpus.get_train_data,
        "validation_data": corpus.get_validation_data,
    }
    test_data = data_to_function_call[data_to_run_on]()

    gc.collect()

    retrieved_proper_doc = 0
    prep_time_end = timer()

    with Pool(Config.max_proc_to_use) as pool:
        with tqdm(total=len(test_data)) as pbar:
            for included in pool.imap_unordered(
                func=partial(check_question, retriever, number_of_docs),
                iterable=test_data,
                chunksize=100,
            ):
                if included:
                    retrieved_proper_doc += 1
                pbar.update()
    calculation_finished = timer()
    results = f"Successfully retrieved {retrieved_proper_doc} out of {len(test_data)} documents"
    print(results)

    with open("./results_for_paper/retriever_results.txt", "a+") as file:
        file.write(
            dumps(
                {
                    "date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
                    "pretty_results": results,
                    "correctly_retrieved_documents": retrieved_proper_doc,
                    "total_documents": len(test_data),
                    "number_of_docs_to_retrieve": number_of_docs,
                    "data_it_run_on": data_to_run_on,
                    "correctly_retrieved": f"{(100 * retrieved_proper_doc) / len(test_data)} %",
                    "extra_info": {
                        "corpus": corpus.friendly_name,
                        "retriever": retriever.friendly_name,
                        "tokenizer": retriever.tokenizer.friendly_name,
                        "max_vocab_size": Config.vocab_size,
                        "actual_vocab_size": len(retriever.tokenizer.vocab)
                        - len(Config.special_tokens_list),
                        "preperation_time": str(
                            timedelta(seconds=prep_time_end - prep_time_start)
                        ),
                        "calculation_time": str(
                            timedelta(seconds=calculation_finished - prep_time_end)
                        ),
                        "total_time": str(
                            timedelta(seconds=calculation_finished - prep_time_start)
                        ),
                    },
                },
                indent=2,
            )
        )


def main():
    Config.debug = True
    corpuses = (TriviaQA, Squad)
    retrievers = [TFIDFRetriever]
    tokenizers = (
        WordTokenizerWithSimpleSplit,
        WordTokenizerWithoutStopWords,
        WordTokenizer,
        WordTokenizerWithoutStopWordsAndNotAlpha,
    )
    vocab_sizes = [80000, 40000, 8000]

    def find_data_to_run_on(corpus):
        return (
            ["validation_data"]
            if corpus.friendly_name == "squad"
            else ["test_data", "validation_data"]
        )

    for corpus_class in corpuses:
        for vocab_size in vocab_sizes:
            Config.vocab_size = vocab_size
            for retriever_class in retrievers:
                for tokenizer_class in tokenizers:
                    for data_to_run_on in find_data_to_run_on(corpus_class):
                        for number_of_docs in [1, 3, 5]:
                            pri = f"{retriever_class.friendly_name} with {tokenizer_class.friendly_name} on {corpus_class.friendly_name}, vocab size {vocab_size} and validated on {data_to_run_on}"
                            print(f"Started testing {pri}")
                            try:
                                run_retriever_test(
                                    corpus_class,
                                    retriever_class,
                                    tokenizer_class,
                                    data_to_run_on=data_to_run_on,
                                    number_of_docs=number_of_docs,
                                )
                            except Exception as e:
                                print(f"Got the following error on {pri}", e)
                                print(f"Retrying {pri}")
                                gc.collect()
                                run_retriever_test(
                                    corpus_class,
                                    retriever_class,
                                    tokenizer_class,
                                    data_to_run_on=data_to_run_on,
                                    number_of_docs=number_of_docs,
                                )
                            print(f"Finished testing {pri}")
                            print("------" * 4, "\n")
                            gc.collect()
        gc.collect()


if __name__ == "__main__":
    main()
