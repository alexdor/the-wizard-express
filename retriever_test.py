import gc
from datetime import datetime, timedelta
from functools import partial
from json import dumps
from multiprocessing import Pool
from timeit import default_timer as timer

from the_wizard_express.config import Config
from the_wizard_express.corpus import TriviaQA
from the_wizard_express.retriever import TFIDFRetriever
from the_wizard_express.tokenizer import WordTokenizer
from tqdm import tqdm


def check_question(retriever, number_of_docs, current_question):
    return current_question["context"] in retriever.retrieve_docs(
        current_question["question"], number_of_docs
    )


def main():

    prep_time_start = timer()
    Config.vocab_size = 8000
    number_of_docs = 5
    data_to_run_on = "test_data"

    corpus = TriviaQA()
    retriever = TFIDFRetriever(corpus=corpus, tokenizer=WordTokenizer(corpus))
    data_to_function_call = {
        "test_data": corpus.get_test_data,
        "train_data": corpus.get_train_data,
        "validation_data": corpus.get_validation_data,
    }
    test_data = data_to_function_call[data_to_run_on]()

    retrieved_proper_doc = 0
    gc.collect()
    prep_time_end = timer()

    with Pool(Config.max_proc_to_use) as pool:
        with tqdm(total=len(test_data)) as pbar:
            for included in pool.imap_unordered(
                func=partial(check_question, retriever, number_of_docs),
                iterable=test_data,
                chunksize=50,
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
                        "vocab_size": Config.vocab_size,
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


if __name__ == "__main__":
    main()
