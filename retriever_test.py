from datetime import datetime, timedelta
from json import dumps
from multiprocessing import Pool
from timeit import default_timer as timer

from the_wizard_express.config import Config
from the_wizard_express.corpus import TriviaQA
from the_wizard_express.retriever import TFIDFRetriever
from the_wizard_express.tokenizer import WordTokenizer
from tqdm import tqdm


def main():

    prep_time_start = timer()

    corpus = TriviaQA()
    retriever = TFIDFRetriever(corpus=corpus, tokenizer=WordTokenizer(corpus))

    retrieved_proper_doc = 0
    test_data = corpus.get_test_data()
    chunksize = 100

    def check_question(current_question):
        return current_question["context"] in retriever.retrieve_docs(
            current_question["question"], 5
        )

    prep_time_end = timer()

    with Pool(Config.max_proc_to_use) as pool:
        with tqdm(total=len(test_data)) as pbar:
            for included in pool.imap_unordered(
                func=check_question,
                iterable=test_data,
                chunksize=chunksize,
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
                sort_keys=True,
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
