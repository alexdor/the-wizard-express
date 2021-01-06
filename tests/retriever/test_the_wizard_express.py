from the_wizard_express.corpus import TriviaQA
from the_wizard_express.retriever import TFIDFRetriever
from the_wizard_express.tokenizer import WordTokenizer


def test_tf_idf_on_trivia_dataset():
    triviaQA = TriviaQA()
    retriever = TFIDFRetriever(corpus=triviaQA, tokenizer=WordTokenizer(triviaQA))

    test_data = triviaQA.get_test_data()
    print(test_data)
    current_question = test_data[0]
    docs = retriever.retrieve_docs(current_question, 5)

    retrieved_proper_doc = 0
    if current_question.context in docs:
        retrieved_proper_doc += 1

    print(retrieved_proper_doc)
