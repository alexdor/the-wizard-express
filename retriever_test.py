from the_wizard_express.corpus import TriviaQA
from the_wizard_express.retriever import TFIDFRetriever
from the_wizard_express.tokenizer import WordTokenizer
from tqdm import tqdm

triviaQA = TriviaQA()
retriever = TFIDFRetriever(corpus=triviaQA, tokenizer=WordTokenizer(triviaQA))

retrieved_proper_doc = 0
test_data = triviaQA.get_test_data()

for current_question in tqdm(test_data):
    docs = retriever.retrieve_docs(current_question["question"], 5)

    if current_question["context"] in docs:
        retrieved_proper_doc += 1

print(
    f"Successfully retrieved {retrieved_proper_doc} out of {len(test_data)} documents"
)
