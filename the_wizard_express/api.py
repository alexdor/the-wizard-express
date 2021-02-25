from typing import Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from the_wizard_express.config import Config
from the_wizard_express.corpus import Squad, TriviaQA
from the_wizard_express.reader import BertOnBertReader
from the_wizard_express.retriever import PyseriniSimple, TFIDFRetriever
from the_wizard_express.tokenizer import WordTokenizer
from transformers import AutoTokenizer

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Quesion(BaseModel):
    question: str
    use_trivia: Optional[bool] = False
    use_tf_idf: Optional[bool] = False
    return_supported_docs: Optional[bool] = False
    headers: Any


@app.get("/")
async def root():
    return {"message": "Hello World"}


gpu = "7"

reader = BertOnBertReader(
    tokenizer=AutoTokenizer.from_pretrained(
        BertOnBertReader.model_name,
        use_fast=True,
        cache_dir=Config.hugging_face_cache_dir,
    ),
    device=f"cuda:{gpu}" if gpu else "cpu",
)

squad = Squad()
trivia = TriviaQA()

retriever_pyserini = PyseriniSimple(squad, None)
retriever_pyserini_trivia = PyseriniSimple(trivia, None)

retriever_tfidf = TFIDFRetriever(squad, WordTokenizer(squad))
retriever_tfidf_trivia = TFIDFRetriever(trivia, WordTokenizer(trivia))


def get_retriever(trivia, tfidf):
    if tfidf:
        if trivia:
            return retriever_tfidf_trivia
        return retriever_tfidf

    if trivia:
        return retriever_pyserini_trivia
    return retriever_pyserini


@app.post("/answer")
async def index(request: Quesion):
    ans = {}
    retriever = get_retriever(request.use_trivia, request.use_tf_idf)
    docs = retriever.retrieve_docs(request.question, 5)

    if request.return_supported_docs:
        docs = tuple(docs)
        ans["docs"] = [{"content": doc[:200], "total_length": len(doc)} for doc in docs]

    ans["answer"] = reader.answer(request.question, docs)

    return ans
