import subprocess
from os.path import dirname, join
from pathlib import Path
from shutil import rmtree
from typing import Iterator

from pyserini.search import SimpleSearcher
from simdjson import Parser

from ..config import Config
from .retriever import Retriever


class PyseriniSimple(Retriever):
    __slots__ = ("tokenizer", "corpus", "searcher", "retriever_path")
    friendly_name = "pyserini-simple"
    _file_ending = ".index"
    _skip_vocab_size = True

    def retrieve_docs(self, question: str, number_of_docs: int) -> Iterator[str]:
        # Retrieve relevant documents, turn their JSON result into a
        #  bytearray, parse it, extract the content, and return results
        return (
            str(self.json_parse.parse(hit.raw.encode())["contents"])
            for hit in self.searcher.search(question, k=number_of_docs)
        )

    def _load_from_file(self) -> None:
        self.json_parse = Parser()
        self.searcher = SimpleSearcher(self.retriever_path)

    def _build(self) -> None:
        tmp_path = f"{self.retriever_path}_tmp"
        Path(dirname(tmp_path)).mkdir(parents=True, exist_ok=True)
        self.corpus.save_to_disk(file_location=join(tmp_path, "doc.json"), as_json=True)
        subprocess.run(
            [
                "python",
                "-m",
                "pyserini.index",
                "-collection",
                "JsonCollection",
                "-generator",
                "DefaultLuceneDocumentGenerator",
                "-threads",
                str(Config.max_proc_to_use),
                "-input",
                tmp_path,
                "-index",
                self.retriever_path,
                "-storePositions",
                "-storeDocvectors",
                "-storeRaw",
            ],
            check=True,
        )
        rmtree(tmp_path)
        self._load_from_file()
