from abc import abstractclassmethod
from os.path import exists, join
from typing import List

import tokenizers
from tokenizers import Tokenizer as HugTokenizer
from tokenizers import models, trainers

from ..config import Config
from ..corpus.corpus import Corpus


class Tokenizer:
    @abstractclassmethod
    def __init__(self, corpus: Corpus) -> None:
        pass

    @abstractclassmethod
    def encode_batch(self, sentance: List[str]) -> List[tokenizers.Encoding]:
        pass

    @abstractclassmethod
    def encode(self, sentance: str) -> tokenizers.Encoding:
        pass

    @abstractclassmethod
    def tokens_to_sentance(self, tokens: List[int]) -> str:
        pass

    @abstractclassmethod
    def decode_batch(self, tokens: List[List[int]]) -> str:
        pass


class WordPiece(Tokenizer):
    def __init__(self, corpus: Corpus) -> None:
        tokenizer_path = join(
            Config.cache_dir, "tokenizers", f"{corpus.get_id()}_tokenizer.json"
        )
        if exists(tokenizer_path):
            self.tokenizer = HugTokenizer.from_file(tokenizer_path)
            return
        self.train(corpus, tokenizer_path)

    def train(self, corpus: Corpus, path_to_save: str):
        self.tokenizer = HugTokenizer(models.WordPiece(unk_token=Config.unk_token))
        file_path = join(Config.cache_dir, "tokenizers", f"{corpus.get_id()}.txt")
        corpus.save_to_disk(file_path)

        trainer = trainers.WordPieceTrainer(vocab_size=Config.vocab_size)
        self.tokenizer.train(trainer, [file_path])
        self.tokenizer.save(path_to_save)

    def encode(self, sentances: str) -> tokenizers.Encoding:
        return self.tokenizer.encode(sentances)

    def encode_batch(self, sentances: List[str]) -> List[tokenizers.Encoding]:
        return self.tokenizer.encode_batch(sentances)

    def tokens_to_sentance(self, tokens: List[int]) -> str:
        return self.tokenizer.decode(tokens)

    def decode_batch(self, tokens: List[List[int]]) -> str:
        return self.tokenizer.decode_batch(tokens)
