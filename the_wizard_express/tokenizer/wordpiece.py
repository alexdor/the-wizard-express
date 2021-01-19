from os.path import join
from typing import Dict, List

from tokenizers import Tokenizer as HugTokenizer
from tokenizers import models, trainers

from ..config import Config
from ..corpus.corpus import Corpus
from ..tokenizer import Tokenizer


class WordPiece(Tokenizer):
    def _build(self, corpus: Corpus, path_to_save: str) -> None:
        self.tokenizer = HugTokenizer(models.WordPiece(unk_token=Config.unk_token))
        file_path = join(Config.cache_dir, "tokenizers", f"{corpus.get_id()}.txt")
        corpus.save_to_disk(file_path)

        trainer = trainers.WordPieceTrainer(vocab_size=Config.vocab_size)
        self.tokenizer.train(trainer, [file_path])
        self.tokenizer.save(path_to_save)
