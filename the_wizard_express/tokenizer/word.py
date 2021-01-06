from collections import Counter
from multiprocessing import Pool
from os.path import dirname, join, lexists
from pathlib import Path
from pickle import load
from typing import Dict, Optional, Union

from nltk import data, download
from nltk.corpus import stopwords
from nltk.tokenize import sent_tokenize, word_tokenize
from tokenizers import AddedToken
from tokenizers import Tokenizer as HuggingFaceTokenizer
from tokenizers import processors
from tokenizers.implementations import BaseTokenizer
from tokenizers.models import WordLevel
from tokenizers.normalizers import Lowercase, Sequence, unicode_normalizer_from_str
from tokenizers.pre_tokenizers import WhitespaceSplit
from tqdm import tqdm

from the_wizard_express.tokenizer.tokenizer import Tokenizer
from the_wizard_express.utils import generate_cache_path
from the_wizard_express.utils.utils import pickle_and_save_to_file

from ..config import Config
from ..corpus.corpus import Corpus

nltk_data_path = join(Config.cache_dir, "nltk")
data.path.append(nltk_data_path)


class WordTokenizer(Tokenizer):
    def _build(self, corpus: Corpus, path_to_save: str) -> None:
        vocab_path = generate_cache_path(
            "vocab", self, corpus, skip_vocab_size=True, file_ending=".pickle"
        )
        vocab = (
            Counter(load(open(vocab_path, "rb"))).most_common(Config.vocab_size)
            if lexists(vocab_path)
            else self._build_vocab(corpus, vocab_path)
        )

        vocab = {value: i for (i, (value, _)) in enumerate(vocab)}
        vocab.update(
            {
                value: i + len(vocab)
                for (i, value) in enumerate(Config.special_tokens_list)
            }
        )
        self.tokenizer = WordLevelBertTokenizer(vocab)

    def _build_vocab(self, corpus: Corpus, vocab_path: str):
        download("punkt", download_dir=nltk_data_path)
        download("stopwords", download_dir=nltk_data_path)
        cor = corpus.get_corpus()
        c = Counter()
        batch_size = 100
        with tqdm(total=len(cor)) as pbar:
            with Pool(Config.max_proc_to_use) as pool:
                for res in pool.imap_unordered(
                    func=WordTokenizer._prep_vocab,
                    iterable=(
                        cor[i : i + batch_size] for i in range(0, len(cor), batch_size)
                    ),
                ):
                    pbar.update(batch_size)
                    c += res
        Path(dirname(vocab_path)).mkdir(parents=True, exist_ok=True)
        pickle_and_save_to_file(c, vocab_path)
        return c.most_common(Config.vocab_size)

    @staticmethod
    def _prep_vocab(sentance_list) -> Counter:
        stop_words = set(stopwords.words("english"))

        # We are droping the stop words and everything that isn't a string
        # we might need to reconsider the numbers if we increase our vocab
        return Counter(
            (
                token
                for sentance in sentance_list
                for sentance in sent_tokenize(sentance.lower())
                for token in word_tokenize(sentance)
                if token.isalpha() and token not in stop_words
            )
        )


class WordLevelBertTokenizer(BaseTokenizer):
    """WordLevelBertTokenizer
    Represents a simple word level tokenization for BERT.
    """

    def __init__(
        self,
        vocab_file: Optional[Union[str, Dict[str, int]]] = None,
        unk_token: Union[str, AddedToken] = Config.unk_token,
        sep_token: Union[str, AddedToken] = Config.sep_token,
        cls_token: Union[str, AddedToken] = Config.cls_token,
        pad_token: Union[str, AddedToken] = Config.pad_token,
        mask_token: Union[str, AddedToken] = Config.mask_token,
        lowercase: bool = True,
        unicode_normalizer: Optional[str] = None,
    ):
        if vocab_file is not None:
            tokenizer = HuggingFaceTokenizer(WordLevel(vocab_file, unk_token=unk_token))
        else:
            # tokenizer = HuggingFaceTokenizer(WordLevel())
            raise TypeError("WordLevelBert requires a vocab file for now")

        # Let the tokenizer know about special tokens if they are part of the vocab
        if tokenizer.token_to_id(str(unk_token)) is not None:
            tokenizer.add_special_tokens([str(unk_token)])
        if tokenizer.token_to_id(str(sep_token)) is not None:
            tokenizer.add_special_tokens([str(sep_token)])
        if tokenizer.token_to_id(str(cls_token)) is not None:
            tokenizer.add_special_tokens([str(cls_token)])
        if tokenizer.token_to_id(str(pad_token)) is not None:
            tokenizer.add_special_tokens([str(pad_token)])
        if tokenizer.token_to_id(str(mask_token)) is not None:
            tokenizer.add_special_tokens([str(mask_token)])

        # Check for Unicode normalization first (before everything else)
        normalizers = []

        if unicode_normalizer:
            normalizers += [unicode_normalizer_from_str(unicode_normalizer)]

        if lowercase:
            normalizers += [Lowercase()]

        # Create the normalizer structure
        if len(normalizers) > 0:
            if len(normalizers) > 1:
                tokenizer.normalizer = Sequence(normalizers)
            else:
                tokenizer.normalizer = normalizers[0]

        tokenizer.pre_tokenizer = WhitespaceSplit()

        if vocab_file is not None:
            sep_token_id = tokenizer.token_to_id(str(sep_token))
            if sep_token_id is None:
                raise TypeError("sep_token not found in the vocabulary")
            cls_token_id = tokenizer.token_to_id(str(cls_token))
            if cls_token_id is None:
                raise TypeError("cls_token not found in the vocabulary")

            tokenizer.post_processor = processors.BertProcessing(
                (str(sep_token), sep_token_id), (str(cls_token), cls_token_id)
            )

        parameters = {
            "model": "WordLevel",
            "unk_token": unk_token,
            "sep_token": sep_token,
            "cls_token": cls_token,
            "pad_token": pad_token,
            "mask_token": mask_token,
            "lowercase": lowercase,
            "unicode_normalizer": unicode_normalizer,
        }

        super().__init__(tokenizer, parameters)

    @staticmethod
    def from_file(vocab: str, **kwargs):
        vocab = WordLevel.from_file(vocab)
        return WordLevelBertTokenizer(vocab, **kwargs)