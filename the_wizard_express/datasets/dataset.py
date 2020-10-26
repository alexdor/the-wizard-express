import torch
import os
import random
import json
from itertools import islice, count


class QADataset(torch.utils.data.Dataset):
    """Dataset"""

    _BLOCK_SIZE = 9000
    """
    Iterable data structure for storing data in blocks on the file system.
    """

    def __init__(
        self, directory, tokenizer, prefix="block", shuffle=True, num_blocks=None
    ):
        self.directory = directory
        self.tokenizer = tokenizer
        self.shuffle = shuffle
        # Potentially limit the number of blocks used
        paths = [path for path in os.listdir(directory) if prefix in path]
        paths = paths[:num_blocks] if num_blocks else paths
        self.block_paths = paths
        # Define the block storage and repeat them forever
        self.block_iter = self._get_block_iter()
        self.block_size = int(self.block_paths[0].split("-")[1])
        self.block_idx, self.line_idx = None, None

    def __iter__(self):
        return self

    def _get_block_iter(self):
        if self.shuffle:
            return iter(random.sample(self.block_paths, len(self.block_paths)))
        else:
            return iter(self.block_paths)

    def __len__(self):
        # True dataset size
        count = lambda f: (1 for _ in open(os.path.join(self.directory, f)))
        s = sum([sum(count(f)) for f in self.block_paths])
        return s

    def __next__(self):
        # No data has been loaded yet
        if self.block_idx is None:
            try:
                self.block_idx = next(self.block_iter)
            except StopIteration:
                self.block_iter = self._get_block_iter()
                self.block_idx = next(self.block_iter)
            with open(os.path.join(self.directory, self.block_idx)) as f:
                block = f.readlines()
            block = random.sample(block, len(block)) if self.shuffle else block
            self.line_idx = iter(block)
        try:
            item = next(self.line_idx)
        except StopIteration:
            # Request a new block
            self.block_idx = None
            item = self.__next__()
        parsedJson = json.loads(item)
        short_answer = []
        for annotation in parsedJson["annotations"]:
            if annotation["short_answers"] == []:
                return self.__next__()
            else:
                short_answer = annotation["short_answers"][0]
        encoded = self.tokenizer(
            parsedJson["document_text"],
            parsedJson["question_text"],
            truncation=True,
            padding=True,
            max_length=512,
        )
        return (
            torch.tensor([encoded["input_ids"]]),
            torch.tensor([encoded["attention_mask"]]),
            torch.tensor([short_answer["start_token"]]),
            torch.tensor([short_answer["end_token"]]),
        )

    @staticmethod
    def write(file, directory, prefix="block"):
        block_size = QADataset._BLOCK_SIZE
        with open(file) as data:
            block_size = min(QADataset._BLOCK_SIZE, sum(1 for _ in data))
        data = open(file, "r")
        if not os.path.exists(directory):
            os.makedirs(directory)
        iterable = iter(data)
        for idx in count():
            block = list(islice(iterable, block_size))
            if len(block) == 0:
                break
            path = os.path.join(
                directory, "{}-{}-{}.jsonl".format(prefix, QADataset._BLOCK_SIZE, idx)
            )
            with open(path, "w") as file:
                [file.write(item) for item in block if item is not None]

    # # Initialize dataset
    # def __init__(self, encodings):
    #     self.encodings = encodings

    # def __getitem__(self, idx):
    #     return {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}

    # def __len__(self):
    #     return len(self.encodings.input_ids)
