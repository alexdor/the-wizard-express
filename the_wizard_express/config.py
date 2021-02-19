from multiprocessing import cpu_count
from os import environ, getenv, path
from typing import List


class _Config:
    """
    A class with all the configuration used across the projects
    """

    def __init__(self):
        self.debug = False
        self.max_proc_to_use = min(cpu_count() - 1, 15)
        self.percent_of_data_to_keep = 1.0

    """Config.cache_dir The cache directory to store models and other temporary files"""
    cache_dir = getenv("CACHE_DIR", path.join(path.realpath("."), ".cache"))

    @property
    def debug(self) -> bool:
        return self._debug

    @debug.setter
    def debug(self, value: bool) -> None:
        self._debug = value

    @property
    def max_proc_to_use(self) -> int:
        return self._proc

    @max_proc_to_use.setter
    def max_proc_to_use(self, value: int) -> None:
        environ["RAYON_RS_NUM_CPUS"] = str(value)
        self._proc = value

    @property
    def percent_of_data_to_keep(self) -> float:
        return self._percent_of_data_to_keep

    @percent_of_data_to_keep.setter
    def percent_of_data_to_keep(self, value: float) -> None:
        value = float(value)
        if value > 1.0:
            value = value / 100.0
        self._percent_of_data_to_keep = value

    vocab_size = 8000
    unk_token = "[UNK]"
    sep_token = "[SEP]"
    pad_token = "[PAD]"
    cls_token = "[CLS]"
    mask_token = "[MASK]"

    @property
    def special_tokens_list(self) -> List[str]:
        return [
            self.unk_token,
            self.sep_token,
            self.cls_token,
            self.pad_token,
            self.mask_token,
        ]


Config: _Config = _Config()
