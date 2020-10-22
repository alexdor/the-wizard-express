"""Top-level package for The wizard express."""

__author__ = """Alexandros Dorodoulis, Amalia Matei, Jesper Lund Petersen"""
__email__ = ""
__version__ = "0.0.1"

__all__ = ["corpus", "datasets", "language_model", "qa", "reader", "retriever"]

from .datasets import initializeNQDataset, QADataset
