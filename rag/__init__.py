"""Retrieval-backed validation of recognized plate text.

Corrects OCR output against a knowledge base of real values. Retrieval is
lexical rather than dense — see :mod:`rag.similarity` for why — so this package
loads no model and needs no vector store.
"""
