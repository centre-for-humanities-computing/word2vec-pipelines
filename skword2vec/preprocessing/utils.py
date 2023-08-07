import awkward as ak
import numpy as np
from awkward.contents import NumpyArray


def create_string_array(documents: list[list[list[str]]]) -> ak.Array:
    tokens = []
    token_lengths = []
    sent_lengths = []
    doc_lengths = []
    for doc in documents:
        doc_lengths.append(len(doc))
        for sent in doc:
            sent_lengths.append(len(sent))
            for token in sent:
                tokens.append(token)
                token_lengths.append(len(token.encode("utf-8")))
    joint_unicode = "".join(tokens).encode("utf-8")
    buffer = np.frombuffer(joint_unicode, dtype=np.uint8)
    chars = NumpyArray(buffer, parameters={"__array__": "char"})
    tokens = ak.unflatten(chars, token_lengths)
    sents = ak.unflatten(tokens, sent_lengths)
    docs = ak.unflatten(sents, doc_lengths)
    return docs
