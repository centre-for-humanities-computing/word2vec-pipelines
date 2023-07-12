import glob
import json
from functools import partial

import spacy
from skpartial.pipeline import make_partial_pipeline

from skword2vec.model.word2vec import Word2VecTransformer
from skword2vec.preprocessing.spacy import Flattener, SpacyPreprocessor
from skword2vec.streams import chunk, pipe_streams, stream_files

# Let's imagine we have some jsonl files in the current directory
paths = glob.glob("./*.jsonl")

# And that all files are made up of entries that contain metadata and
# textual content for texts.
# Kinda like this:
"""
{"content": "Blablablabla...", "author": "Me and I"}
{"content": "Habadahabda...", "author": "You"}
...
"""

# We build a streaming pipeline:
stream_text_chunks = pipe_streams(
    # Streams lines (aka. records) from files.
    partial(stream_files, lines=True),
    # We parse them to json
    partial(map, json.loads),
    # Then we grab the "content" field
    partial(map, lambda record: record["content"]),
    # Then we take chunks of size 10_000
    # For each iteration of training
    partial(chunk, chunk_size=10_000),
)

# Then we build a machine learning pipeline

# We load a Spacy model for English
nlp = spacy.load("en_core_web_sm")
# We want all alphabetical tokens, that aren't stop words
patterns = [[{"IS_ALPHA": True, "IS_STOP": False}]]
preprocessor = SpacyPreprocessor(nlp, patterns=patterns, out_attribute="LEMMA")

# We create a Skip-gram model with 200 dimensions and window size of 10
model = Word2VecTransformer(n_components=200, window=10, algorithm="sg")

# Then we create a partial pipeline
# We need skpartial, since the regular pipeline
# Won't like partial_fit
pipeline = make_partial_pipeline(
    # Spacy preprocessor component
    preprocessor,
    # We need to flatten along the zeroth axis
    # Since the preprocessor returns lists of documents,
    # But we want list of sentences
    Flattener(axis=0),
    # The actual model
    model,
)

# Train the model on each chunk
for i_chunk, text_chunk in enumerate(stream_text_chunks(paths)):
    pipeline.partial_fit(text_chunk)
    # We can even save checkpoints
    model.save(f"models/checkpoint_chunk_{i_chunk}")
