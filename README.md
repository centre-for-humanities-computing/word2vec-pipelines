# embedding-pipelines
Sklearn pipeline components for tokenizing and cleaning text, and then using these to train word/subword/document embeddings.
Everything in this repo is built- with messy, noisy, large datasets in mind that have to be streamed from disk.

## Overview

### Streams
This repo contains numerous utilities for streaming texts from disks and dealing with deeply nested iterable structures.
The  `.streams` module contains utilities for lazily streaming and chunking texts.
Here is an example of how you would set up a pipeline for streaming and chunking text from a list of jsonl files with a `"content"` field:

```python
from functools import partial
import json
from skword2vec.streams import pipe_streams, stream_files, chunk

stream_chunks = pipe_streams(
  partial(stream_files, lines=True),
  partial(map, json.loads),
  partial(map, lambda record: records["content"]),
  partial(chunk, chunk_size=10_000),
)

# let's say you have a list of file paths
files: list[str] = [...]

chunks = stream_chunks(files)
```

### Preprocessors
We provide a couple of preprocessing/tokenizing components that split texts into sentences and tokens.
These components always return nested lists or iterables where the zeroth axis is the document,
the first one is sentences and the second one is tokens. Sentencization can be disabled on all components then the document will be treated as one huge sentence.
Here's an example of a spaCy tokenizer components:
```python
import spacy
from skword2vec.preprocessing.spacy import SpacyPreprocessor

nlp = spacy.load("en_core_web_sm")
preprocessor = SpacyPreprocessor(nlp, sentencize=True, out_attribute="LEMMA")
```
These can be used in conjunction with...

### Embedding Models
We provide a handful of scikit-learn compatible components of embedding models that can be incrementally fitted given chunks of tokenized text.

Here's an example with Word2Vec:
```python
from skword2vec.models.word2vec import Word2VecVectorizer
embedding_model = Word2VecVectorizer(n_components=100, algorithm="sg")
```

`Word2VecVectorizer` and all word embedding models provide Awkward Arrays as their outputs, this is incredibly useful, as you can do arithmetic with the ragged arrays just like with numpy arrays, but you can still retain the ragged structure of the input documents.

### Wranglers
We provide scikit-learn compatible components for wrangling nested iterables and tensors/awkward arrays. These can help in transforming the output or input of models into a desirable format.

Let us build a pipeline that pools word embeddings in a document thereby providing one embedding for each document.
We are going to use scikit-partial, so that we can create a training loop later with  `partial_fit()`

```python
from skpartial.pipeline import make_partial_pipeline
from skword2vec.wranglers import ArrayFlattener, Pooler

embedding_pipeline = make_partial_pipeline(
  preprocessor,
  embedding_model,
  # Here we need to flatten out sentences
  ArrayFlattener(),
  # Then pool all embeddings in a document
  # mean is the default
  Pooler(),
)
```

### Training Loop

We do not provide built in training loops, so these have to be manually written by you.
Here's an example of a training loop that fits the pipeline over all chunks and saves a checkpoint after each chunk to disk:

```python
for i_chunk, text_chunk in enumerate(chunks):
  embedding_pipeline.partial_fit(text_chunk)
  embedding_model.save(f"checkpoints/model_checkpoint_{i_chunk}.word2vec")
```

More tools coming in the future for unsupervised/zero-shot and rule based text filtering to build
the highest quality pipelines out of messy data.
