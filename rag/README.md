# RAG System

This directory contains a toy retrieval-augmented generation system built from scratch.

## Setup

```bash
conda create -n env python=3.11 -y
conda activate myenv
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-ant-...
```


## Files and Directories

| File/Directory        | What it is                                                                     |
|-----------------------|--------------------------------------------------------------------------------|
| `rag.py`              | Contains the RAG system.                                                       |
| `check.py`            | Tests for each step of the system: `python check.py 1`, `python check.py 2`, … |
| `corpus`              | The text corpus used to answer the files.                                      |
| `eval_questions.json` | 16 questions with the file that answers each one.                              |
| `llm.py`              | Helper that calls Claude.                                                      |

---

## System Overview

The RAG system is designed in 6 steps that are explained below.


### Step 1: Chunking

Whole documents are too long and cover several topics, so we split them into small overlapping pieces ("chunks").
Retrieval then works on chunks, not files. For this, we create a list of dictionaries, where each dictionary defines
one chunk. 

The dictionaries contain a piece of text `text`, the document identifier `doc_id` pointing to the document the text was taken from
and a unique identifier `chunk_id`.


### Step 2: Embeddings

An embedding maps text to a vector so that texts about similar things point in similar directions. This system uses
a **TF-IDF** embedding, which consists of:

- **TF (term frequency):** how often a word appears in this chunk.
- **IDF (inverse document frequency):** how rare the word is across all chunks. "Bertha" is rare and very informative; "days" is common and less so.
- A chunk's vector has one slot per vocabulary word, holding `tf × idf`, and is then scaled to length 1.


### Step 3: Similarity search

Cosine similarity measures the angle between two vectors: 1 means same direction, 0 means nothing in common. The 
retrieval uses the cosine similarity as a score denoting how equal the question is to a chunk. The system then boils
down to:

- Embedding the chunks and question
- Scoring the similarity between each chunk and question
- Selecting the top k similar chunks
- Letting an LLM decide whether any of the chunks contains the required information


### Step 4: Prompt assembly

To allow an LLM to filter the chunks for an answer to a question, it must be presented with the chunks, question
and other instructions. The system prompt contains instructions and rules. The LLM is told to:

- Scan the chunks it is presented for an answer to the provided question
- Decline answering the question if it can not find a suitable chunk
- Citing which document contains the relevant information

The prompt contains:

- The question
- Each top k chunks wrapped in a tag `<excerpt source="file.md">..</excerpt>`


## Step 5: Generation

An LLM is tasked with solving the question.


## Step 6: Evaluation

Two points worth keeping in mind when using RAG are:

1. **Retrieval hit rate:** If the top k documents do not contain the required information, the system will not be able to provide a response.
2. **Answer accuracy:** Whether the LLM's answer actually contains the required information.
