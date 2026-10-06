"""RAG from scratch.

When everything is done:
    python rag.py ask "How many vacation days do I get?"
    python rag.py eval
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

from llm import call_claude, has_api_key

ROOT = Path(__file__).resolve().parent
CORPUS_DIR = ROOT / "corpus"
EVAL_FILE = ROOT / "eval_questions.json"

STOPWORDS = set(
    "a an and are as at be by can do does for from get has have how i if in is it "
    "its my of on or our so that the their there this to up we what when where "
    "which who will with you your".split()
)


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[dict]:
    """Return [{"doc_id": "vacation.md", "text": "..."}, ...]."""
    return [
        {"doc_id": path.name, "text": path.read_text(encoding="utf-8")}
        for path in sorted(corpus_dir.glob("*.md"))
    ]


def tokenize(text: str) -> list[str]:
    """Lowercase, split into words, drop very common words.

    >>> tokenize("How many vacation days do I get?")
    ['many', 'vacation', 'days']
    """
    words = re.findall(r"[a-z0-9]+", text.lower())
    return [w for w in words if w not in STOPWORDS]


# ---------------------------------------------------------------------------
# Step 1: Chunking
# ---------------------------------------------------------------------------

def chunk_text(text: str, chunk_size: int = 50, overlap: int = 10) -> list[str]:
    """Split `text` into chunks of `chunk_size` words.

    Each chunk starts `chunk_size - overlap` words after the previous one, so
    neighboring chunks share `overlap` words.

    Example with chunk_size=4, overlap=1:
        "a b c d e f g"  ->  ["a b c d", "d e f g"]
    """
    if len(text) == 0:
        return []

    text = text.split()
    chunks = []
    lower_bound = 0
    while lower_bound <= len(text):
        chunks.append(" ".join(text[lower_bound:lower_bound + chunk_size]))
        if lower_bound + chunk_size >= len(text):
            break
        lower_bound += chunk_size - overlap
    return chunks


def build_chunks(docs: list[dict], chunk_size: int = 50, overlap: int = 10) -> list[dict]:
    """Chunk every document and remember where each chunk came from.

    Return a flat list like:
        [{"doc_id": "vacation.md", "chunk_id": "vacation.md#0", "text": "..."},
         {"doc_id": "vacation.md", "chunk_id": "vacation.md#1", "text": "..."}, ...]
    """
    chunk_list = []
    for doc in docs:
        if "text" not in doc or "doc_id" not in doc:
            continue
        doc_id = doc["doc_id"]
        doc_text = doc["text"]

        chunks = chunk_text(doc_text, chunk_size, overlap)
        for idx, chunk in enumerate(chunks):
            chunk_dict = {
                "doc_id": doc_id,
                "chunk_id": f"{doc_id}#{idx}",
                "text": chunk
            }
            chunk_list.append(chunk_dict)
    return chunk_list


# ---------------------------------------------------------------------------
# Step 2: Embeddings
# ---------------------------------------------------------------------------

class TfidfEmbedder:
    """Tfidf Embedder"""
    def __init__(self):
        self.vocab = None
        self.idf = None

    def fit(self, texts: list[str]) -> "TfidfEmbedder":
        """Learn the vocabulary and IDF weights from the chunk texts.

        Set two attributes:
          self.vocab: dict mapping each word (from tokenize) to a column index
                      0...V-1. Sort the words so the order is deterministic.
          self.idf:   numpy array of length V where
                      idf[i] = log((1 + N) / (1 + df)) + 1
                      N  = number of texts
                      df = number of texts that contain word i at least once
        Return self.
        """
        word_count = Counter()
        for sentence in texts:
            tokenized = set(tokenize(sentence))
            word_count.update(tokenized)

        words = list(sorted(word_count.keys()))
        n = len(texts)

        self.vocab = {}
        self.idf = np.zeros(len(words))
        for i, word in enumerate(words):
            self.vocab[word] = i
            self.idf[i] = np.log((1 + n) / (1 + word_count[word])) + 1
        return self

    def embed(self, texts: list[str]) -> np.ndarray:
        """Turn texts into an (n_texts, V) matrix of L2-normalized TF-IDF vectors.

        For each text: count how often each known word appears (term frequency),
        multiply each count by that word's idf, then divide the row by its length
        (np.linalg.norm) so every vector has length 1. Ignore words that are not
        in the vocabulary. A text with no known words stays an all-zero row.
        """
        embedding = np.zeros((len(texts), len(self.vocab)))
        for i, sentence in enumerate(texts):
            word_count = Counter(tokenize(sentence))
            for word in word_count.keys():
                if word not in self.vocab:
                    continue
                word_idx = self.vocab[word]
                word_idf = self.idf[word_idx]
                embedding[i, word_idx] = word_idf * word_count[word]

        vec_length = np.linalg.norm(embedding, axis=1).reshape(-1, 1)
        vec_length[vec_length == 0] = 1
        embedding /= vec_length
        return embedding


# ---------------------------------------------------------------------------
# Step 3: Similarity search
# ---------------------------------------------------------------------------

def cosine_similarity(query_vec: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    """Return the cosine similarity between `query_vec` (shape V) and every row
    of `matrix` (shape n x V), as an array of shape n.

    Hint: all vectors are already normalized to length 1.
    """
    return matrix @ query_vec


def search(query: str, index: dict, k: int = 3) -> list[dict]:
    """Return the k chunks most similar to `query`, best first.

    `index` is what build_index() returns. Each result is the chunk dict with
    an extra "score" key (a plain float).
    """
    embedder = index["embedder"]
    chunks = index["chunks"]
    vectors = index["vectors"]

    query_embed = embedder.embed([query]).reshape(-1)

    similarities = cosine_similarity(query_embed, vectors)
    sort_indices = np.argsort(similarities, descending=True)

    search_result = []
    for index in sort_indices[:k]:
        chunk_dict = chunks[index]
        chunk_dict["score"] = similarities[index]
        search_result.append(chunk_dict)
    return search_result


def build_index(chunk_size: int = 50, overlap: int = 10) -> dict:
    """Load, chunk and embed the corpus. Provided once steps 1 and 2 work."""
    chunks = build_chunks(load_corpus(), chunk_size, overlap)
    texts = [c["text"] for c in chunks]
    embedder = TfidfEmbedder().fit(texts)
    return {"chunks": chunks, "embedder": embedder, "vectors": embedder.embed(texts)}


# ---------------------------------------------------------------------------
# Step 4: Prompt assembly
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "Answer the following question based on the handbook excerpts you are given."
    "If the excerpts do not contain the answer, be truthful about it and say you do not know the answer."
    "If you can find the answer, cite the file by its name in brackets, for example [file.md]."
)


def build_prompt(question: str, retrieved: list[dict]) -> str:
    """Build the user message: the retrieved chunks plus the question.

    Label every chunk with its doc_id so Claude can cite it. Wrapping each one
    in a tag such as <excerpt source="vacation.md">...</excerpt> works well.
    """
    prompt = question
    for retrieved_chunk in retrieved:
        prompt += f"<excerpt source='{retrieved_chunk['doc_id']}'>{retrieved_chunk['text']}</excerpt>"
    return prompt


# ---------------------------------------------------------------------------
# Step 5: Generation
# ---------------------------------------------------------------------------

def answer(question: str, index: dict, k: int = 3) -> dict:
    """Retrieve, build the prompt, and ask Claude.

    Return {"question": ..., "retrieved": [...], "prompt": ..., "answer": ...}.
    """
    retrieved = search(question, index, k)
    prompt = build_prompt(question, retrieved)
    response = call_claude(SYSTEM_PROMPT, prompt)
    return {
        "question": question,
        "retrieved": retrieved,
        "prompt": prompt,
        "answer": response
    }


# ---------------------------------------------------------------------------
# Step 6: Evaluation
# ---------------------------------------------------------------------------

def retrieval_hit_rate(index: dict, questions: list[dict], k: int = 3) -> float:
    """Fraction of answerable questions whose `source` file appears among the
    doc_ids of the top-k search results.
    """
    relevant, found = 0, 0
    for question in questions:
        required_source = question["source"]
        if required_source is None:
            continue
        relevant += 1
        nearest_chunks = search(question["question"], index, k)
        for chunk in nearest_chunks:
            if chunk["doc_id"] == required_source:
                found += 1
                break
    return found / relevant


def answer_is_correct(reply: str, expected: list[str]) -> bool:
    """True if any string in `expected` appears in `reply`, ignoring case."""
    for e in expected:
        if re.search(e, reply, re.IGNORECASE):
            return True
    return False


def evaluate(k: int = 3) -> None:
    """Provided: prints retrieval hit rate, then grades Claude's answers."""
    questions = json.loads(EVAL_FILE.read_text())
    index = build_index()

    print(f"Retrieval hit rate @ {k}:")
    print(f"  {retrieval_hit_rate(index, questions, k):.0%}\n")

    if not has_api_key():
        print("Set ANTHROPIC_API_KEY to also grade the generated answers.")
        return

    correct = 0
    for q in questions:
        reply = answer(q["question"], index, k)["answer"]
        ok = answer_is_correct(reply, q["answer_contains"])
        correct += ok
        print(f"[{'PASS' if ok else 'FAIL'}] {q['question']}\n       {reply.strip()}\n")
    print(f"Answer accuracy: {correct}/{len(questions)}")


# ---------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) >= 3 and sys.argv[1] == "ask":
        result = answer(" ".join(sys.argv[2:]), build_index())
        print("Retrieved chunks:")
        for c in result["retrieved"]:
            print(f"  {c['score']:.3f}  {c['chunk_id']}  {c['text'][:70]}...")
        if result["answer"] is None:
            print("\nNo ANTHROPIC_API_KEY set, so here is the prompt Claude would get:\n")
            print(result["prompt"])
        else:
            print("\nAnswer:\n" + result["answer"])
    elif len(sys.argv) == 2 and sys.argv[1] == "eval":
        evaluate()
    else:
        print(__doc__)


if __name__ == "__main__":
    main()
