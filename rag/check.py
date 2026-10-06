"""Check your work one step at a time.

    python check.py 1      # checks step 1
    python check.py all    # checks every step
"""

import importlib
import sys

import numpy as np

module_name = "rag"
if "--solution" in sys.argv:
    sys.argv.remove("--solution")
    sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent / "solution"))
rag = importlib.import_module(module_name)


def ok(msg):
    print(f"  ok    {msg}")


def check_step_1():
    assert rag.chunk_text("") == [], "empty text should give []"
    got = rag.chunk_text("a b c d e f g", chunk_size=4, overlap=1)
    assert got == ["a b c d", "d e f g"], f"expected ['a b c d', 'd e f g'], got {got}"
    ok("chunk_text splits with overlap")
    got = rag.chunk_text("a b c", chunk_size=4, overlap=1)
    assert got == ["a b c"], f"short text should be one chunk, got {got}"
    ok("short text is a single chunk")
    got = rag.chunk_text(" ".join(str(i) for i in range(10)), chunk_size=5, overlap=0)
    assert got == ["0 1 2 3 4", "5 6 7 8 9"], f"no trailing duplicate chunk expected, got {got}"
    ok("no duplicate chunk at the end")

    chunks = rag.build_chunks(rag.load_corpus())
    assert chunks and set(chunks[0]) >= {"doc_id", "chunk_id", "text"}, "chunks need doc_id, chunk_id, text"
    assert chunks[0]["chunk_id"].endswith("#0"), "chunk_id should look like 'equipment.md#0'"
    assert len({c["chunk_id"] for c in chunks}) == len(chunks), "chunk_ids must be unique"
    assert len({c["doc_id"] for c in chunks}) == 9, "every corpus file should produce chunks"
    ok(f"build_chunks made {len(chunks)} chunks from 9 documents")


def check_step_2():
    texts = ["vacation days vacation", "sick days", "espresso machine"]
    emb = rag.TfidfEmbedder().fit(texts)
    assert emb.vocab == {"days": 0, "espresso": 1, "machine": 2, "sick": 3, "vacation": 4}, \
        f"vocab should be the sorted unique tokens, got {emb.vocab}"
    ok("vocab")
    assert np.isclose(emb.idf[emb.vocab["days"]], np.log(4 / 3) + 1), "idf of 'days' (in 2 of 3 texts) is off"
    assert np.isclose(emb.idf[emb.vocab["sick"]], np.log(4 / 2) + 1), "idf of 'sick' (in 1 of 3 texts) is off"
    assert emb.idf[emb.vocab["sick"]] > emb.idf[emb.vocab["days"]], "rarer words should get higher idf"
    ok("idf weights")

    vecs = emb.embed(texts + ["completely unknown words"])
    assert vecs.shape == (4, 5), f"expected shape (4, 5), got {vecs.shape}"
    assert np.allclose(np.linalg.norm(vecs[:3], axis=1), 1), "each vector should have length 1"
    assert np.allclose(vecs[3], 0), "text with no known words should be all zeros"
    v = vecs[0]
    assert v[emb.vocab["vacation"]] > v[emb.vocab["days"]] > 0, "'vacation' appears twice, it should weigh more"
    ok("embed returns normalized TF-IDF vectors")


def check_step_3():
    m = np.array([[1.0, 0.0], [0.6, 0.8], [0.0, 1.0]])
    sims = rag.cosine_similarity(np.array([1.0, 0.0]), m)
    assert np.allclose(sims, [1.0, 0.6, 0.0]), f"expected [1, 0.6, 0], got {sims}"
    ok("cosine_similarity")

    index = rag.build_index()
    results = rag.search("What is the espresso machine called?", index, k=3)
    assert len(results) == 3, "search should return k results"
    assert isinstance(results[0]["score"], float), "score should be a float"
    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"], "results should be sorted best first"
    assert results[0]["doc_id"] == "office.md", f"expected office.md first, got {results[0]['doc_id']}"
    ok("search finds office.md for the espresso question")


def check_step_4():
    assert len(rag.SYSTEM_PROMPT) > 50, "write a real SYSTEM_PROMPT"
    ok("SYSTEM_PROMPT is filled in")
    retrieved = [
        {"doc_id": "vacation.md", "chunk_id": "vacation.md#0", "text": "You get 27 days.", "score": 0.5},
        {"doc_id": "office.md", "chunk_id": "office.md#3", "text": "Bertha is the espresso machine.", "score": 0.2},
    ]
    prompt = rag.build_prompt("How many days?", retrieved)
    for needle in ["How many days?", "You get 27 days.", "Bertha is the espresso machine.", "vacation.md", "office.md"]:
        assert needle in prompt, f"prompt is missing {needle!r}"
    ok("prompt contains the question, every chunk and every source name")


def check_step_5():
    index = rag.build_index()
    result = rag.answer("What is the espresso machine called?", index)
    for key in ["question", "retrieved", "prompt", "answer"]:
        assert key in result, f"answer() result is missing {key!r}"
    assert "Bertha" in result["prompt"], "the retrieved context should mention Bertha"
    if result["answer"] is None:
        ok("answer() works (no API key set, so Claude was not called)")
    else:
        assert "bertha" in result["answer"].lower(), f"Claude said: {result['answer']}"
        ok(f"Claude answered: {result['answer'].strip()[:80]}")


def check_step_6():
    qs = [
        {"question": "What is the espresso machine called?", "source": "office.md"},
        {"question": "How many vacation days do I get?", "source": "vacation.md"},
        {"question": "Parental leave?", "source": None},
    ]
    rate = rag.retrieval_hit_rate(rag.build_index(), qs, k=3)
    assert rate == 1.0, f"both answerable questions should hit, got {rate}"
    ok("retrieval_hit_rate")
    assert rag.answer_is_correct("It is called Bertha.", ["bertha"])
    assert rag.answer_is_correct("You get 27 days", ["twenty-seven", "27"])
    assert not rag.answer_is_correct("I don't know", ["27"])
    ok("answer_is_correct")


STEPS = {str(i): globals()[f"check_step_{i}"] for i in range(1, 7)}

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    for step in STEPS if which == "all" else [which]:
        print(f"Step {step}")
        try:
            STEPS[step]()
        except NotImplementedError:
            print("  todo  not implemented yet")
            sys.exit(1)
        except AssertionError as e:
            print(f"  FAIL  {e}")
            sys.exit(1)
    print("\nAll checks passed!")
