from wmt26.fewshot import load_pool, retrieve
from wmt26.postedit import postedit


POOL = [
    ("我喜欢你", "aku suka kamu"),
    ("你好世界", "halo dunia"),
    ("我喜欢他", "aku suka dia"),
]


def test_retrieve_ranks_by_similarity():
    # closest source to "我喜欢你" is itself, then the near-duplicate "我喜欢他"
    out = retrieve("我喜欢你", POOL, k=2)
    assert out[0] == ("我喜欢你", "aku suka kamu")
    assert out[1] == ("我喜欢他", "aku suka dia")


def test_retrieve_empty_pool():
    assert retrieve("anything", [], k=5) == []


def test_load_pool_skips_blank_and_malformed(tmp_path):
    f = tmp_path / "pool.tsv"
    f.write_text("我喜欢你\taku suka kamu\n\nno-tab-line\n你好\thalo\n", encoding="utf-8")
    assert load_pool(f) == [("我喜欢你", "aku suka kamu"), ("你好", "halo")]


def test_postedit_runs_exactly_one_pass():
    calls = []

    def fake_complete(client, prompt, model, temperature=None):
        calls.append(prompt)
        return "refined"

    out = postedit("src", "draft", "en", client=object(), complete_fn=fake_complete)
    assert out == "refined"
    assert len(calls) == 1  # one iteration only, never a refinement loop
