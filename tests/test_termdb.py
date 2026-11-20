from pathlib import Path

from wmt26 import termdb

DB_PATH = Path(__file__).parent / "data" / "termdb_sample.jsonl"


def db():
    return termdb.load(DB_PATH)


def test_load_skips_comments_and_parses_fields():
    d = db()
    assert len(d) == 5
    e = d.match("他重出江湖")[0]
    assert e.equivalents["id"] == ["kembali ke dunia persilatan"]
    assert e.category == "chengyu"


def test_match_prefers_longest_headword():
    # 重出江湖 contains 江湖; only the longer entry should fire.
    got = [e.zh for e in db().match("他重出江湖了")]
    assert got == ["重出江湖"]


def test_match_finds_shorter_entry_when_the_long_one_is_absent():
    assert [e.zh for e in db().match("江湖险恶")] == ["江湖"]


def test_match_dedupes_repeated_headwords():
    assert [e.zh for e in db().match("陛下，陛下！")] == ["陛下"]


def test_match_returns_multiple_distinct_entries():
    got = {e.zh for e in db().match("师兄，陛下驾到")}
    assert got == {"师兄", "陛下"}


def test_no_match_on_a_clean_line():
    assert db().match("今天天气很好") == []


def test_hits_and_required_count_only_the_target_lang():
    entries = db().match("画蛇添足")
    assert termdb.required(entries, "en") == 1
    assert termdb.required(entries, "id") == 0  # fixture has no Indonesian for it
    assert termdb.hits("don't gild the lily", entries, "en") == 1
    assert termdb.hits("draw a snake and add feet", entries, "en") == 0


def test_hits_is_case_insensitive():
    entries = db().match("陛下")
    assert termdb.hits("YOUR MAJESTY, please", entries, "en") == 1
