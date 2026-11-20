import json

from wmt26 import termdb_build as tb

CEDICT = """\
# CC-CEDICT
# ! license CC-BY-SA 4.0
江湖 江湖 [jiang1 hu2] /rivers and lakes/the world of martial arts/
畫蛇添足 画蛇添足 [hua4 she2 tian1 zu2] /lit. draw a snake and add feet (idiom)/fig. to ruin the effect by adding sth superfluous/
的 的 [de5] /of/~'s (possessive particle)/
"""

CIDICT = """\
江湖 江湖 [jiang1 hu2] /dunia persilatan/
"""


def write(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text, encoding="utf-8")
    return p


def test_parse_cedict_reads_glosses_and_flags_idioms(tmp_path):
    rows = tb.parse_cedict(
        write(tmp_path, "cedict.txt", CEDICT), "en", "CC-CEDICT", "CC-BY-SA-4.0"
    )
    by_zh = {r["zh"]: r for r in rows}

    assert set(by_zh) == {"江湖", "画蛇添足", "的"}  # simplified column, comments skipped
    assert by_zh["江湖"]["equivalents"]["en"] == ["rivers and lakes", "the world of martial arts"]
    # the "(idiom)" marker CC-CEDICT already carries is the free chengyu detector
    assert by_zh["画蛇添足"]["category"] == "chengyu"
    assert "category" not in by_zh["的"]


def test_parse_cedict_reads_cidict_as_indonesian(tmp_path):
    # CC-CIDICT is a direct CC-CEDICT derivative -- same parser, different lang.
    rows = tb.parse_cedict(
        write(tmp_path, "cidict.txt", CIDICT), "id", "CC-CIDICT", "CC-BY-SA-4.0"
    )
    assert rows[0]["equivalents"] == {"id": ["dunia persilatan"]}


def test_parse_manual_reads_optional_columns(tmp_path):
    p = write(
        tmp_path,
        "wuxia.tsv",
        "zh\ten\tid\tcategory\tregister\tmeaning_en\tavoid\n"
        "江湖\tjianghu;the martial world\tdunia persilatan\twuxia_term\thistorical\tthe martial world\trivers and lakes\n"
        "陛下\tYour Majesty\t\ttitle_honorific\t\t\t\n",
    )
    rows = tb.parse_manual(p)

    assert rows[0]["equivalents"] == {"en": ["jianghu", "the martial world"], "id": ["dunia persilatan"]}
    assert rows[0]["avoid"] == ["rivers and lakes"]
    assert rows[0]["confidence"] == "verified"
    assert rows[1]["equivalents"] == {"en": ["Your Majesty"]}  # blank id column dropped
    assert rows[1]["register"] == "neutral"  # blank defaults


def test_parse_idiomkb_takes_meanings(tmp_path):
    p = write(
        tmp_path, "idiomkb.json",
        json.dumps([{"idiom": "画蛇添足", "en_meaning": "to ruin by adding excess"}]),
    )
    row = tb.parse_idiomkb(p)[0]
    assert row["meaning_en"] == "to ruin by adding excess"
    assert row["confidence"] == "llm"  # GPT-3.5-generated per the repo's own warning


def test_merge_unions_equivalents_and_keeps_the_strictest_license():
    merged = tb.merge(
        [{"zh": "江湖", "equivalents": {"en": ["jianghu"]}, "category": "wuxia_term",
          "provenance": ["CC-CEDICT"], "license": "CC-BY-SA-4.0"}],
        [{"zh": "江湖", "equivalents": {"en": ["the martial world"], "id": ["dunia persilatan"]},
          "provenance": ["PETCI"], "license": "CC-BY-NC-SA-4.0"}],
    )
    assert len(merged) == 1
    e = merged[0]
    assert e["equivalents"]["en"] == ["jianghu", "the martial world"]
    assert e["equivalents"]["id"] == ["dunia persilatan"]
    assert e["provenance"] == ["CC-CEDICT", "PETCI"]
    assert e["license"] == "CC-BY-NC-SA-4.0"  # most restrictive wins
    assert e["category"] == "wuxia_term"


def test_merge_does_not_duplicate_identical_equivalents():
    merged = tb.merge(
        [{"zh": "陛下", "equivalents": {"en": ["Your Majesty"]}}],
        [{"zh": "陛下", "equivalents": {"en": ["Your Majesty"]}}],
    )
    assert merged[0]["equivalents"]["en"] == ["Your Majesty"]


def test_filter_drops_uncategorised_vocabulary_and_single_chars():
    entries = [
        {"zh": "画蛇添足", "category": "chengyu", "equivalents": {"en": ["x"]}},
        {"zh": "的", "category": "", "equivalents": {"en": ["of"]}},          # generic
        {"zh": "好", "category": "chengyu", "equivalents": {"en": ["good"]}},  # too short
    ]
    assert [e["zh"] for e in tb.filter_entries(entries)] == ["画蛇添足"]


def test_filter_attested_keeps_only_what_fires_on_the_corpus(tmp_path):
    (tmp_path / "v1_zh.srt").write_text(
        "1\n00:00:00,000 --> 00:00:02,000\n他重出江湖了\n\n", encoding="utf-8"
    )
    entries = [
        {"zh": "江湖", "category": "wuxia_term", "equivalents": {"en": ["jianghu"]}},
        {"zh": "画蛇添足", "category": "chengyu", "equivalents": {"en": ["gild the lily"]}},
    ]
    kept = tb.filter_entries(entries, tests_dir=tmp_path)
    assert [e["zh"] for e in kept] == ["江湖"]


def test_enrich_only_fills_missing_fields():
    reply = (
        "meaning: the world of martial artists\n"
        "register: historical\n"
        "category: wuxia_term\n"
        "indonesian: dunia persilatan\n"
        "avoid: rivers and lakes"
    )
    entries = [{
        "zh": "江湖",
        "meaning_en": "KEEP ME",  # source data already has it
        "equivalents": {"en": ["jianghu"]},
        "category": "",
        "register": "neutral",
    }]
    out = tb.enrich(entries, client=object(), complete_fn=lambda *a, **k: reply)[0]

    assert out["meaning_en"] == "KEEP ME"  # source wins over the LLM
    assert out["category"] == "wuxia_term"  # was missing -> filled
    assert out["equivalents"]["id"] == ["dunia persilatan"]  # English-pivot
    assert out["avoid"] == ["rivers and lakes"]
    assert out["confidence"] == "llm"


def test_enrich_skips_entries_that_need_nothing():
    calls = []

    def fn(*a, **k):
        calls.append(a)
        return ""

    tb.enrich(
        [{"zh": "江湖", "meaning_en": "m", "category": "wuxia_term",
          "equivalents": {"en": ["jianghu"], "id": ["dunia persilatan"]}}],
        client=object(), complete_fn=fn,
    )
    assert calls == []


def test_qc_drops_entries_with_no_usable_rendering():
    kept, report = tb.qc([
        {"zh": "江湖", "equivalents": {"en": ["jianghu"]}, "category": "wuxia_term",
         "license": "CC-BY-SA-4.0", "confidence": "verified"},
        {"zh": "无用", "equivalents": {}, "category": "chengyu"},
    ])
    assert [e["zh"] for e in kept] == ["江湖"]
    assert report["dropped"] == 1
    assert report["category"] == {"wuxia_term": 1}


def test_emit_roundtrips_through_the_runtime_loader(tmp_path):
    from wmt26 import termdb

    out = tmp_path / "termdb.jsonl"
    n = tb.emit([
        {"zh": "江湖", "meaning_en": "m", "equivalents": {"en": ["jianghu"]},
         "avoid": ["rivers and lakes"], "register": "historical",
         "category": "wuxia_term", "provenance": ["x"], "license": "CC-BY-SA-4.0",
         "confidence": "verified"},
    ], out)

    assert n == 1
    db = termdb.load(out)  # build output must be readable by the runtime, unchanged
    e = db.match("重出江湖")[0]
    assert e.zh == "江湖" and e.avoid == ["rivers and lakes"]


def test_emit_exclude_nc_drops_noncommercial_entries(tmp_path):
    out = tmp_path / "termdb.jsonl"
    n = tb.emit([
        {"zh": "江湖", "equivalents": {"en": ["jianghu"]}, "license": "CC-BY-SA-4.0"},
        {"zh": "画蛇添足", "equivalents": {"en": ["gild the lily"]}, "license": "CC-BY-NC-SA-4.0"},
    ], out, exclude_nc=True)

    assert n == 1
    assert "画蛇添足" not in out.read_text(encoding="utf-8")
