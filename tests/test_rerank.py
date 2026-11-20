from wmt26.rerank import offtarget_filter, pick_best


def test_offtarget_drops_han_for_non_chinese_target():
    assert offtarget_filter(["ok lah", "有中文"], "id") == ["ok lah"]


def test_offtarget_keeps_all_if_filter_empties():
    # all candidates leak Han -> keep them rather than return nothing
    cands = ["全中文", "还是中文"]
    assert offtarget_filter(cands, "id") == cands


def test_offtarget_noop_for_chinese_target():
    cands = ["繁體中文", "ok"]
    assert offtarget_filter(cands, "zh-TW") == cands


def test_pick_best_takes_argmax():
    def fake_score(srcs, mts):
        return [len(m) for m in mts]  # longest wins

    best = pick_best("src", ["a", "bbb", "cc"], "en", score_fn=fake_score)
    assert best == "bbb"


def test_pick_best_single_candidate_skips_scorer():
    called = []

    def fake_score(srcs, mts):
        called.append(1)
        return [1.0]

    assert pick_best("src", ["only"], "en", score_fn=fake_score) == "only"
    assert not called  # scorer not invoked for a single candidate


def test_pick_best_filters_before_scoring():
    def fake_score(srcs, mts):
        return [1.0] * len(mts)

    # Han candidate dropped by the id filter; only the clean one reaches scoring
    assert pick_best("src", ["有中文", "clean"], "id", score_fn=fake_score) == "clean"
