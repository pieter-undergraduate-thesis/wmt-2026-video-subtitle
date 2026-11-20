from wmt26.candidates import generate_candidates


def test_generates_n_distinct_temps():
    # fake _complete: echo the temperature so each of the 6 temps is a distinct string
    seen = []

    def fake_complete(client, prompt, model, temperature=None):
        seen.append(temperature)
        return f"cand@{temperature}"

    out = generate_candidates("你好", "en", client=object(), complete_fn=fake_complete, n=6)
    assert len(out) == 6
    assert seen == list((0.6, 0.7, 0.8, 0.9, 1.0, 1.1))  # varied temperature, in order


def test_dedups_identical_outputs():
    def fake_complete(client, prompt, model, temperature=None):
        return "same"  # every temperature collapses to one candidate

    out = generate_candidates("你好", "en", client=object(), complete_fn=fake_complete, n=6)
    assert out == ["same"]


def test_n_limits_candidate_count():
    def fake_complete(client, prompt, model, temperature=None):
        return f"c{temperature}"

    out = generate_candidates("hi", "id", client=object(), complete_fn=fake_complete, n=3)
    assert len(out) == 3
