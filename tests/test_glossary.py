from wmt26.glossary import _parse_glossary, build_glossary, enforce
from wmt26.subtitle_io import Cue


def test_enforce_replaces_leaked_source_term():
    assert enforce("I am 沈晚意", {"沈晚意": "Shen Wanyi"}) == "I am Shen Wanyi"


def test_enforce_noop_when_term_absent():
    assert enforce("nothing to do", {"沈晚意": "Shen Wanyi"}) == "nothing to do"


def test_parse_glossary_skips_malformed_lines():
    raw = "沈晚意 -> Shen Wanyi\ngarbage line\n天云宗 -> Tianyun Sect\n"
    assert _parse_glossary(raw) == {"沈晚意": "Shen Wanyi", "天云宗": "Tianyun Sect"}


def test_build_glossary_uses_injected_complete_fn():
    def fake_complete(client, prompt, model, temperature=None):
        assert "沈晚意" in prompt  # episode text reached the prompt
        return "沈晚意 -> Shen Wanyi"

    cues = [Cue(1, 0, 1000, "沈晚意 来了")]
    g = build_glossary(cues, None, "en", client=object(), complete_fn=fake_complete)
    assert g == {"沈晚意": "Shen Wanyi"}
