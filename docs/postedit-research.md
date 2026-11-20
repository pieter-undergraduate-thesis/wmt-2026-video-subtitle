# Retrieval-Based Post-Editing with an Idiom/Terminology Database for Chinese→English and Chinese→Indonesian Subtitle MT (WMT26 Video Subtitle Translation)

**Status: implemented.** `src/wmt26/{termdb,termdb_build}.py`, the selective pass
in `src/wmt26/postedit.py`, `metrics.term_recall`, and
`scripts/{run_postedit,build_termdb}.py` implement §3–§5 of this document. This
file is kept as the original research record — see
[`docs/PIPELINE.md`](./PIPELINE.md) for the current, maintained module reference.
Two deliberate departures, both argued there: the QE gate (§3.1) is opt-in rather
than default, because §Key-Findings-3 of this very document reports QE
correlating near-zero on idiom lines; and detection uses substring matching
rather than jieba + Aho-Corasick (§5.4), since Chinese has no spaces and
segmentation only loses recall on fixed expressions. Embedding retrieval for
paraphrased idioms and IdiomTranslate30 remain deferred.

## TL;DR
- **Build a selective, retrieval-augmented LLM post-editor on top of Hy-MT2-7B**: detect idiom/term-bearing subtitle lines, retrieve entries (figurative meaning + target equivalents) from a purpose-built idiom/term database, and prompt the LLM to revise only the flagged lines. The literature strongly supports this "translate meanings, not just words" approach for exactly the failure mode that hurts historical cdrama — literal rendering of chengyu, honorifics, imperial titles, and cultivation/martial-arts terms.
- **Evidence supports it**: IdiomKB-style figurative-meaning injection improved GPT-4-judged idiom accuracy by +0.06 to +0.58 on a 1–3 scale (Li et al., AAAI 2024); LLM terminology post-editing raised term usage from 36.67% to 72.88% on the blind set (Moslem et al., WMT23) and Zh→En term usage rose 28.33%→82.65%. Even GPT-4 makes idiom-translation errors in 28% of cases (IdiomEval, arXiv:2508.10421), so headroom is large.
- **The database is the moat**: harvest zh→en glossaries (CC-CEDICT, PETCI/IdiomTranslate30/IdiomKB, wuxia/xianxia fan glossaries, honorific/title references, cdrama community glossaries), enrich with LLM-generated meanings/register tags, and generate Indonesian equivalents by English-pivot seeded with CC-CIDICT, with human/LLM verification. Prioritize historical-drama categories (titles/honorifics, cultivation/martial arts, chengyu) for maximum test-set coverage.

## Key Findings
1. **Neural-on-neural APE historically underdelivered, but LLM + retrieval changed the calculus.** Classic WMT APE findings (Junczys-Dowmunt & Grundkiewicz, 2018, dual-source transformer) concluded "neural-on-neural APE might not actually be useful" for strong in-domain NMT. The revival comes from (a) LLM post-editors that carry world/cultural knowledge, and (b) *external retrieval* that supplies knowledge the base model lacks — precisely the out-of-domain gap in historical cdrama.
2. **Selective post-editing beats blanket editing.** APE systems overcorrect; the WMT25 QE-informed error-correction winners minimized edits to maximize a gain-to-edit ratio. A detection gate (idiom/term match + QE) is essential to avoid degrading already-correct lines.
3. **Standard metrics are unreliable for idioms.** On Chinese idiom translation the highest metric–human correlation is only 0.386 at full-context level and 0.483 at idiom level (COMET/MetricX-23; IdiomEval, arXiv:2508.10421), and reference-free QE (COMETKiwi) is near-zero or negative. Evaluation must combine COMET/chrF with term-recall and an LLM-as-judge idiom-accuracy rubric.
4. **The zh→id resource gap is real.** Indonesian has under 50k Chinese parallel pairs total and is treated as low-resource; almost all cdrama/idiom glossaries are zh→en, forcing an English-pivot construction strategy with verification.
5. **WMT26 constraints shape the design.** Constrained track only; final model <20B params; weights must be open-sourced; SRT output; rich metadata (synopsis, character lists, glossaries) is provided and should be fed into retrieval. Hy-MT2 is the recommended base and carries prize eligibility.

## Details

### 1. Task and pipeline context
The WMT26 Video Subtitle Translation task (organized by Tencent Hunyuan and Tencent Video) translates Simplified-Chinese subtitles into English, Thai, Indonesian, Malay, and Traditional Chinese (Taiwan). It is a **constrained track only**: any training data/models are allowed provided final weights are released under a permissive license (e.g., Apache/MIT) and total parameters are under 20B; submissions are SRT files (`vid_langshort.srt`) evaluated by both automatic metrics and human evaluation. Metadata (video description/synopsis, character lists, domain, glossaries) is provided and encouraged. Hy-MT2 (Tencent's "fast-thinking" multilingual model family: 1.8B/7B/30B-A3B, 33 languages; arXiv 2605.22064) is the recommended base and carries monetary-prize eligibility. The user's stack is Hy-MT2-7B served via vLLM, LoRA fine-tuned with LLaMA-Factory, with reranking and few-shot example retrieval already implemented. This document adds a **retrieval-augmented post-editing stage** on top of the first-pass output.

Much observed test data is HISTORICAL cdrama (wuxia, xianxia, costume/palace), where a generic MT model mishandles chengyu, honorifics, imperial titles, cultivation/martial-arts terms, and culture-specific items. Modern-register cdrama and internet slang also appear.

### 2. Prior-work synthesis

**2.1 Automatic post-editing (APE).** APE improves MT output by learning from human edits (Knight & Chander 1994; WMT APE shared tasks since 2015, NMT from 2018). A robust finding: APE helped SMT substantially but struggled to improve strong NMT (Chatterjee et al. 2018/2019; Junczys-Dowmunt & Grundkiewicz 2018). Later work (arXiv 2009.14395) showed larger human-PE corpora can still help. The modern reframing is LLM-based APE and quality-aware post-editing: MQM-APE (arXiv 2409.14335) found all tested LLMs improved translation quality over the base output on COMETKiwi and BLEURT; the WMT25 QE-informed segment-level error-correction task saw training-free "QE-informed retranslation" and error-span replacement win, with a heuristic to minimize edits (maximize gain-to-edit ratio).

**2.2 Retrieval-augmented MT and post-editing.** kNN-MT (Khandelwal et al. 2021) augments an NMT decoder with a token-level datastore for training-free domain adaptation, but is slow (often an order-of-magnitude decoding slowdown; hence Fast kNN-MT, arXiv 2105.14528). Translation-memory augmentation via fuzzy matches (Bulté & Tezcan 2019) and segment-level TM retrieval (ScienceDirect S2667305326000487) concentrate gains on "high-similarity segments containing idioms, domain terminology, and formulaic language — precisely the cases where NMT is often unreliable." This motivates a *phrase/entry-level* retrieval DB rather than a token datastore.

**2.3 Idiom-specific MT.** IdiomKB (Li et al., AAAI 2024; arXiv:2308.13961; repo lishuang-w/IdiomKB) is the anchor: a multilingual KB storing idioms + figurative meanings (En/Zh/Ja), built by distilling meanings from LLMs, then injected into the prompt as a chain-of-thought "transition" (KB-CoT). Concrete results: KB-CoT improved GPT-4-judged idiom translation (1–3 scale) consistently, with the largest gains on weaker models — "KB-CoT prompting with meaning retrieved from IdiomKB consistently surpasses direct prompting for smaller models" (Alpaca-7B Zh→En +0.58: 1.54→2.12; InstructGPT-6.7B Zh→En +0.42: 1.66→2.08; ChatGPT/GPT-4 +0.06 to +0.13). The KB scored 2.92/3 in human quality evaluation (reported as high quality). IdiomKB's Chinese side aggregates PETCI (4,310 idioms), CCT, and ChID into 8,643 Chinese idioms. Crucially, **sacreBLEU often DROPS while GPT-4 idiom scores rise** (e.g., InstructGPT Zh→En BLEU 14.26→9.64), and GPT-4-judge correlates far better with humans (Zh→En r=0.69) than COMET (0.25) or BLEU (0.09). The follow-up (arXiv:2407.03518) extends beyond meaning to *target-language idiom alignment* via SentenceTransformers cosine-similarity lookup (SIA) vs LLM idiom alignment (LIA), with SIA best in GPT-4o.

Datasets/benchmarks:
- **PETCI** (Tang 2022; arXiv:2202.09509): 4,310 Chinese idioms × 29,936 English translations (idiom dictionary + Google/DeepL), CC-BY-NC-SA.
- **IdiomTranslate30** (kenantang; HuggingFace): 2,719,800 context-aware translations, 3 source (zh/ja/ko) × 10 target languages (includes English; NOT Indonesian directly), Gemini-3.0-Flash-generated, three strategies (creative/analogy/author) with idiom span annotations, CC-BY-NC-SA. Companion EMNLP 2024 Findings paper "Creative and Context-Aware Translation of East Asian Idioms with GPT-4."
- **IdiomEval** ("Evaluating LLMs on Chinese Idiom Translation," Yang et al., Georgia Tech, arXiv:2508.10421): a 623K-sentence auto corpus across News/Web/Wikipedia/Social Media (built over Tan & Jiang's 30,999-idiom vocabulary) plus 900 human-annotated pairs across 9 systems. Error taxonomy: **No Error, Mistranslation, Unnatural, Literal, Addition, Partial, Repetition, No Translation** (severity 1–3). "The best-performing system, GPT-4, makes errors in 28% of cases." Google Translate produces Literal Translations 36% of the time on Social Media; most common errors overall are Mistranslation, Literal, and Partial; idiom frequency in pretraining was not correlated with quality. Standard metrics correlate weakly (best 0.386 full / 0.483 idiom level; reference-free QE near-zero/negative), and xCOMET's best idiom-error-span F1 is only ~0.30.

**2.4 Terminology-aware/constraint-based decoding & soft-constraint prompting.** DiPMT (Ghazvininejad et al. 2023) injects dictionary translations into prompts. Terminology-constrained MT with LLMs (Moslem et al. 2023, "Adaptive MT with LLMs," arXiv:2301.13294) inserts glossary terms into prompts (zero-shot + terms; two-shot + fuzzy matches). Terminology-constrained *APE* (Moslem et al., WMT23, "Domain Terminology Integration into Machine Translation," aclanthology.org/2023.wmt-1.82): the LLM inserts missing glossary terms into MT output — term usage "increases from an average of 36.67% with the generic model to an average of 72.88%… successful utilisation of terms nearly doubles across the three language pairs," with Zh→En term usage rising 28.33%→82.65% (Table 4) while BLEU/chrF/COMET improved. Terminology-Aware Translation with constrained decoding + LLM prompting (Bogoychev & Chen, WMT23, aclanthology.org/2023.wmt-1.80): ~80% recall overall; Zh→En recall via LLM refinement reaches 82.03–83.06 (from ~50 with training-only). Translate-and-Revise (arXiv:2407.13164): iterative revision using **rule-detected** unmet constraints gives ~15% average CCR (constraint completion rate) gain, reaching 95–97.6% CCR on En↔Zh without hurting COMET; feeding the *exact* unmet constraints (not LLM self-detection) is critical. Practical pattern (Lokalise/Translated): match the source string lexically and inject only the terms that appear — per-segment glossary injection keeps context clean.

**2.5 Self-refinement.** TEaR (Feng et al., NAACL 2025; arXiv:2402.16379) = Translate, Estimate, Refine: the LLM assesses its own output with MQM-style error typing, then refines; it outperforms internal-refinement and external-feedback baselines. This maps directly onto our Estimate (detect + QE) → Refine (retrieval-augmented edit) stages.

**2.6 Subtitle-specific constraints.** Standard subtitling guidelines: ≤2 lines; ~42 CPL for Latin scripts (47 online), 12–16 for Chinese; reading speed ~15–21 CPS (Netflix caps English at ~20 CPS/42 CPL); linguistic-whole segmentation; roughly equal line lengths; **never extend cue duration to fix reading speed — shorten the wording instead**. Text expansion matters: translating zh→en/id expands character count, so the post-editor must be length-aware. Document-level coherence (speaker consistency, term consistency across lines, register/tone) is a subtitle-specific requirement; VideoDubber and HOMURA (arXiv:2601.10187) address length/isochrony control.

### 3. Proposed method: Retrieval-Augmented Post-Editing 

**Pipeline (text diagram):**
```
Source SRT line s_i (+ neighbors s_{i-k..i+k}, metadata: synopsis, character list, domain, glossary)
        │
        ▼
[1] First-pass translation  t_i  ← Hy-MT2-7B (vLLM) + existing few-shot retrieval + reranking
        │
        ▼
[2] DETECTION / TRIGGER GATE  (decide if line needs editing)
     • Idiom/term matching against DB: exact + fuzzy (jieba segmentation, Aho-Corasick/trie)
     • Embedding retrieval for variable phrasing (top-k over DB entry embeddings)
     • QE signal: COMETKiwi score below threshold  OR  model uncertainty/logprob
     • Register/slang classifier flags internet-slang & honorific lines
     → If no trigger fires: PASS THROUGH t_i unchanged
        │ (triggered)
        ▼
[3] RETRIEVAL: pull DB entries for matched spans
     {simplified, pinyin, literal, figurative meaning, register, category,
      EN equivalents + usage notes, ID equivalents, example pairs}
        │
        ▼
[4] LLM POST-EDITOR (Hy-MT2-7B or a ≤20B general LLM), prompt =
     source line + neighbor context + draft t_i + retrieved knowledge + subtitle constraints
     → produces revised t_i'
        │
        ▼
[5] QE/RERANK ACCEPT-OR-REJECT GATE
     • Generate 1–n candidates; score with COMETKiwi + term-recall check + length/CPS check
     • Accept t_i' only if it (a) uses required target equivalents, (b) does not regress QE,
       (c) satisfies CPL/CPS. Else keep t_i.  (maximize gain-to-edit ratio)
        │
        ▼
Output revised SRT
```

**3.1 Detection strategy (which lines to edit).** Combine cheap high-precision signals so post-editing stays selective:
- **Dictionary/idiom matching (primary trigger, near-free):** segment the source with jieba (custom dictionary loaded with all DB headwords) plus a trie/Aho-Corasick pass for multi-character fixed expressions; chengyu are typically 4 characters and highly matchable.
- **Embedding retrieval** for variable/paraphrased phrasing (cosine over multilingual sentence embeddings).
- **QE gate:** COMETKiwi (wmt22-cometkiwi-da, 0.5B, or the 23-XL variant) on (source, draft); low score → candidate for editing. Caveat: QE is weak specifically on idioms, so it should *supplement*, not replace, dictionary matching.
- **Uncertainty:** low token log-probabilities / high entropy over the idiom span.
- **Register classifier:** flag internet-slang and honorific/title lines for register-consistency editing.

**3.2 Prompt design (sketch).**
```
System: You are an expert subtitle post-editor for Chinese historical drama (wuxia/xianxia/
palace). Revise the DRAFT translation of ONE subtitle line so it is accurate, natural, and
consistent with the scene. Use the GLOSSARY for culture-specific terms, idioms (chengyu),
honorifics, and titles. Keep register consistent (archaic/formal for historical; casual for
modern; slang where appropriate). Obey subtitle limits: ≤2 lines, ≤{CPL} chars/line, reading
speed ≤{CPS} cps; if the draft is already correct, return it UNCHANGED.

Context (previous/next lines): {s_{i-2..i-1}} → {t prev} ; {s_{i+1..i+2}}
Character list / metadata: {names, roles, domain, synopsis snippet}
GLOSSARY (only entries matched in this line):
- 江湖 (jiānghú) [martial-arts world / underworld of martial artists] category=wuxia;
  EN="jianghu / the martial world"; ID="dunia persilatan"; note: do NOT translate literally as "rivers and lakes".
- 陛下 (bìxià) [honorific for emperor] category=honorific; EN="Your Majesty"; ID="Yang Mulia / Baginda".
SOURCE: {s_i}
DRAFT: {t_i}
Return only the revised subtitle text.
```
Design notes: English prompt scaffolding + target-language meaning is optimal per IdiomKB; inject *only matched* entries (per-segment) to keep context clean; give the *exact* required target equivalent (rule-detected) rather than asking the model to self-detect (per Translate-and-Revise); include neighbor lines for pronoun/speaker/term coherence but cap context (e.g., ±2 lines) for latency.

**3.3 Context-window strategy.** Use a window of ±2 subtitle lines for local coherence plus a running per-episode term memory (character names, chosen equivalents) to enforce consistency across the whole file — critical because subtitle cues are short and ambiguous alone. Inject the metadata synopsis/character list once per batch.

**3.4 Failure modes and mitigations.**
- **Over-editing / regression on correct lines** → the accept-or-reject QE+term gate; "return UNCHANGED" instruction; gain-to-edit heuristic.
- **Literal idiom rendering** (the core out-of-domain failure) → figurative-meaning + target-equivalent retrieval (IdiomKB pattern).
- **Hallucinated / added content** → constrain to minimal edits; reject candidates that add unsupported content (IdiomEval "Addition" error); verify entities against the character list.
- **Register drift** (modern phrasing in a historical scene, or over-formalizing slang) → register tag in the retrieved entry + explicit register instruction.
- **Length violation** → CPL/CPS check in the gate; instruct shortening, never extend the cue.
- **Metric blindness** → do not tune solely on BLEU/COMET for idiom lines; use term-recall + LLM-judge.

### 4. Evaluation methodology
- **Overall:** BLEU/chrF (sacreBLEU), COMET (reference-based), COMETKiwi (reference-free) — treated with caution on idioms.
- **Term/idiom-specific:** **term recall / constraint conformance rate (CCR)** — fraction of required DB target equivalents present in the output (the metric that moved 36.67%→72.88%, and Zh→En to 82.65%, in prior work); **idiom translation accuracy** via an LLM-as-judge 1–3 rubric (IdiomKB) or the IdiomEval 7-category error taxonomy (Mistranslation/Literal/Partial/etc.).
- **Ablations:** base vs +detection vs +retrieval vs +QE-gate; measure gain-to-edit ratio (quality gain per edited line) and over-edit rate (regressions on originally-correct lines).
- **Subtitle compliance:** % lines within CPL/CPS; document-level term-consistency rate.
- **Human eval** mirrors WMT26's own protocol (final ranking is human).

### 5. PART 2 — Building the idiom/term database

**5.1 Data sources to harvest.**
- **CC-CEDICT** (MDBG): 124,762 entries as of the 2026-07-05 release (mdbg.net), CC-BY-SA 4.0, `Traditional Simplified [pin1 yin1] /def1/def2/` format — the backbone for headwords, simplified/traditional, pinyin, base English glosses; includes many chengyu.
- **PETCI** (CC-BY-NC-SA): 4,310 chengyu × ~30k English translations (multiple renderings + error/paraphrase variety).
- **IdiomTranslate30 / IdiomKB**: figurative meanings (En/Zh/Ja) and context-aware target renderings; IdiomKB's `zh_idiom_meaning.json` gives id/idiom/en_meaning/zh_meaning/ja_meaning (GPT-3.5-generated, may contain errors).
- **Wuxia/xianxia/xuanhuan fan glossaries**: Immortal Mountain glossary; Wuxiaworld "General Glossary of Terms" — cultivation, jianghu/wulin, sects, martial-arts terms, tribulation, Dao/Qi, beast cores, and forms of address (师兄/师弟/师姐/师妹). High-value for historical drama.
- **Chinese honorifics & imperial titles**: Wikipedia "Chinese honorifics"/"Chinese titles"; fan glossaries (Spiderlily "Complete Glossary of Chinese Imperial Titles & Ranks," Spicy Chicken "Honorifics") — 陛下/殿下/娘娘/王爷/本宫, five noble ranks (公侯伯子男), the nine-rank official system, and Daoist/cultivation honorifics.
- **User-provided sources**: StudySmarter Chinese theatre terms; Reddit r/CDrama glossary; Wikipedia Chinese Internet slang; eChineseLearning "50 must-know cdrama phrases"; nyanovels historical terms glossary.
- **Parallel subtitle mining**: OpenSubtitles/OPUS (zh-en, zh-id, zh-ms, zh-th all available; Lison & Tiedemann 2016) to mine attested term pairs and example subtitle pairs; note noisy alignment and less-literal subtitle translations.
- **WMT24/25 subtitle dev data** and the WMT26 demo/test metadata glossaries.

**5.2 The zh→Indonesian gap.** Almost all glossaries are zh→en. Strategy:
- **Seed with CC-CIDICT** (cidict.org): "CC-CIDICT (over 124,000 entries) is a free, open-source Chinese–Indonesian dictionary… a direct derivative of the CC-CEDICT project" (Wikipedia "CEDICT"), CC-BY-SA 4.0, synced to CC-CEDICT — the single most valuable existing zh-id lexical resource.
- **English-pivot generation**: for each entry, generate Indonesian equivalents from the curated English equivalent(s) + Chinese figurative meaning via LLM, constrained by register/category, then verify. Verification = LLM cross-check + spot human review + back-translation consistency + OpenSubtitles zh-id / id-en attestation where available.
- **Indonesian fansub conventions**: mine how Indonesian cdrama communities render terms (e.g., "dunia persilatan" for 江湖, "Yang Mulia/Baginda" for imperial address) from Indonesian-subtitled cdrama (CPOP Home listings, community SRTs) and Glosbe zh-id parallel examples.
- Flag every zh-id entry with provenance and a confidence score (verified vs LLM-generated-unverified).

**5.3 Schema (JSON example).**
```json
{
  "id": 1421,
  "simplified": "江湖", "traditional": "江湖", "pinyin": "jiāng hú",
  "literal": "rivers and lakes",
  "figurative_zh": "旧时指四方各地闯荡谋生的社会；武侠中指武林社会",
  "figurative_en": "the world of martial artists operating outside mainstream society",
  "register": "historical",              // historical | modern | internet_slang | neutral
  "category": "wuxia_term",              // title_honorific | martial_arts | cultivation | chengyu | internet_slang | theatre | culture_item
  "en_equivalents": [
     {"text": "jianghu", "note": "often kept untranslated with gloss"},
     {"text": "the martial world", "note": "descriptive"}
  ],
  "id_equivalents": [
     {"text": "dunia persilatan", "confidence": "verified", "source": "fansub+CC-CIDICT"}
  ],
  "avoid": ["rivers and lakes (literal)"],
  "examples": [{"src": "他重出江湖。", "en": "He returned to the martial world.", "id": "Dia kembali ke dunia persilatan."}],
  "provenance": ["CC-CEDICT", "ImmortalMountain-glossary", "CC-CIDICT"],
  "license": "CC-BY-SA-4.0-compatible"
}
```

**5.4 Retrieval design.**
- **Exact/fuzzy string matching** for fixed expressions (chengyu, titles): jieba with all headwords loaded as a custom dictionary + trie/Aho-Corasick for multi-token spans; handle simplified/traditional normalization and multi-character overlap (longest-match preference).
- **Embedding retrieval** for variable/paraphrased phrasing: index entry (headword + figurative meaning) embeddings; top-k cosine at inference.
- **Latency**: precompute the jieba custom dict and an in-memory trie (µs lookups); cache embeddings; retrieve only for triggered lines; batch embedding queries. Per-line retrieval budget should stay well under the LLM post-edit cost.

**5.5 Construction workflow.**
1. **Scrape/parse** each source into a common intermediate (CC-CEDICT parser; HTML/table scrapers for glossaries; SRT aligners for OPUS).
2. **Normalize**: unify simplified/traditional, pinyin (numbered→diacritic), strip markup.
3. **Deduplicate/merge** by (simplified, pinyin) key; union English equivalents; keep the provenance list.
4. **LLM-assisted enrichment**: generate figurative meaning, register tag, category, and Indonesian equivalents (English-pivot); flag confidence.
5. **QC passes**: schema validation; human spot-check of high-frequency historical entries; back-translation consistency; reject low-confidence idiom/slang entries lacking two independent attestations.
6. **License/provenance tracking** per entry (WMT26 requires documenting data sources; note that CC-BY-NC-SA sources like PETCI/IdiomTranslate30 restrict commercial use — segregate them and confirm compatibility with the constrained-track/non-commercial research use).

**5.6 Size/effort estimates & prioritization.** Rough targets: CC-CEDICT backbone (~125k entries, free); chengyu subset (~8–9k high-value, PETCI+IdiomKB); wuxia/xianxia/cultivation terms (~500–1,500 curated); honorifics/imperial titles (~150–400); theatre terms (~100–200); internet slang (~300–800, volatile, needs periodic refresh). For HISTORICAL cdrama test coverage, prioritize in order: (1) titles/honorifics/forms-of-address, (2) cultivation/martial-arts/xianxia terms, (3) chengyu, (4) culture-specific items, then modern register & slang. Effort concentrates on zh→id verification and register/category tagging, not headword collection.

## Recommendations
1. **Stage 1 (fastest ROI):** Build the exact-match idiom/term DB from CC-CEDICT + PETCI + IdiomKB + wuxia/honorific glossaries; wire jieba+trie detection and a minimal-edit LLM post-editor prompt applied only to flagged lines. Benchmark term-recall and LLM-judge idiom accuracy on WMT24/25 dev data. Threshold to advance: term-recall gain >15 points and no COMET regression on non-idiom lines.
2. **Stage 2:** Add a COMETKiwi QE gate + candidate rerank accept/reject; add embedding retrieval for paraphrased idioms; add per-episode term memory for consistency. Threshold: over-edit (regression) rate <5%; gain-to-edit ratio positive.
3. **Stage 3 (zh→id):** Seed with CC-CIDICT, English-pivot-generate Indonesian equivalents, verify via LLM + fansub attestation + spot human review. Threshold: ≥90% of high-frequency historical entries have a verified Indonesian equivalent.
4. **Evaluation discipline:** Never gate idiom edits on BLEU alone; use term-recall + LLM-judge + human eval. Track subtitle CPL/CPS compliance as a hard constraint.
5. **Compliance:** Keep NC-licensed data (PETCI, IdiomTranslate30) segregated with provenance; document all sources for the system paper; ensure the final model is <20B params and release the weights.

## Caveats
- **Metric unreliability on idioms** (best correlation 0.386 full / 0.483 idiom; reference-free QE near-zero/negative, per IdiomEval) means automatic gains may under- or over-state real improvement; rely on LLM-judge + human eval.
- **IdiomKB meanings are LLM-generated** (GPT-3.5) and "may contain errors" (repo note); verify high-frequency entries.
- **IdiomKB's GPT-4 evaluator may favor GPT outputs** — the authors explicitly note "GPT-4 may exhibit a bias towards the translation generated by GPT models," and its human correlation is below 0.7.
- **APE can overcorrect**; the accept/reject gate is not optional.
- **zh→id equivalents are the weakest link** — low-resource, mostly pivot-generated; treat unverified entries as low-confidence.
- **Latency**: retrieval + a second LLM pass adds cost; keep post-editing selective to stay within a practical subtitle-pipeline budget.
- **Licensing**: PETCI and IdiomTranslate30 are CC-BY-NC-SA (non-commercial); confirm compatibility with the WMT26 constrained-track open-weight requirement before shipping derived data.