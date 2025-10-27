hello-py
===

Setup instructions:

1. Clone the repository:
   ```
   git clone https://github.com/preferencemodel/hello-py.git
   ```

2. Navigate to the project directory:
   ```
   cd hello-py
   ```

3. Set up `ANTHROPIC_API_KEY` environment variable:
   ```
   export ANTHROPIC_API_KEY=your_api_key_here
   ```

4. Run the agent:
   ```
   uv run main.py
   ```

## Execution Modes

The test suite supports both concurrent and sequential execution. 

To change modes, edit the `concurrent` parameter at the bottom of `main.py`:

```python
asyncio.run(main(concurrent=True))
asyncio.run(main(concurrent=False))
```

When running concurrently, results print as they complete (not in run order) for faster overall execution.


# Usecase: Risk-Controlled Selective QA (Answer-or-Abstain) for RAG

## What this is
A minimal but complete selective QA system that answers questions or abstains to meet a risk (error-rate) constraint under a token budget, using both lexical (BM25) and semantic (embedding) retrieval. It demonstrates:

 - * Measurement discipline: risk@coverage, ECE (calibration), EM/F1 on answered subset, token cost

 - * Policy learning: calibrate a score→probability mapping and pick a threshold τ to decide answer vs. abstain

 - * Budgeting: keep mean tokens per query under a target B by controlling retrieval k and answer length

 - * Repeatability: calibration on train_calib.jsonl, frozen parameters on test.jsonl

 - * This pattern is used in production search/assistants, safety triage, support, medical pre-screening—anywhere “don’t guess” matters.

## Sample Run Output

#### *Increasing variants_per_fact in the make_data.py file in the make_qa(variants_per_fact= ??) method, typically increases the test set size (unless capped), which can improve pass rate under fixed constraints because the allowed number of errors grows with N_test (≈ floor(R * C * N_test)). This makes meeting risk ≤ R at coverage ≥ C more forgiving. However, this also increases cost, and if the true error rate exceeds R, a larger test set can reduce pass rate by revealing that gap more reliably.* 
```
=== RUN 1/10 ===
=== CAL LOG ===
{"tool": "retrieve_bm25", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d7", "d6", "d16", "d2", "d4"], "tokens": 41}
{"tool": "retrieve_embed", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d6", "d16", "d2", "d7", "d4"], "tokens": 41}
{"tool": "lexical_score", "phase": "calib", "em": 0.0, "f1": 0.18181818181818182}

[debug] calib.json exists after calibration?  True C:\Users\vipul\Documents\GitHub\Reinforcement Learning\hello-py\tasks\selqa\calib.json
=== CAL LOG ===
{"tool": "retrieve_bm25", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d7", "d6", "d16", "d2", "d4"], "tokens": 41}
{"tool": "retrieve_embed", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d6", "d16", "d2", "d7", "d4"], "tokens": 41}
{"tool": "lexical_score", "phase": "calib", "em": 0.0, "f1": 0.18181818181818182}

[debug] test items: 20, preds written: 20
{
  "passed": true,
  "reasons": [],
  "coverage": 1.0,
  "risk": 0.09999999999999998,
  "ece": 0.10750000000000004,
  "mean_tokens": 52.75,
  "answered": 20,
  "total": 20,
  "em_answered": 0.9,
  "f1_answered": 0.9
}

=== RUN 2/10 ===
=== CAL LOG ===
{"tool": "retrieve_bm25", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d7", "d6", "d16", "d2", "d4"], "tokens": 41}
{"tool": "retrieve_embed", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d6", "d16", "d2", "d7", "d4"], "tokens": 41}

[debug] calib.json exists after calibration?  True C:\Users\vipul\Documents\GitHub\Reinforcement Learning\hello-py\tasks\selqa\calib.json
=== CAL LOG ===
{"tool": "retrieve_bm25", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d7", "d6", "d16", "d2", "d4"], "tokens": 41}
{"tool": "retrieve_embed", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d6", "d16", "d2", "d7", "d4"], "tokens": 41}

[debug] test items: 20, preds written: 20
{
  "passed": true,
  "reasons": [],
  "coverage": 1.0,
  "risk": 0.09999999999999998,
  "ece": 0.1347,
  "mean_tokens": 52.2,
  "answered": 20,
  "total": 20,
  "em_answered": 0.9,
  "f1_answered": 0.9
}

=== RUN 3/10 ===
=== CAL LOG ===
{"tool": "retrieve_bm25", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d7", "d6", "d16", "d2", "d4"], "tokens": 41}
{"tool": "retrieve_embed", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d6", "d16", "d2", "d7", "d4"], "tokens": 41}
{"tool": "lexical_score", "phase": "calib", "em": 0.0, "f1": 0.18181818181818182}

[debug] calib.json exists after calibration?  False C:\Users\vipul\Documents\GitHub\Reinforcement Learning\hello-py\tasks\selqa\calib.json
[debug] wrote fallback calib.json
=== CAL LOG ===
{"tool": "retrieve_bm25", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d7", "d6", "d16", "d2", "d4"], "tokens": 41}
{"tool": "retrieve_embed", "phase": "calib", "qid": "q13", "k": 5, "doc_ids": ["d6", "d16", "d2", "d7", "d4"], "tokens": 41}
{"tool": "lexical_score", "phase": "calib", "em": 0.0, "f1": 0.18181818181818182}

[debug] test items: 20, preds written: 20
{
  "passed": false,
  "reasons": [
    "risk 0.16 > 0.15"
  ],
  "coverage": 0.95,
  "risk": 0.1578947368421053,
  "ece": 0.10526315789473678,
  "mean_tokens": 52.3,
  "answered": 19,
  "total": 20,
  "em_answered": 0.8421052631578947,
  "f1_answered": 0.8421052631578947
}
```

## Repo Layout:
```
<repo-root>/
  main.py                 # generic agent loop (Anthropic)
  README.md
  tasks/
    selqa/
      entry_selqa.py      # task driver (imports run_agent_loop from main.py)
      make_data.py        # synthetic corpus & splits
      grade.py            # strict, per-qid grader
      # runtime artifacts:
      corpus.jsonl
      train_calib.jsonl
      test.jsonl
      calib.json
      preds.jsonl
      report.json
      log_calib.jsonl
      log_test.jsonl
      runs_summary.json   # pass-rate summary (when running multiple times)
```
Imports are path-safe: entry_selqa.py anchors to TASK_DIR and REPO_ROOT so you can run from anywhere.


---

## Use case & design

**Goal.** Build a policy that, for each question:

1) retrieves context via **BM25** and **embeddings** (both are required),  
2) generates an answer + a **confidence**,  
3) **answers or abstains** to meet constraints:
  - **Risk** (error-rate on answered subset) ≤ **R** (e.g., 0.15)
  - **Coverage** (fraction answered) ≥ **C** (e.g., 0.80)
  - **Mean tokens/query** ≤ **B** (budget; e.g., 300)

**Calibration.** On `train_calib.jsonl`, learn a score→probability mapping and choose a confidence threshold **τ** to meet (R, C, B). Save to `tasks/selqa/calib.json`. Lock it for test.

**Confidence features (examples).**
- Cosine similarity between `embed(answer)` and top docs from both BM25 & embedding retrieval.
- Agreement between the two modalities (overlap in top-k facts).
- Question–answer lexical overlap and simple reader span scores.

**Policy.** Answer if `p(correct) ≥ τ`, else **abstain**.

**Cost control.** Keep `k` small (2–3), adapt downwards when agreement is strong. Tokens = retrieval context + answer.

---

## How it runs (end-to-end)

1. **Generate data** (`make_data.py`):
   - `corpus.jsonl` – small synthetic knowledge base.
   - `train_calib.jsonl` – calibration questions **with** golds.
   - `test.jsonl` – test questions (no golds in file).

2. **Calibration phase** (`entry_selqa.py`):
   - Use **both** retrievals + `lexical_score` (allowed only here) to build and calibrate confidence.
   - Save `calib.json` (contains at least `"threshold": τ`).

3. **Test phase**:
   - For **each** test QID, use **both** retrieval modalities, produce a short answer with the reader, compute confidence, then **answer/abstain**.
   - Append one line per QID to `preds.jsonl`:
     ```json
     {"qid":"...", "answer":"...", "decision":"answer|abstain", "confidence": 0.0-1.0, "tokens_used": int}
     ```
   - Write a short `report.json` (notes/estimates; grader recomputes metrics).

4. **Grading** (`grade.py`):
   - Validates usage: **both** retrieval tools were used **per QID** on **test**; **no** `lexical_score` on test.
   - Requires exactly **one prediction per test QID**.
   - Computes **coverage**, **risk** (error-rate on answered subset), **ECE** (calibration), and **mean tokens** across all test items.
   - Enforces pass condition: `coverage ≥ C` **AND** `risk ≤ R` **AND** `mean_tokens ≤ B`.

---

## Requirements

- Python 3.10+  
- [uv](https://github.com/astral-sh/uv) (for env + run)  
- Anthropic API key set as `ANTHROPIC_API_KEY`

---

## Quickstart (Windows, VS Code – Command Prompt)

1) **Open terminal** → **Command Prompt** in VS Code.

2) **cd to repo root** (quote path if it has spaces):

cd "C:\Users\<you>\Documents\GitHub\Reinforcement Learning\hello-py"


3) Set your Anthropic API key:

- In a .env file 
or
- set ANTHROPIC_API_KEY=sk-ant-...your-key...

4) Run one pass:
```
cmd:

uv run python tasks\selqa\entry_selqa.py

or 

set RUNS=20 && uv run python ...  # You can set Runs according to your desire.


## You’ll see a JSON summary like:

{
  "passed": false,
  "reasons": ["risk 0.23 > 0.15"],
  "coverage": 0.83,
  "risk": 0.23,
  "ece": 0.08,
  "mean_tokens": 205.4,
  "answered": 15,
  "total": 18
}
```
## Measuring pass-rate

The runner aggregates multiple runs and writes tasks/selqa/runs_summary.json.
```
Expected final summary:
{
  "runs": 20,
  "passed": 6,
  "pass_rate_pct": 30.0
}
```
- Tune R/C/B or dataset difficulty to get better outcome.

## Design choices (what we implemented)

- Dual retrieval is mandatory: both BM25 and embedding are used per qid on test (enforced by grader via log_test.jsonl).

- Confidence uses lexical/semantic signals, e.g.:

- * cosine between embed(answer) and top doc(s) from each modality

- * agreement/overlap between BM25 and embedding hits

- * lightweight question–answer overlap

- * (calibration may use lexical_score but test may not)

- Calibration: learn/choose τ on train_calib.jsonl; saved to calib.json; frozen for test

- Cost control: prefer small k (2–3); token budget includes retrieved texts + answer length

- Guarded submit: submit_answer only succeeds when all test qids have a preds.jsonl line

- Strict grader:

- * per-qid retrieval usage checks

- * coverage over all test qids

- * mean tokens over all test qids

- * correctness proxy via corpus normalization (synthetic but deterministic)

- * pass if coverage ≥ C, risk ≤ R, mean_tokens ≤ B

## Implementation details

- ### Agent loop (main.py)

- * Emits tool_result blocks for every tool call; never appends empty message content (prevents API errors).

- * submit_answer is guarded: it only succeeds when every test QID has a preds.jsonl line (prevents early exit).

- ### Dual-retrieval enforcement

- * entry_selqa.py logs both retrieve_bm25 and retrieve_embed usage per QID into log_test.jsonl.

- * grade.py requires both tools to appear for each QID on test.

- ### Calibration vs Test

- * lexical_score is permitted only in calibration (uses golds from train_calib.jsonl).

- * Threshold τ is saved to calib.json and locked for test.

- ### Cost accounting

- * Retrieval tokens (from logs) + answer tokens (approx. by whitespace tokens via count_tokens).

- ### Evaluation

- * coverage = answered / total_test_qids

- * risk = 1 − accuracy_on_answered

- * ece computed over answered items

- * mean_tokens averaged over all test QIDs

## Tuning knobs

- Constraints (grade.py): R (risk), C (coverage), B (token budget)

- Retrieval k: lower k saves tokens; combined with modality agreement to modulate confidence

- Calibration size: larger calibration → more stable τ

- Question variants (make_data.py): increase test size & difficulty

- Steps: test loop uses either host-driven per-qid loop or a single prompt with adaptive max_steps

## Troubleshooting

- ### “Missing calib.json”
Ensure calibration prompt explicitly calls write_calibration({...}). We also guard with a tiny fallback right after calibration (development only).

- ### “preds.jsonl missing X test qids” / coverage ≈ 0.05
The LLM stopped early. We fix this with a guarded submit and/or a host-driven per-qid test loop that forces one append_pred(...) per qid.

- ### Anthropic 400: messages.* must have non-empty content
The loop must not append an empty "content": []. We patch run_agent_loop to only append tool results when non-empty and to always wrap handler calls with try/except.

- ### Paths
Grader runs from repo root so relative paths like tasks/selqa/calib.json resolve. entry_selqa.py uses cwd correctly when calling scripts.

### Example outputs

*preds.jsonl (one per qid):*

{"qid":"q15","answer":"Mount Everest is the highest mountain on Earth.","decision":"answer","confidence":0.84,"tokens_used":172}
{"qid":"q32","answer":"","decision":"abstain","confidence":0.28,"tokens_used":95}
...

*report.json:*
```
{
  "summary": "selective QA test run",
  "estimated_coverage": 0.8,
  "estimated_mean_tokens": 210.0,
  "notes": "dual retrieval; budgeted k; answer-or-abstain"
}
```
Final grader printout:
```
{
  "passed": true,
  "reasons": [],
  "coverage": 0.81,
  "risk": 0.13,
  "ece": 0.07,
  "mean_tokens": 205.4,
  "answered": 17,
  "total": 21,
  "em_answered": 0.76,
  "f1_answered": 0.76
}

```
## Why this matters


This project shows how to operationalize uncertainty in RAG: calibrate confidence, make principled answer/abstain decisions, and verify with tight, per-qid evaluation under realistic cost constraints. The pattern generalizes to many production setups (search, ads, safety, support, medical triage).


