# tasks/selqa/entry_selqa.py
import os, sys, json, textwrap, subprocess, asyncio, re, math, hashlib
from collections import Counter
from typing import Any, Callable
from pathlib import Path
TASK_DIR = Path(__file__).resolve().parent           # .../tasks/selqa


REPO_ROOT = TASK_DIR.parent.parent  # os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from main import run_agent_loop, submit_answer_tool  # python_expression_tool :contentReference[oaicite:0]{index=0}

# Logging & phase flags (so the grader can verify usage)
PHASE = "calib"  # or "test"
LOG_CAL = TASK_DIR / "log_calib.jsonl"
LOG_TEST = TASK_DIR / "log_test.jsonl"

def _log(tool, payload):
    fp = LOG_CAL if PHASE == "calib" else LOG_TEST
    with open(fp, "a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": tool, "phase": PHASE, **payload}) + "\n")

def _as_texts(docs):
    # Accept str, list[str], list[dict], or mixed
    if isinstance(docs, str):
        return [docs]
    out = []
    for d in docs:
        if isinstance(d, str):
            out.append(d)
        elif isinstance(d, dict):
            out.append(d.get("text", json.dumps(d)))
        else:
            out.append(str(d))
    return out

def _read_jsonl(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            yield json.loads(line)

def guarded_submit(answer="done"):
    test_p = TASK_DIR / "test.jsonl"
    preds_p = TASK_DIR / "preds.jsonl"
    test_qids = [j["qid"] for j in _read_jsonl(test_p)]
    pred_qids = set()
    if preds_p.exists():
        for j in _read_jsonl(preds_p):
            q = j.get("qid")
            if q is not None:
                pred_qids.add(q)
    missing = [q for q in test_qids if q not in pred_qids]
    if missing:
        return {"ok": False, "missing": missing[:5], "message": f"still missing {len(missing)} qids"}
    return {"ok": True, "result": {"answer": answer}}

# ------- tiny retrieval/embedding utilities (deterministic) -------
def tokenize(s): return re.findall(r"[a-z0-9]+", s.lower())
def load_corpus():
    docs=[]; 
    # with open("tasks/selqa/corpus.jsonl") as f:
    with open(TASK_DIR / "corpus.jsonl", "r", encoding="utf-8") as f:
        for line in f: docs.append(json.loads(line))
    return docs

def bm25_rank(q, docs):
    qtok = tokenize(q); N=len(docs); df=Counter()
    for d in docs:
        for t in set(tokenize(d["text"])): df[t]+=1
    def score(d):
        toks = tokenize(d["text"]); tf=Counter(toks); dl=len(toks); avgdl=50.0; s=0.0
        for t in qtok:
            idf = math.log(1 + (N - df.get(t,0) + .5)/(df.get(t,0)+.5))
            tfw = (tf.get(t,0)*(1.2+1))/(tf.get(t,0)+1.2*(1-0.75+0.75*dl/avgdl))
            s += idf*tfw
        return s
    return sorted(docs, key=score, reverse=True)

DIM=64
def vec(text):
    v=[0.0]*DIM
    for t in tokenize(text):
        h=int(hashlib.md5(t.encode()).hexdigest(),16)%DIM; v[h]+=1.0
    n=math.sqrt(sum(x*x for x in v)) or 1.0
    return [x/n for x in v]
def cos(a,b): return sum(x*y for x,y in zip(a,b))

def embed_rank(q, docs):
    qv=vec(q); return sorted(docs, key=lambda d: cos(qv, vec(d["text"])), reverse=True)

def best_sentence_answer(q, docs):
    qset=set(tokenize(q)); best=("",-1)
    for text in docs:
        for sent in re.split(r"(?<=[.!?])\s+", text):
            ov=len(qset & set(tokenize(sent)))
            if ov>best[1]: best=(sent.strip(), ov)
    return best[0] or (docs[0] if docs else "")

# ------- wrap them as Anthropic tools compatible with main.py -------
def retrieve_bm25(q: str, k: int, qid: str): 
    docs=bm25_rank(q, load_corpus())[:k]
    toks=sum(len(tokenize(d["text"])) for d in docs)
    _log("retrieve_bm25", {"qid": qid, "k": k, "doc_ids": [d["doc_id"] for d in docs], "tokens": toks})
    return {"result": [d["text"] for d in docs], "tokens": toks, "qid": qid}

def retrieve_embed(q: str, k: int, qid: str):
    docs=embed_rank(q, load_corpus())[:k]
    toks=sum(len(tokenize(d["text"])) for d in docs)
    _log("retrieve_embed", {"qid": qid, "k": k, "doc_ids": [d["doc_id"] for d in docs], "tokens": toks})
    return {"result": [d["text"] for d in docs], "tokens": toks, "qid": qid}

def answer_with_reader(q: str, docs):
    texts = _as_texts(docs)
    return {"result": best_sentence_answer(q, texts)}

def embed(text: str): return {"result": vec(text)}
def lexical_score(a: str, b: str):
    # calibration-only tool; grader will forbid on test
    A,B=" ".join(tokenize(a))," ".join(tokenize(b))
    em = 1.0 if A==B else 0.0
    ta, tb = tokenize(a), tokenize(b)
    inter = sum((Counter(ta) & Counter(tb)).values())
    prec = inter/len(ta) if ta else 0.0; rec = inter/len(tb) if tb else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
    _log("lexical_score", {"em": em, "f1": f1})
    return {"result":{"em":em,"f1":f1}}

def count_tokens(texts_or_answer):
    def to_text(x):
        if isinstance(x, dict):
            return x.get("text", json.dumps(x))
        return str(x)
    if isinstance(texts_or_answer, list):
        n = sum(len(tokenize(to_text(t))) for t in texts_or_answer)
    else:
        n = len(tokenize(to_text(texts_or_answer)))
    return {"result": int(n)}

# Dataset IO tools (safe, task-scoped)
def load_train_calib():
    rows = []
    with open(TASK_DIR / "train_calib.jsonl", "r", encoding="utf-8") as f:
        for line in f: rows.append(json.loads(line))
    return {"result": rows}

def load_test():
    rows = []
    with open(TASK_DIR / "test.jsonl", "r", encoding="utf-8") as f:
        for line in f: rows.append(json.loads(line))
    return {"result": rows}

def write_calibration(obj: dict):
    with open(TASK_DIR / "calib.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return {"result": True}

def append_pred(qid: str, answer: str, decision: str, confidence: float, tokens_used: int):
    with open(TASK_DIR / "preds.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({
            "qid": qid, "answer": answer, "decision": decision,
            "confidence": float(confidence), "tokens_used": int(tokens_used)
        }) + "\n")
    return {"result": True}

def write_report(obj: dict):
    with open(TASK_DIR / "report.json", "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    return {"result": True}

# TOOLS = [
#   {"name":"python_expression","description":"Executes Python","input_schema":{"type":"object","properties":{"expression":{"type":"string"}},"required":["expression"]}},
#   {"name":"submit_answer","description":"Submit final","input_schema":{"type":"object","properties":{"answer":{}},"required":["answer"]}},
#   {"name":"retrieve_bm25","description":"BM25 retrieval","input_schema":{"type":"object","properties":{"q":{"type":"string"},"k":{"type":"integer"},"qid":{"type":"string"}},"required":["q","k","qid"]}},
#   {"name":"retrieve_embed","description":"Embedding retrieval","input_schema":{"type":"object","properties":{"q":{"type":"string"},"k":{"type":"integer"},"qid":{"type":"string"}},"required":["q","k","qid"]}},
#   {"name":"answer_with_reader","description":"Return best span/sentence","input_schema":{"type":"object","properties":{"q":{"type":"string"},"docs":{"type":"array"}},"required":["q","docs"]}},
#   {"name":"embed","description":"Return tiny embedding","input_schema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},
#   {"name":"lexical_score","description":"EM/F1 (calibration only)","input_schema":{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["a","b"]}},
# ]
# HANDLERS = {
#   "python_expression": python_expression_tool,
#   "submit_answer":     submit_answer_tool,
#   "retrieve_bm25":     retrieve_bm25,
#   "retrieve_embed":    retrieve_embed,
#   "answer_with_reader":answer_with_reader,
#   "embed":             embed,
#   "lexical_score":     lexical_score,
# }

TOOLS = [
  {"name":"retrieve_bm25","description":"BM25 retrieval",
   "input_schema":{"type":"object","properties":{"q":{"type":"string"},"k":{"type":"integer"},"qid":{"type":"string"}},"required":["q","k","qid"]}},
  {"name":"retrieve_embed","description":"Embedding retrieval",
   "input_schema":{"type":"object","properties":{"q":{"type":"string"},"k":{"type":"integer"},"qid":{"type":"string"}},"required":["q","k","qid"]}},
  {"name":"answer_with_reader","description":"Return best sentence",
   "input_schema":{"type":"object","properties":{"q":{"type":"string"},"docs":{"type":"array"}},"required":["q","docs"]}},
  {"name":"embed","description":"Tiny embedding vector",
   "input_schema":{"type":"object","properties":{"text":{"type":"string"}},"required":["text"]}},
  {"name":"lexical_score","description":"EM/F1 (calibration only)",
   "input_schema":{"type":"object","properties":{"a":{"type":"string"},"b":{"type":"string"}},"required":["a","b"]}},
  {"name":"count_tokens","description":"Rough token count",
   "input_schema":{"type":"object","properties":{"texts_or_answer":{}},"required":["texts_or_answer"]}},

  # dataset IO
  {"name":"load_train_calib","description":"Read calibration set","input_schema":{"type":"object","properties":{}}}, 
  {"name":"load_test","description":"Read test set","input_schema":{"type":"object","properties":{}}}, 
  {"name":"write_calibration","description":"Save calib mapping","input_schema":{"type":"object","properties":{"obj":{"type":"object"}},"required":["obj"]}},
  {"name":"append_pred","description":"Append a prediction line",
   "input_schema":{"type":"object","properties":{
        "qid":{"type":"string"},"answer":{"type":"string"},"decision":{"type":"string"},
        "confidence":{"type":"number"},"tokens_used":{"type":"integer"}},"required":["qid","answer","decision","confidence","tokens_used"]}},
  {"name":"write_report","description":"Save metrics","input_schema":{"type":"object","properties":{"obj":{"type":"object"}},"required":["obj"]}},

  # keep submit tool from main.py
  {"name":"submit_answer","description":"Submit final answer",
   "input_schema":{"type":"object","properties":{"answer":{}},"required":["answer"]}},
]

HANDLERS = {
  "retrieve_bm25": retrieve_bm25,
  "retrieve_embed": retrieve_embed,
  "answer_with_reader": answer_with_reader,
  "embed": embed,
  "lexical_score": lexical_score,
  "count_tokens": count_tokens,
  "load_train_calib": load_train_calib,
  "load_test": load_test,
  "write_calibration": write_calibration,
  "append_pred": append_pred,
  "write_report": write_report,
  "submit_answer": guarded_submit, #submit_answer_tool,   # from main.py
}


# CAL_PROMPT = textwrap.dedent("""
# Selective QA (calibration).
# Use BOTH retrieval modalities (retrieve_bm25 & retrieve_embed), build a confidence
# combining lexical+embedding evidence, calibrate a threshold on tasks/selqa/train_calib.jsonl,
# save it to tasks/selqa/calib.json, then submit.
# """).strip()

# CAL_PROMPT = """
# Selective QA (calibration phase).

# Files you can read:
# - tasks/selqa/train_calib.jsonl  (each: {"qid","question","answers":[...]})
# - tasks/selqa/corpus.jsonl       (each: {"doc_id","text"})

# TOOLS you must use:
# - retrieve_bm25(q,k,qid) and retrieve_embed(q,k,qid)  ← use BOTH modalities
# - answer_with_reader(q, docs_texts)
# - embed(text)
# - lexical_score(a,b)            ← allowed ONLY during calibration
# - count_tokens(x)
# - load_train_calib()
# - write_calibration(obj)        ← use this to SAVE your learned mapping/threshold
# - submit_answer(answer)         ← call this when done

# Goal:
# 1) Build a confidence function that combines lexical + embedding evidence.
# 2) Calibrate a decision threshold τ on the calibration split so that, on a held-out test,
#    we can target risk ≤ 0.15 and coverage ≥ 0.80 with a mean token budget ≤ 300.
# 3) SAVE your learned parameters by calling write_calibration({...}) with a JSON
#    object like: {"threshold": τ, "weights": {...}, "notes":"..."} to save your parameters to tasks/selqa/calib.json.
# 4) Finally, call submit_answer("done").

# Do NOT attempt to write files directly; use write_calibration().
# """.strip()

CAL_PROMPT = """
Selective QA — CALIBRATION PHASE

Files you can read:
- tasks/selqa/train_calib.jsonl   (each line: {"qid","question","answers":[...]} )
- tasks/selqa/corpus.jsonl        (each line: {"doc_id","text"})

TOOLS you must use:
- load_train_calib()
- retrieve_bm25(q, k, qid)    -> use this AND retrieve_embed for EVERY question
- retrieve_embed(q, k, qid)
- answer_with_reader(q, docs_texts)
- embed(text)
- lexical_score(a, b)         -> allowed ONLY during calibration
- count_tokens(x)
- write_calibration(obj)      -> SAVE your learned mapping/threshold to calib.json
- submit_answer(answer)       -> call at the very end

Goal:
1) Build a confidence function that combines lexical + embedding evidence (e.g., overlap/similarity
   from BM25 and embedding retrieval; agreement between top hits; cosine(embed(answer), embed(doc))).
2) Calibrate a decision threshold τ (0..1) on the calibration split so that, on held-out test, the policy
   can target risk ≤ 0.15 and coverage ≥ 0.80 with mean tokens ≤ 300.
3) SAVE your learned parameters by calling:
   write_calibration({
     "threshold": τ,
     "weights": {...},        # any feature weights or mapping params you learned
     "notes": "how confidence is computed"
   })
   This must create: tasks/selqa/calib.json
4) Finally, call submit_answer("done").

Important:
- Use BOTH retrieval modalities (BM25 AND embedding) for EVERY calibration question.
- You MAY use lexical_score(a,b) here (since gold answers are available), but you MUST NOT rely
  on lexical_score at test time.
- Do NOT write files directly; always use write_calibration(...).
""".strip()



# TEST_PROMPT = textwrap.dedent("""
# Selective QA (test).
# Load tasks/selqa/calib.json; for each item in tasks/selqa/test.jsonl, retrieve with BOTH
# modalities, answer with answer_with_reader, compute confidence, then write
# tasks/selqa/preds.jsonl lines:
# {"qid":..., "answer":"...", "decision":"answer|abstain", "confidence": float, "tokens_used": int}
# Also write tasks/selqa/report.json with EM, F1, risk, coverage, ECE, mean_tokens, then submit.
# DO NOT call lexical_score on test.
# """).strip()

TEST_PROMPT = textwrap.dedent("""
Selective QA — TEST PHASE

You already calibrated on train_calib.jsonl and saved a threshold/mapping to:
  tasks/selqa/calib.json
DO NOT modify that file in test.

Available files to READ:
- tasks/selqa/test.jsonl      (each line: {"qid","question"})
- tasks/selqa/corpus.jsonl    (each line: {"doc_id","text"})
- tasks/selqa/calib.json      (your locked parameters from calibration)

TOOLS you MUST use in test:
- load_test()                                       -> returns test items
- retrieve_bm25(q, k, qid)                          -> texts of top-k BM25 docs (also logged)
- retrieve_embed(q, k, qid)                         -> texts of top-k embedding docs (also logged)
- answer_with_reader(q, docs_texts)                 -> short extractive answer string
- embed(text)                                       -> 64-d vector (for cosine sims)
- count_tokens(x)                                   -> token count for text or list of texts
- append_pred(qid, answer, decision, confidence, tokens_used)
- write_report(obj)
- submit_answer(answer)                             -> call at the very end

IMPORTANT RULES:
1) Use BOTH retrieval modalities (BM25 AND embedding) for EVERY question.
2) DO NOT call lexical_score on test.
3) Respect the locked calibration: compute a confidence in [0,1] and decide
   "answer" if confidence >= τ (τ read from calib.json), else "abstain".
4) Control cost: keep mean tokens/query ≤ 300. Tokens = count_tokens(retrieved_texts) + count_tokens(answer).
   Use small k (e.g., k=2..3) and adapt downwards when evidence is strong.

WHAT TO DO FOR EACH TEST QUESTION:
A) Call retrieve_bm25(q, k, qid) and retrieve_embed(q, k, qid). You may set k=3 initially.
B) Build evidence and a confidence score using BOTH modalities, for example:
   - cosine(embed(answer), embed(top_doc_text_from_each modality))
   - agreement/overlap between the two retrieved sets (e.g., shared facts/phrases)
   - optional heuristics such as the similarity of the question to the selected answer sentence
C) Generate an answer with answer_with_reader(q, merged_docs_texts).
D) Compute confidence in [0,1], then decision = "answer" if confidence >= τ else "abstain".
E) Compute tokens_used = count_tokens(all_retrieved_texts) + count_tokens(answer).
F) Call append_pred(qid, answer, decision, confidence, tokens_used).

AFTER ALL QUESTIONS:
- Compute a brief summary object with fields:
  {"summary": "test run",
   "notes": "selective QA with answer/abstain; both modalities used; budgeted retrieval",
   "estimated_mean_tokens": <float>, "estimated_coverage": <float>}
  (Exact EM/F1/risk/ECE will be evaluated by the grader; just include your own estimates.)
- Call write_report(summary_object).
- Finally, call submit_answer("done").

Constraints target (used by the grader): risk ≤ 0.15, coverage ≥ 0.80, mean tokens ≤ 300.
Do not write files directly; ALWAYS use append_pred(...) and write_report(...).
""").strip()

async def run_test_host_loop():
    # ensure we are in test phase for logging
    global PHASE
    PHASE = "test"

    # read locked calibration
    cal = json.loads((TASK_DIR / "calib.json").read_text(encoding="utf-8"))
    tau = float(cal.get("threshold", 0.7))

    # load test items
    items = []
    with open(TASK_DIR / "test.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            items.append(json.loads(line))

    # start fresh preds for this run
    try: (TASK_DIR / "preds.jsonl").unlink()
    except: pass

    # run once per qid
    for i, it in enumerate(items, 1):
        qid = it["qid"]
        q   = it["question"]

        per_q_prompt = f"""
            Selective QA — TEST (single item)

            QID: {qid}
            Question: {q}

            Locked threshold τ = {tau:.3f}

            TOOLS to use (in order):
            1) retrieve_bm25(q="{q}", k=3, qid="{qid}")
            2) retrieve_embed(q="{q}", k=3, qid="{qid}")
            3) answer_with_reader(q="{q}", docs_texts=...)  # synthesize a short answer sentence
            4) embed(...), count_tokens(...) as needed
            5) Compute confidence in [0,1]; decision="answer" if confidence ≥ τ else "abstain".
            6) append_pred(qid="{qid}", answer=..., decision=..., confidence=..., tokens_used=...)

            Rules:
            - Use BOTH retrieval modalities for THIS qid.
            - DO NOT call lexical_score on test.
            - DO NOT submit yet; handle exactly ONE qid in this turn.
            """.strip()

        await run_agent_loop(per_q_prompt, TOOLS, HANDLERS, max_steps=13, verbose=False)

    # finalize (optional report + submit)
    finalize_prompt = """
            All test items have been processed. Call write_report({
            "summary": "host-driven selective QA",
            "estimated_coverage": 0.0,
            "estimated_mean_tokens": 0.0
            }) then submit_answer("done").
            """.strip()

    await run_agent_loop(finalize_prompt, TOOLS, HANDLERS, max_steps=6, verbose=False)



async def run_once():
    # fresh data
    #subprocess.check_call([sys.executable, "tasks/selqa/make_data.py"])

    # fresh data - run generator with cwd=TASK_DIR so files land in tasks/selqa/
    subprocess.check_call(
        [sys.executable, str(TASK_DIR / "make_data.py")],
        cwd=str(TASK_DIR)
    )
    # clear old logs/outputs (optional but tidy)
    for p in ["log_calib.jsonl","log_test.jsonl","preds.jsonl","report.json","calib.json"]:
        try: (TASK_DIR / p).unlink()
        except: pass


    # calibration -> test (two-phase run) using main.py’s loop
    global PHASE
    PHASE = "calib"
    await run_agent_loop(CAL_PROMPT, TOOLS, HANDLERS, max_steps=20, verbose=False)
    print("=== CAL LOG ===")
    try:
        print((TASK_DIR / "log_calib.jsonl").read_text(encoding="utf-8")[:2000])
    except Exception as e:
        print("no log", e)

    # ensure calib.json exists (fallback while iterating)
    calib_path = TASK_DIR / "calib.json"
    print("[debug] calib.json exists after calibration? ", calib_path.exists(), str(calib_path))
    if not calib_path.exists():
        calib_path.write_text(
            json.dumps({"threshold": 0.7, "weights": {}, "meta": "fallback"}, indent=2),
            encoding="utf-8"
        )
        print("[debug] wrote fallback calib.json")


    print("=== CAL LOG ===")
    cal_log = TASK_DIR / "log_calib.jsonl"
    if cal_log.exists():
        try:
            # show first ~2KB to keep terminal tidy
            print(cal_log.read_text(encoding="utf-8")[:2000])
        except Exception as e:
            print("(could not read calib log)", str(e))
    else:
        print("(no calibration tools called yet — this is OK)")

    # test_count = sum(1 for _ in open(TASK_DIR / "test.jsonl", "r", encoding="utf-8"))
    # steps_per_q = 6        # conservative
    # extra = 20             # for write_report + submit + slack
    # test_steps = steps_per_q * test_count + extra

    
    PHASE = "test"
    # await run_agent_loop(TEST_PROMPT, TOOLS, HANDLERS, max_steps=test_steps, verbose=False)
    await run_test_host_loop()

    try:
        test_n  = sum(1 for _ in open(TASK_DIR / "test.jsonl", "r", encoding="utf-8"))
        preds_n = sum(1 for _ in open(TASK_DIR / "preds.jsonl", "r", encoding="utf-8"))
        print(f"[debug] test items: {test_n}, preds written: {preds_n}")
    except Exception as _e:
        pass


    # grade
    #out = subprocess.check_output([sys.executable, "tasks/selqa/grade.py"])
    out = subprocess.check_output([sys.executable, str(REPO_ROOT / "tasks" / "selqa" / "grade.py")], cwd=str(REPO_ROOT))
    print(out.decode().strip())
    res = json.loads(out.decode())
    return res

    """ 

    try - the above block, if necessary to catch errors wih missing :
    except subprocess.CalledProcessError as e:
        msg = e.output.decode(errors="ignore") if e.output else str(e)
        res = {"passed": False, "reasons": [f"grader/child process failed: {msg}"]}
        print(json.dumps(res, indent=2))
        return res

    except Exception as e:
        res = {"passed": False, "reasons": [f"run_once exception: {type(e).__name__}: {e}"]}
        print(json.dumps(res, indent=2))
        return res
    """



async def amain():
    runs = int(os.environ.get("RUNS", "10"))
    passes = 0
    results = []
    for i in range(runs):
        print(f"\n=== RUN {i+1}/{runs} ===")
        res = await run_once()  # run_once is async
        results.append(res)
        if res.get("passed"):
            passes += 1

    pass_rate = round((passes / runs) * 100.0, 1)
    summary = {"runs": runs, "passed": passes, "pass_rate_pct": pass_rate}
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    (TASK_DIR / "runs_summary.json").write_text(
        json.dumps({**summary, "examples": results[:5]}, indent=2),
        encoding="utf-8"
    )

if __name__ == "__main__":
    asyncio.run(amain())
    # runs = int(os.environ.get("RUNS","10"))
    # asyncio.run(asyncio.gather(*[run_once() for _ in range(runs)]))
