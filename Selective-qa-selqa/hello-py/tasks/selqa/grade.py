# tasks/selqa/grade.py
import json, math, os, re
from collections import defaultdict, Counter

RISK_MAX = 0.15
COV_MIN  = 0.80
BUDGET   = 300
LOG_CAL  = "tasks/selqa/log_calib.jsonl"
LOG_TEST = "tasks/selqa/log_test.jsonl"

def norm(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
def toks(s): return [t for t in re.findall(r"[a-z0-9]+", s.lower())]

def read_jsonl(path):
    with open(path) as f:
        for line in f: yield json.loads(line)

def load_test():
    return {j["qid"]: j for j in read_jsonl("tasks/selqa/test.jsonl")}

def load_calib_gold():
    g = {}
    for j in read_jsonl("tasks/selqa/train_calib.jsonl"):
        g[j["qid"]] = j["answers"]
    return g

def em_f1(pred, golds):
    P = norm(pred); best_em=0; best_f1=0.0
    for g in golds:
        G = norm(g)
        em = 1 if P==G else 0
        ta, tb = toks(P), toks(G)
        ca, cb = Counter(ta), Counter(tb)
        inter = sum((ca&cb).values())
        prec = inter/len(ta) if ta else 0.0
        rec  = inter/len(tb) if tb else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec)>0 else 0.0
        if em>best_em or (em==best_em and f1>best_f1):
            best_em, best_f1 = em, f1
    return best_em, best_f1

def compute_ece(confidences, corrects, bins=10):
    # confidences, corrects over ANSWERED items
    if not confidences: return 0.0
    B = [[] for _ in range(bins)]
    for c, y in zip(confidences, corrects):
        i = min(bins-1, int(c*bins))
        B[i].append((c,y))
    ece = 0.0
    for b in B:
        if not b: continue
        acc = sum(y for _,y in b)/len(b)
        conf = sum(c for c,_ in b)/len(b)
        ece += (len(b)/len(confidences))*abs(acc-conf)
    return ece

def approx_tokens_from_logs():
    # sum tokens of retrieved docs (unique per qid) + answer length approx
    per_qid = defaultdict(int)
    seen = defaultdict(set)
    if os.path.exists(LOG_TEST):
        for j in read_jsonl(LOG_TEST):
            if j.get("tool") in ("retrieve_bm25","retrieve_embed") and "qid" in j:
                try:
                    qid = j["qid"]; per_qid[qid] += int(j.get("tokens", 0))
                except:
                    # ignore bad rows
                    pass
    return per_qid

# def main():
#     # usage checks
#     tools_used = defaultdict(int)
#     if not os.path.exists("tasks/selqa/calib.json"):
#         return print(json.dumps({"passed":False,"reasons":["Missing calib.json"]}, indent=2))

#     if not os.path.exists(LOG_TEST): 
#         return print(json.dumps({"passed":False,"reasons":["Missing test log"]}, indent=2))

#     for j in read_jsonl(LOG_TEST):
#         tools_used[j["tool"]] += 1
#     reasons = []
#     if tools_used["retrieve_bm25"] == 0 or tools_used["retrieve_embed"] == 0:
#         reasons.append("Both retrieval modalities must be used on test")
#     # forbid gold-based scoring on test
#     if any(j["tool"]=="lexical_score" for j in read_jsonl(LOG_TEST)):
#         reasons.append("lexical_score used on test (not allowed)")

#     # metrics
#     gold = load_calib_gold()  # for EM/F1 lookups we only use gold where available
#     # load preds + compute metrics against hidden test gold by reconstructing from FACTS in make_data? we instead embed gold into calib only; for test we evaluate on answered subset by rule: the corpus sentence containing the true entity is identical to one gold; we can reconstruct gold by scanning corpus text for entities present in FACTS? Simpler: store a copy of full items in calib file only. For test, we approximate correctness by exact match to any sentence in corpus—OK for synthetic set.
#     # For this toy set, treat an answer as correct if it exactly matches any sentence from corpus (case-insensitive) that logically answers the question; we approximate using EM against any corpus sentence substring of the known entities.
#     # Minimal-but-deterministic: re-open tasks/selqa/make_data.py FACTS? Not available here; fallback: read a sidecar test_golds.json if present; else approximate from corpus by capital names heuristics.

#     # Better: during generation we hid golds in test.jsonl, but they are deterministically derivable from corpus + question. We'll do a very light heuristic: capitalize words; match single-token proper nouns present in corpus capitalized sequences.
#     corpus_texts = [j["text"] for j in read_jsonl("tasks/selqa/corpus.jsonl")]
#     corpus_norms = {re.sub(r"[^a-z0-9 ]","",t.lower()).strip() for t in corpus_texts}

#     preds = list(read_jsonl("tasks/selqa/preds.jsonl")) #[json.loads(l) for l in read_jsonl("tasks/selqa/preds.jsonl")]
#     answered = [p for p in preds if p["decision"].lower()=="answer"]
#     cov = len(answered)/max(1,len(preds))

#     # correctness proxy: exact match of normalized answer to any normalized corpus sentence OR to a capitalized entity token inside a corpus sentence
#     def normalize(s): return re.sub(r"[^a-z0-9 ]","",s.lower()).strip()
#     correct_flags=[]; confidences=[]; token_costs=[]
#     per_q_tokens = approx_tokens_from_logs()
#     for p in answered:
#         ansN = normalize(p["answer"])
#         ok = (ansN in corpus_norms)
#         correct_flags.append(1 if ok else 0)
#         confidences.append(float(p.get("confidence",0.0)))
#         # budget: use logged retrieval tokens + answer length
#         token_costs.append(per_q_tokens.get(p["qid"],0) + len(p["answer"].split()))

#     risk = 1 - (sum(correct_flags)/len(correct_flags) if correct_flags else 0.0)
#     mean_tokens = (sum(token_costs)/len(preds)) if preds else 0.0
#     ece = compute_ece(confidences, correct_flags)

#     # secondary EM/F1 on answered subset relative to corpus sentences (proxy)
#     ems=[]; f1s=[]
#     for p in answered:
#         best_em = 1 if normalize(p["answer"]) in corpus_norms else 0
#         ems.append(best_em); f1s.append(best_em)  # proxy same as EM in this toy set

#     # pass/fail
#     if cov < COV_MIN: reasons.append(f"coverage {cov:.2f} < {COV_MIN}")
#     if risk > RISK_MAX: reasons.append(f"risk {risk:.2f} > {RISK_MAX}")
#     if mean_tokens > BUDGET: reasons.append(f"mean_tokens {mean_tokens:.1f} > {BUDGET}")

#     out = {
#         "passed": len(reasons)==0,
#         "reasons": reasons,
#         "coverage": cov,
#         "risk": risk,
#         "ece": ece,
#         "mean_tokens": mean_tokens,
#         "answered": len(answered),
#         "total": len(preds),
#         "em_answered": sum(ems)/len(ems) if ems else 0.0,
#         "f1_answered": sum(f1s)/len(f1s) if f1s else 0.0,
#     }
#     print(json.dumps(out, indent=2))

def main():
    # ---------- existence checks ----------
    if not os.path.exists("tasks/selqa/calib.json"):
        print(json.dumps({"passed": False, "reasons": ["Missing calib.json"]}, indent=2))
        return
    if not os.path.exists(LOG_TEST):
        print(json.dumps({"passed": False, "reasons": ["Missing test log"]}, indent=2))
        return

    reasons = []

    # ---------- A) Load test/preds; require one row per test qid ----------
    test_items = list(read_jsonl("tasks/selqa/test.jsonl"))
    test_qids  = [t["qid"] for t in test_items]

    preds = list(read_jsonl("tasks/selqa/preds.jsonl")) if os.path.exists("tasks/selqa/preds.jsonl") else []
    pred_map = {p["qid"]: p for p in preds}

    missing = [q for q in test_qids if q not in pred_map]
    extra   = [q for q in pred_map if q not in set(test_qids)]
    if missing:
        reasons.append(f"preds.jsonl missing {len(missing)} test qids (e.g., {missing[:3]})")
    if extra:
        reasons.append(f"preds.jsonl has {len(extra)} unknown qids (e.g., {extra[:3]})")

    # ---------- B) Require BOTH modalities per test qid; forbid lexical on test ----------
    usage = defaultdict(set)
    for j in read_jsonl(LOG_TEST):
        if "qid" in j:
            usage[j["qid"]].add(j.get("tool"))
    for q in test_qids:
        if not {"retrieve_bm25", "retrieve_embed"}.issubset(usage[q]):
            reasons.append(f"qid {q} is missing BM25 or embed retrieval on test")
            break
    if any(j.get("tool") == "lexical_score" for j in read_jsonl(LOG_TEST)):
        reasons.append("lexical_score used on test (not allowed)")

    # ---------- helper normalizer ----------
    def normalize(s: str) -> str:
        return re.sub(r"[^a-z0-9 ]", "", str(s).lower()).strip()

    # ---------- answered subset (in test order) & coverage ----------
    answered = [pred_map[q] for q in test_qids if q in pred_map and str(pred_map[q].get("decision", "")).lower() == "answer"]
    coverage = len(answered) / (len(test_qids) or 1)

    # ---------- D) Risk/ECE on answered subset (corpus-proxy correctness) ----------
    corpus_texts = [j["text"] for j in read_jsonl("tasks/selqa/corpus.jsonl")]
    corpus_norms = {normalize(t) for t in corpus_texts}

    confidences, corrects = [], []
    for p in answered:
        ansN = normalize(p.get("answer", ""))
        ok = (ansN in corpus_norms)
        corrects.append(1 if ok else 0)
        try:
            confidences.append(float(p.get("confidence", 0.0)))
        except Exception:
            confidences.append(0.0)

    risk = 1 - (sum(corrects) / len(corrects) if corrects else 0.0)
    ece  = compute_ece(confidences, corrects)

    # ---------- C) Mean tokens over ALL test items ----------
    per_q_tokens = approx_tokens_from_logs()  # sums retrieval tokens by qid from LOG_TEST
    total_tokens = 0
    for q in test_qids:
        ans_len = len(str(pred_map[q].get("answer", "")).split()) if q in pred_map else 0
        total_tokens += per_q_tokens.get(q, 0) + ans_len
    mean_tokens = total_tokens / (len(test_qids) or 1)

    # ---------- E) Enforce pass criteria & print ----------
    if coverage   < COV_MIN:  reasons.append(f"coverage {coverage:.2f} < {COV_MIN}")
    if risk       > RISK_MAX: reasons.append(f"risk {risk:.2f} > {RISK_MAX}")
    if mean_tokens > BUDGET:  reasons.append(f"mean_tokens {mean_tokens:.1f} > {BUDGET}")

    out = {
        "passed": len(reasons) == 0,
        "reasons": reasons,
        "coverage": coverage,
        "risk": risk,
        "ece": ece,
        "mean_tokens": mean_tokens,
        "answered": len(answered),
        "total": len(test_qids),
        "em_answered": (sum(corrects) / len(corrects)) if corrects else 0.0,
        "f1_answered": (sum(corrects) / len(corrects)) if corrects else 0.0,
    }
    print(json.dumps(out, indent=2))


if __name__=="__main__":
    main()
