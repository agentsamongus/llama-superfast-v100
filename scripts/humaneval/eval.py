#!/usr/bin/env python3
"""HumanEval+ client for a running llama-server: HumanEval/0..31, one seed, T=1.0 top_p 0.95 top_k 20, thinking on (template default xhigh, no reasoning_effort sent); EvalPlus's own chat instruction and sanitizer, 16384 max tokens,
streamed, single stream. Base tests run per problem; plus at the end of each stage (8 and 32). usage: eval.py <port> <server.log> <seed>
Needs evalplus 0.3.1 (pip install evalplus==0.3.1). <server.log> is the log file the server writes to (the script reads its verified-round counts).
Writes progress.txt, run-seed<seed>/ and stage-{8,32}-summary.{txt,json} next to itself."""
import json, os, re, sys, time, http.client, statistics as st
import multiprocessing; multiprocessing.set_start_method("fork")  # python 3.14 forkserver breaks evalplus untrusted_check
sys.path.insert(0, os.path.dirname(__file__))
from evalplus.data import get_human_eval_plus, get_human_eval_plus_hash
from evalplus.evaluate import check_correctness, get_groundtruth
from evalplus.sanitize import sanitize
port, slog, seed = int(sys.argv[1]), sys.argv[2], int(sys.argv[3])
H = os.path.dirname(os.path.abspath(__file__)); RUN = f"{H}/run-seed{seed}"; os.makedirs(RUN, exist_ok=True)
probs = get_human_eval_plus(); gt = get_groundtruth(probs, get_human_eval_plus_hash(), [])
IDS = [f"HumanEval/{i}" for i in range(32)]; STAGES = (8, 32)
rows = []; logoff = os.path.getsize(slog)
IPREFIX = "Please provide a self-contained Python script that solves the following problem in a markdown code block:"  # evalplus/codegen.py instruction_prefix
def prompt_of(p): return IPREFIX + f"\n```python\n{p['prompt'].strip()}\n```"  # evalplus/provider/openai.py chat message
def rounds_of(n):
    global logoff
    for _ in range(100):
        time.sleep(0.1)
        d = open(slog, errors="replace").read().encode(errors="replace")[logoff:].decode(errors="replace")
        if "stop processing" in d or "release" in d:
            logoff += len(d.encode()); m = re.search(r"(\d+) of (\d+) verified rounds", d)
            a = re.search(r"\(\s*(\d+) accepted /\s*(\d+) generated\)", d)
            return (int(m.group(2)) if m else (n - int(a.group(1)) if a else None)), (int(a.group(1)), int(a.group(2))) if a else None
    return None, None
def totals():
    n = len(rows); tok = sum(r["n"] for r in rows); ms = sum(r["ms"] for r in rows); rd = sum(r["rounds"] or 0 for r in rows if r["rounds"])
    tk = sum(r["n"] for r in rows if r["rounds"])
    return (f"TOTALS done {n}/32 base_pass {sum(r['base'] for r in rows)} agg_out_tok/s {tok/ms*1e3:.2f} mean_steady_decode_tok/s {st.mean(r['tps'] for r in rows):.2f} "
            f"tok/round_pooled {tk/max(rd,1):.3f} mean_len {tok/n:.0f}")
def progress():
    L = [totals()] + [f"{r['id']:14s} base {'PASS' if r['base'] else 'FAIL'}  out_tokens {r['n']:5d}  finish {r['fin']:6s}  decode {r['tps']:6.2f} tok/s  tok/round {r['tpr']:.3f}  ({r['wall']:.0f}s wall)" for r in rows]
    open(f"{H}/progress.txt.tmp", "w").write("\n".join(L) + "\n"); os.replace(f"{H}/progress.txt.tmp", f"{H}/progress.txt")
def stage(k):
    res = []
    for r in rows[:k]:
        o = check_correctness("humaneval", 0, probs[r["id"]], r["solution"], gt[r["id"]], base_only=False, fast_check=False, identifier=r["id"])
        res.append((r["id"], o["base"][0] == "pass", o["plus"][0] == "pass"))
    sub = rows[:k]; tok = sum(r["n"] for r in sub); ms = sum(r["ms"] for r in sub); rr = [r for r in sub if r["rounds"]]
    S = {"problems": k, "seed": seed, "base": sum(b for _, b, _ in res), "plus": sum(p for _, _, p in res),
         "natural_stop": sum(r["fin"] == "stop" for r in sub), "length_capped": sum(r["fin"] == "length" for r in sub),
         "agg_out_tok_s": tok/ms*1e3, "mean_steady_decode_tok_s": st.mean(r["tps"] for r in sub),
         "tok_per_round_pooled": sum(r["n"] for r in rr)/sum(r["rounds"] for r in rr), "tok_per_round_mean_of_requests": st.mean(r["tpr"] for r in rr),
         "mean_output_len": tok/k, "total_out_tokens": tok, "total_decode_s": ms/1e3, "per_problem": res}
    json.dump(S, open(f"{H}/stage-{k}-summary.json", "w"), indent=1)
    with open(f"{H}/stage-{k}-summary.txt", "w") as f:
        f.write(f"stage {k} (HumanEval/0-{k-1}, seed {seed})\npass@1 base {S['base']}/{k}  plus {S['plus']}/{k}\nnatural stops {S['natural_stop']}  length-capped {S['length_capped']}\n"
                f"aggregate output {S['agg_out_tok_s']:.2f} tok/s  mean steady decode {S['mean_steady_decode_tok_s']:.2f} tok/s\n"
                f"tokens/round pooled {S['tok_per_round_pooled']:.3f}  per-request mean {S['tok_per_round_mean_of_requests']:.3f}\nmean output length {S['mean_output_len']:.0f} tokens\n")
        for i, b, p in res: f.write(f"{i} base {'pass' if b else 'FAIL'} plus {'pass' if p else 'FAIL'}\n")
for tid in IDS:
    p = probs[tid]; t0 = time.time()
    body = {"model": "x", "messages": [{"role": "user", "content": prompt_of(p)}], "temperature": 1.0, "top_p": 0.95, "top_k": 20, "seed": seed, "max_tokens": 16384,
            "stream": True, "timings_per_token": True}
    c = http.client.HTTPConnection("127.0.0.1", port, timeout=7200); c.request("POST", "/v1/chat/completions", json.dumps(body), {"Content-Type": "application/json"})
    r = c.getresponse(); content = []; reas = []; fin = None; tim = None
    while True:
        line = r.readline()
        if not line: break
        line = line.strip()
        if not line.startswith(b"data: "): continue
        if line[6:] == b"[DONE]": break
        ch = json.loads(line[6:])
        if ch.get("timings"): tim = ch["timings"]
        for cc in ch.get("choices") or []:
            d = cc.get("delta", {}); content.append(d.get("content") or ""); reas.append(d.get("reasoning_content") or "")
            if cc.get("finish_reason"): fin = cc["finish_reason"]
    c.close()
    txt = "".join(content); n = tim["predicted_n"]; rd, acc = rounds_of(n)
    sol = sanitize(txt, p["entry_point"])  # evalplus.sanitize.sanitize on the raw reply, as evalplus does
    open(f"{RUN}/{tid.replace('/', '_')}.json", "w").write(json.dumps({"id": tid, "finish": fin, "timings": tim, "rounds": rd, "acc": acc, "reasoning": "".join(reas), "content": txt, "solution": sol}))
    b = check_correctness("humaneval", 0, p, sol, gt[tid], base_only=True, identifier=tid)["base"][0] == "pass"
    rows.append({"id": tid, "base": b, "n": n, "fin": fin or "?", "ms": tim["predicted_ms"], "tps": tim["predicted_per_second"], "rounds": rd, "tpr": n/rd if rd else float("nan"),
                 "wall": time.time()-t0, "solution": sol, "acc": acc})
    progress()
    if len(rows) in STAGES: stage(len(rows))
