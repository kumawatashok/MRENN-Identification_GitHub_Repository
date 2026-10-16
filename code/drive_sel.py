import runq, selection, json, os
ok = runq.run(selection.tasks(), budget=110)
if ok:
    s = selection.summarize(); json.dump(s, open(os.path.join(runq.RES, "selection_all.json"), "w"), indent=1, default=float)
    for a, v in s.items():
        for r in sorted(v, key=lambda r: r["score"]): print(a, r["name"], round(r["score"], 4), [round(x, 3) for x in r["per_set"]])
