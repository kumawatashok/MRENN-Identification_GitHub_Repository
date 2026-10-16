"""Resumable task runner: executes tasks (each writing its own result file), skipping finished ones,
stops dispatching after a wall-clock budget so that it can be called repeatedly."""
import sys, os, time, json
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")


def _work(t):
    kind = t[0]
    if kind == "sel":
        import selection
        return selection.sel_task(*t[1:])
    if kind == "exp":
        import run_experiments
        run_experiments.main(*t[1:])
        return None


def done(t):
    if t[0] == "sel":
        import selection
        return os.path.exists(selection.sel_path(*t[1:]))
    if t[0] == "exp":
        return os.path.exists(os.path.join(RES, "raw", f"{t[1]}_{t[2]}_{t[3]}.json"))


def run(tasks, budget=420, procs=2):
    t0 = time.time(); todo = [t for t in tasks if not done(t)]
    print(f"{len(tasks)-len(todo)}/{len(tasks)} done, {len(todo)} to go", flush=True)
    if not todo: return True
    with Pool(procs) as p:
        it = iter(todo); active = []; 
        while True:
            while len(active) < procs and time.time() - t0 < budget:
                try: t = next(it)
                except StopIteration: break
                active.append(p.apply_async(_work, (t,)))
            if not active: break
            active[0].get(); active = [a for a in active if not a.ready()] if False else [a for a in active if a is not active[0]]
            active = [a for a in active if not a.ready() or a.get() is not None or True]
    left = [t for t in tasks if not done(t)]
    print(f"remaining {len(left)}  elapsed {time.time()-t0:.0f}s", flush=True)
    return not left
