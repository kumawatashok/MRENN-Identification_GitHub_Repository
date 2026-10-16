import sys, runq
group = sys.argv[1]; seeds = [int(s) for s in sys.argv[2].split(",")]; budget = int(sys.argv[3]) if len(sys.argv) > 3 else 100
keys = ["ex1", "ex2", "ex3", "ex4", "ex5", "ex6"]
tasks = [("exp", k, s, group) for s in seeds for k in keys]
runq.run(tasks, budget=budget)
