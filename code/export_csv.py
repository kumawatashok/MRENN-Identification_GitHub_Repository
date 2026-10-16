"""Export the JSON results to CSV files (results/csv/). Run from code/."""
import json, csv, os
R = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'results')
O = os.path.join(R, 'csv'); os.makedirs(O, exist_ok=True)

def flat(d, pre=''):
    for k, v in d.items():
        if isinstance(v, dict): yield from flat(v, pre + str(k) + '.')
        else: yield pre + str(k), v

# 1. per-seed metrics
S = json.load(open(os.path.join(R, 'summary.json')))
with open(os.path.join(O, 'metrics_per_seed.csv'), 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['mode', 'metric', 'model', 'example', 'seed0', 'seed1', 'seed2', 'seed3', 'seed4', 'mean', 'std', 'median'])
    for key, v in S.items():
        mode, metric, model, ex = key.split('|', 3)
        if not isinstance(v, dict):
            w.writerow([mode, metric, model, ex, '', '', '', '', '', v, '', '']); continue
        vals = list(v['vals']) + [''] * (5 - len(v['vals']))
        w.writerow([mode, metric, model, ex] + vals[:5] + [v['mean'], v['std'], v['median']])
# 2. other result files, flattened to key,value
for name, out in [('numbers.json', 'statistics_and_ratios.csv'), ('timing.json', 'timing.csv'),
                  ('lyap_diag.json', 'descent_check.csv'), ('selection_all.json', 'hyperparameter_selection.csv')]:
    p = os.path.join(R, name)
    if not os.path.exists(p): continue
    D = json.load(open(p))
    with open(os.path.join(O, out), 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['key', 'value'])
        for k, v in flat(D if isinstance(D, dict) else {'data': D}): w.writerow([k, json.dumps(v) if isinstance(v, (list, tuple)) else v])
print('written to', O)
