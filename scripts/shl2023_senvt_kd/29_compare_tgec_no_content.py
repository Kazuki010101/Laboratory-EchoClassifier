from pathlib import Path
import json
import statistics

root = Path('experiments/shl2023_senvt_kd/heldout_evaluation/user23_test')
for method in ['tgec', 'tgec_no_content']:
    acc, f1 = [], []
    for seed in (0, 1, 2):
        path = root / f'{method}_seed{seed}' / 'evaluation_summary.json'
        if not path.is_file():
            raise FileNotFoundError(path)
        d = json.loads(path.read_text())
        a = float(d['acc1'])
        f = float(d['f1_macro'])
        acc.append(a * 100 if abs(a) <= 1 else a)
        f1.append(f * 100 if abs(f) <= 1 else f)
    print(method)
    print('  acc seeds:', acc)
    print('  acc mean±sd:', statistics.mean(acc), statistics.stdev(acc))
    print('  f1  seeds:', f1)
    print('  f1  mean±sd:', statistics.mean(f1), statistics.stdev(f1))