"""Verify delivered holdout hashes and numeric feasibility, without running models."""
import csv
import hashlib
import json
from pathlib import Path

SOURCE = Path('/Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26')
OUT = Path(__file__).resolve().parent
pre = SOURCE / '03_preregistration'
checks = []
for number, line in enumerate((pre / 'preeg_freeze_2026-10-24.sha256').read_text().splitlines(), 1):
    if not line.strip() or line.startswith('#'):
        continue
    expected, name = line.split(maxsplit=1)
    mapped = name.removeprefix('docs/')
    actual = hashlib.sha256((pre / mapped).read_bytes()).hexdigest()
    checks.append(dict(manifest_line=number, original_path=name, package_path=str(pre/mapped),
                       expected=expected, actual=actual, matches=expected == actual))

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

curve = read(SOURCE/'07_holdout/eval/curve.csv')
summary = read(SOURCE/'07_holdout/eval/summary.csv')
def feasible(r):
    return float(r['same'] or 'nan') >= .90 and int(r['evaluable']) >= 100

result = dict(freeze_checked=len(checks), freeze_passed=sum(r['matches'] for r in checks), freeze_details=checks,
    curve_rows=len(curve), configurations=len({r['config'] for r in curve}),
    feasible_configurations=sorted({r['config'] for r in curve if feasible(r)}),
    max_evaluable_pairs=max(int(r['evaluable']) for r in curve),
    max_pairs_with_same_at_least_point90=max(int(r['evaluable']) for r in curve if float(r['same'] or 'nan') >= .90),
    query_counts=sorted({int(r['n_query']) for r in summary}),
    note='68 evaluable queries cannot contain 100 answerable queries. This is distinct from the 100-pair operating-point constraint. Hash agreement does not establish freeze time.')
(OUT/'holdout_summary.json').write_text(json.dumps(result, indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k != 'freeze_details'}, indent=2))
