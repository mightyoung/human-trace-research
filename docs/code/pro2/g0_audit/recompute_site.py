"""G0 audit of exported results; never writes into the source package.

This checks arithmetic and exported verdict consistency, not video truth,
cluster membership reconstruction, or a model rerun.
"""
import csv
import hashlib
import json
import math
import platform
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import scipy
from scipy.stats import spearmanr

SOURCE = Path('/Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26')
OUT = Path(__file__).resolve().parent


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def indexed(path):
    rows = read(path)
    assert len({r['config'] for r in rows}) == len(rows)
    return {r['config']: r for r in rows}


def exact_p(lost, gained):
    n = lost + gained
    return min(1.0, 2 * sum(math.comb(n, k) for k in range(min(lost, gained) + 1)) / 2**n)


def write_csv(name, rows):
    with (OUT / name).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


manifest = []
for number, line in enumerate((SOURCE / 'MANIFEST.sha256').read_text().splitlines(), 1):
    if not line.strip():
        continue
    expected, name = line.split(maxsplit=1)
    file = SOURCE / name.lstrip('*')
    actual = hashlib.sha256(file.read_bytes()).hexdigest() if file.is_file() else None
    manifest.append(dict(line=number, path=name, expected=expected, actual=actual, matches=actual == expected))
write_csv('source_integrity.csv', manifest)

precision = indexed(SOURCE / '04_results/site/precision.csv')
h1 = indexed(SOURCE / '04_results/site/onsite_h1.csv')
supp = indexed(SOURCE / '04_results/supp_table_site.csv')
multicomp = indexed(SOURCE / '04_results/a8_multicomp.csv')
verdicts = {p.stem: read(p) for p in sorted((SOURCE / '04_results/site/verdicts').glob('*.csv'))}
assert set(precision) == set(h1) == set(supp) == set(multicomp) == set(verdicts)
base = {r['query_id']: r for r in verdicts['A_baseline']}
issues, comparisons, by_identity, metadata = [], [], [], []
all_q_pid = {}
for config, rows in verdicts.items():
    assert len({r['query_id'] for r in rows}) == len(rows), (config, 'duplicate query')
    counts = Counter(r['verdict'] for r in rows)
    assert set(counts) <= {'C', 'W', 'U'}
    for row in rows:
        qid = row['query_id']
        if qid in all_q_pid:
            assert all_q_pid[qid] == row['q_pid'], (qid, 'truth differs across configurations')
        all_q_pid[qid] = row['q_pid']
        expected = 'U' if not row['c_pid'] else ('C' if row['q_pid'] == row['c_pid'] else 'W')
        assert row['verdict'] == expected, (config, qid, expected)
    current = {r['query_id']: r for r in rows}
    lost = [q for q, r in base.items() if r['verdict'] == 'C' and current.get(q, {}).get('verdict') != 'C']
    gained = [q for q, r in current.items() if r['verdict'] == 'C' and base.get(q, {}).get('verdict') != 'C']
    pure = sum(int(r['cluster_n_ids']) == 1 for r in rows)
    recomputed = dict(config=config, accepted=len(rows), C=counts['C'], W=counts['W'], U=counts['U'],
                      precision=counts['C'] / (counts['C'] + counts['W']), C_over_N=counts['C'] / 229,
                      inherited_numerator=pure, inherited_from_exported_metadata=pure / len(rows),
                      delta=counts['C']-117, lost=len(lost), gained=len(gained), p=exact_p(len(lost), len(gained)))
    comparisons.append(recomputed)
    for table_name, table in [('precision', precision), ('supplement', supp)]:
        for key in ('accepted', 'C', 'W', 'U'):
            if recomputed[key] != int(table[config][key]):
                issues.append([config, table_name, key, recomputed[key], table[config][key]])
    if round(pure / len(rows), 4) != float(h1[config]['inh_strict']):
        issues.append([config, 'onsite_h1', 'inh_strict', pure / len(rows), h1[config]['inh_strict']])
    for key in ('C', 'delta', 'lost', 'gained'):
        if recomputed[key] != int(multicomp[config][key]):
            issues.append([config, 'a8_multicomp', key, recomputed[key], multicomp[config][key]])
    if not math.isclose(recomputed['p'], float(multicomp[config]['p']), rel_tol=1e-12, abs_tol=1e-15):
        issues.append([config, 'a8_multicomp', 'p', recomputed['p'], multicomp[config]['p']])
    if config == 'C_dmax0.30':
        delta = defaultdict(lambda: Counter(lost=0, gained=0))
        for q in lost:
            delta[base[q]['q_pid']]['lost'] += 1
        for q in gained:
            delta[current[q]['q_pid']]['gained'] += 1
        by_identity = [dict(pid=p, **d, delta=d['gained']-d['lost']) for p, d in sorted(delta.items())]

others = sorted([r for r in comparisons if r['config'] != 'A_baseline'], key=lambda r: r['p'])
previous = 1.0
for i in range(len(others)-1, -1, -1):
    row = others[i]
    row['bh_q'] = min(previous, row['p'] * len(others) / (i+1))
    previous = row['bh_q']
    row['bonf_sig'] = row['p'] < .05 / len(others)
    row['bh_sig'] = row['bh_q'] < .05
    if not math.isclose(row['bh_q'], float(multicomp[row['config']]['bh_q']), rel_tol=1e-12, abs_tol=1e-15):
        issues.append([row['config'], 'a8_multicomp', 'bh_q', row['bh_q'], multicomp[row['config']]['bh_q']])
    for flag in ('bonf_sig', 'bh_sig'):
        if row[flag] != (multicomp[row['config']][flag].lower() == 'true'):
            issues.append([row['config'], 'a8_multicomp', flag, row[flag], multicomp[row['config']][flag]])
for row in comparisons:
    for key in ('bh_q', 'bonf_sig', 'bh_sig'):
        row.setdefault(key, '')
write_csv('site_configuration_checks.csv', comparisons)
write_csv('site_veto_identity_deltas.csv', by_identity)

loss = read(SOURCE / '04_results/site/loss_decomposition.csv')
for row in loss:
    assert int(row['answerable']) == sum(int(row[k]) for k in ('correct', 'wrong', 'unresolved', 'unanswered'))
    assert int(row['evaluable']) == sum(int(row[k]) for k in ('answerable', 'nonperson', 'gallery_out'))
all_loss = next(r for r in loss if r['group'] == 'ALL')
for key in all_loss:
    if key != 'group':
        assert int(all_loss[key]) == sum(int(r[key]) for r in loss if r['group'] != 'ALL')

summary = dict(
    source=str(SOURCE), python=platform.python_version(), numpy=np.__version__, scipy=scipy.__version__,
    source_manifest_sha256=hashlib.sha256((SOURCE/'MANIFEST.sha256').read_bytes()).hexdigest(),
    manifest_checked=len(manifest), manifest_passed=sum(r['matches'] for r in manifest),
    configurations_checked=len(comparisons), cross_table_mismatches=issues,
    verdict_pid_consistency='passed for all exported accepted rows; omitted/rejected queries not tested',
    stored_h1_rho=float(spearmanr([float(r['inh_strict']) for r in h1.values()], [int(r['C']) for r in h1.values()]).statistic),
    reconstructed_h1_rho=float(spearmanr([r['inherited_from_exported_metadata'] for r in comparisons], [r['C'] for r in comparisons]).statistic),
    baseline=next(r for r in comparisons if r['config'] == 'A_baseline'),
    veto=next(r for r in comparisons if r['config'] == 'C_dmax0.30'),
    bonferroni_count=sum(r['bonf_sig'] for r in others), bh_count=sum(r['bh_sig'] for r in others),
    bonferroni_all_negative=all(r['delta'] < 0 for r in others if r['bonf_sig']),
    bh_all_negative=all(r['delta'] < 0 for r in others if r['bh_sig']),
    loss_closure='passed for ALL and four cameras; camera sums equal ALL',
    accepted_union_queries=len(all_q_pid), accepted_union_identities=len(set(all_q_pid.values())),
    limitations=['GT, all rejected-query PIDs, and cluster-member rosters absent from package.',
                 'Cannot independently rebuild majority identities or absolute identity-clustered confidence intervals.',
                 'Inherited reconstruction uses exported n_ids; it cannot restore UNK/nonperson members omitted upstream.',
                 'McNemar and BH results are arithmetic reproduction of query-level calculations, not validation of independence assumptions.'])
(OUT/'site_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n')
print(json.dumps(summary, ensure_ascii=False, indent=2))
