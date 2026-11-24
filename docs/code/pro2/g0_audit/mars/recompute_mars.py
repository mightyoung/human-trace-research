"""Read-only source audit. Outputs only into this script's directory.

B is conditional on the audit hypothesis that PID 0782 is gallery-out.
The package does not include full MARS GT or per-config cluster membership.
R is an explicitly simulated forced-rejection policy, not observed output.
"""
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np

SOURCE = Path('/Users/muyi/Downloads/dev/human-count/docs/1026-v3/human-count-package-2026-10-26')
OUT = Path(__file__).resolve().parent

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

def ranks(values):
    v=np.asarray(values,float); order=v.argsort(); result=np.empty(len(v),float)
    i=0
    while i<len(v):
        j=i
        while j+1<len(v) and v[order[j+1]]==v[order[i]]: j+=1
        result[order[i:j+1]]=(i+j)/2
        i=j+1
    return result

def rho(x,y):
    x=ranks(x);y=ranks(y);x-=x.mean();y-=y.mean()
    return float(x@y / math.sqrt((x@x)*(y@y)))

def exact_p(lost,gained):
    n=lost+gained
    return min(1., 2*sum(math.comb(n,i) for i in range(min(lost,gained)+1))/2**n)

def auc(score,label):
    score=np.asarray(score,float);label=np.asarray(label,bool)
    if not label.any() or label.all(): return float('nan')
    p=score[label,None];n=score[~label][None,:]
    return float((p>n).mean()+.5*(p==n).mean())

def bootstrap(score,label,n=2000,seed=7):
    rng=np.random.default_rng(seed);score=np.asarray(score);label=np.asarray(label)
    values=[]
    for _ in range(n):
        ix=rng.choice(len(label),len(label),replace=True)
        value=auc(score[ix],label[ix])
        if np.isfinite(value): values.append(value)
    return [float(x) for x in np.percentile(values,[2.5,97.5])]

verdict={p.stem:read(p) for p in sorted((SOURCE/'04_results/mars/verdicts').glob('*.csv'))}
h1={r['config']:r for r in read(SOURCE/'04_results/mars/h1_rank.csv')}
supp={r['config']:r for r in read(SOURCE/'04_results/supp_table_mars.csv')}
baseline={r['query_id']:r['verdict'] for r in verdict['A_baseline']}
accepted_union={r['query_id']:r['q_pid'] for rows in verdict.values() for r in rows}
cpids={r['c_pid'] for rows in verdict.values() for r in rows if r['c_pid']}
unmatched=sorted(set(accepted_union.values())-cpids)
rows=[]; differences=[]; excluded=[]
for cfg,vs in verdict.items():
    for policy in ('A','B','R'):
        kept=vs if policy=='A' else [r for r in vs if r['q_pid'] not in {'0782','1172'}]
        counts=Counter(r['verdict'] for r in kept)
        current={r['query_id']:r['verdict'] for r in kept}
        lost=sum(v=='C' and current.get(q)!='C' for q,v in baseline.items())
        gained=sum(v=='C' and baseline.get(q)!='C' for q,v in current.items())
        denominator=480 if policy=='B' else 484
        row=dict(config=cfg,policy=policy,query_denominator=denominator,
                 accepted=len(kept),C=counts['C'],W=counts['W'],U=counts['U'],
                 unanswered=denominator-len(kept),precision=counts['C']/(counts['C']+counts['W']),
                 C_over_484=counts['C']/484,delta_C=counts['C']-255,lost=lost,gained=gained,
                 paired_p=exact_p(lost,gained))
        assert row['accepted']==row['C']+row['W']+row['U']
        assert denominator==row['accepted']+row['unanswered']
        rows.append(row)
        expected=h1[cfg] if policy=='A' else supp[cfg]
        for key in ('accepted','C','W','U'):
            if row[key]!=int(expected[key]): differences.append([cfg,policy,key,row[key],expected[key]])
    for r in vs:
        if r['q_pid'] in {'0782','1172'}: excluded.append(dict(config=cfg,**r))

harm=json.loads((SOURCE/'04_results/mars/harm_auc.json').read_text())
cluster_stats={}
for policy in ('A','B','R'):
    kept=verdict['A_baseline'] if policy=='A' else [r for r in verdict['A_baseline'] if r['q_pid'] not in {'0782','1172'}]
    grouped=defaultdict(list)
    for r in kept: grouped[int(r['cluster_local'])].append(r)
    cs=[]
    for c in harm['clusters']:
        v=grouped[c['cluster']]
        cs.append(dict(c,n_query=len(v),harmful=int(any(r['verdict']=='W' for r in v)),n_wrong=sum(r['verdict']=='W' for r in v)))
    withq=[c for c in cs if c['n_query']]
    score=[c['d_max'] for c in withq];labels=[c['harmful'] for c in withq]
    pure=sum(c['n_query'] for c in cs if c['n_ids']==1)
    cluster_stats[policy]=dict(n_clusters=len(cs),n_with_query=len(withq),n_harmful=sum(labels),
        inherited=pure/len(kept),inherited_numerator=pure,
        auc_mixed=auc([c['d_max'] for c in cs],[c['mixed'] for c in cs]),
        auc_harmful_with_query=auc(score,labels),ci95=bootstrap(score,labels),bootstrap_seed=7,bootstrap_n=2000)
    (OUT/f'clusters_{policy}.json').write_text(json.dumps(cs,indent=2)+'\n')

def save_csv(name,data):
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(data[0]));w.writeheader();w.writerows(data)
save_csv('configuration_results.csv',rows)
save_csv('excluded_accepted_instances.csv',excluded)
best=max(h1,key=lambda c:float(h1[c]['inherited']))
summary=dict(source=str(SOURCE),configuration_count=len(verdict),
    accepted_union_queries=len(accepted_union),accepted_union_pids=len(set(accepted_union.values())),
    query_pids_never_seen_as_predicted_cluster_pid=unmatched,
    missing_gt_warning='Absence from predicted c_pid is not proof of absence from gallery GT. PID 1172 and its two rejected queries are not observable in accepted-only verdicts.',
    excluded_accepted_instances=len(excluded),excluded_query_ids=sorted({r['query_id'] for r in excluded}),
    observed_A_spearman=rho([float(h1[c]['inherited']) for c in h1],[int(h1[c]['C']) for c in h1]),
    observed_A_spearman_input='Published rounded 4-digit inherited ratios; all C counts independently verified from accepted verdicts.',
    best_inherited_config=best,best_inherited_delta_C=int(h1[best]['C'])-255,
    baseline=[r for r in rows if r['config']=='A_baseline'],
    H2c=[r for r in rows if r['config']=='C_dmax0.30'],
    comparison_mismatches=differences,cluster_metrics=cluster_stats,
    B_R_spearman='Not independently identifiable: per-configuration cluster n_ids/members missing; baseline n_ids available in harm_auc.json only.',
    assumptions_B_R='Conditionally remove known accepted PID 0782/1172; use B N=480 and R N=484. Full GT query universe absent. R simulates forced rejection and therefore accepted/C/W/U identical to B.',
    source_A_cluster_metadata_consistent=all(c['n_query']==len([r for r in verdict['A_baseline'] if int(r['cluster_local'])==c['cluster']]) for c in harm['clusters']))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,indent=2))
