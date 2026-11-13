"""Read-only checks against the current source; output numerical audit evidence."""
import sys, json, math, platform, hashlib
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.special import ndtr
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from private_llm_analytics.mechanisms.gaussian import calibrate_analytic_gaussian, calibrate_classical_gaussian
from private_llm_analytics.config import load_app_config
from private_llm_analytics.planner.planner import AnalyticsPlanner
from private_llm_analytics.planner.query import AnalyticsQuery
from private_llm_analytics.continual.mechanism import calibrate_tree_gaussian_sigma, compute_level_rdp_vector, evaluate_rdp_epsilon
from private_llm_analytics.continual.tree import BinaryTimeTree
from private_llm_analytics.continual.contributions import build_temporal_user_contributions

rows=[]
for name,sens in [('Prevalence',3.),('Normalized Share',1.),('Interaction Volume',20.),('Success Rate',math.sqrt(2)*20),('Top-K conservative',math.sqrt(5)),('Top-K tight T=3',math.sqrt(3))]:
    c=sens*math.sqrt(2*math.log(1.25/1e-6))
    a=calibrate_analytic_gaussian(1.,1e-6,sens)
    independent=sens*brentq(lambda r: ndtr(1/(2*r)-r)-math.e*ndtr(-1/(2*r)-r)-1e-6,1.,10.,xtol=1e-14)
    assert abs(a-independent)<1e-9
    assert abs(c-calibrate_classical_gaussian(1.,1e-6,sens))<1e-10
    rows.append(dict(metric=name,sensitivity=sens,classical=c,analytic=a,independent_analytic=independent,sd_reduction_pct=100*(1-a/c),variance_reduction_pct=100*(1-(a/c)**2)))

cfg=load_app_config(ROOT/'configs')
def key(config, metric):
    return AnalyticsPlanner(config).create_plan(AnalyticsQuery(metric=metric,recipient='internal'),dataset_version='audit_fixed',public_population_size=50000).release_key
keychecks={}
for metric,field,value in [('success_rate','min_noisy_denominator',5.),('success_rate','clip_norm_rate',21.),('top_k_tasks','top_k',2),('top_k_tasks','max_tasks_per_user',2),('user_prevalence','max_cells_per_user',8),('interaction_volume','clip_norm',21.)]:
    mod=cfg.model_copy(deep=True)
    setattr(mod.metrics.metrics[metric].contribution,field,value)
    keychecks[field]=dict(changes_key=key(cfg,metric)!=key(mod,metric),new_value=value)

tree=BinaryTimeTree(90)
sigma=calibrate_tree_gaussian_sigma(90,40.,2.,1e-6)
rv=compute_level_rdp_vector(40.,sigma)
eps=evaluate_rdp_epsilon({a:tree.num_levels*r for a,r in rv.items()},1e-6)
# Actual online bounding on a user active on every day; all node and level bounds.
events=pd.DataFrame([dict(user_id='u',timestamp=f'2026-01-{t:02d}',provider='provider_a',task='technical') for t in range(1,9) for j in range(20)])
z,_,_=build_temporal_user_contributions(events,['u'],'2026-01-01',8,daily_l2_bound=10.,epoch_l12_bound=25.)
level_norms=[]
t8=BinaryTimeTree(8)
for level in range(t8.num_levels):
    node_vectors=[]
    for i in range(t8.tree_size//(2**level)):
        start,end=t8.node_interval(level,i)
        v=z[0,start-1:end].sum(axis=0)
        assert np.linalg.norm(v)<=min(25.,(end-start+1)*10.)+1e-12
        node_vectors.append(v)
    level_norms.append(float(np.linalg.norm(np.concatenate(node_vectors))))
assert max(level_norms)<=25.+1e-12
assert abs(np.linalg.norm(z,axis=2).sum()-25.)<1e-12

data=ROOT/'data'/'interactions_n50000.parquet'
df=pd.read_parquet(data,columns=['user_id','provider','task'])
task_users=df.groupby('task')['user_id'].nunique().to_dict()
cell_users=df.groupby(['provider','task'])['user_id'].nunique()
result=dict(python=platform.python_version(),gaussian=rows,release_key_checks=keychecks,tree=dict(days=90,levels=tree.num_levels,sigma=sigma,epoch_epsilon_bound=eps,test_daily_norms=np.linalg.norm(z,axis=2).tolist(),test_level_norms=level_norms),data=dict(source='synthetic (DATASET_MANIFEST.md and scenario.yaml)',rows=len(df),users=df.user_id.nunique(),sha256=hashlib.sha256(data.read_bytes()).hexdigest(),task_users=task_users,provider_a_users=df[df.provider=='provider_a'].user_id.nunique(),provider_a_technical=int(cell_users.loc[('provider_a','technical')])),config=dict(tau=cfg.metrics.metrics['success_rate'].contribution.min_noisy_denominator,delta_rdp=cfg.privacy.accountant.delta_global,delta_basic=cfg.privacy.global_budget.delta,top_k=cfg.metrics.metrics['top_k_tasks'].contribution.top_k,tasks=cfg.scenario.tasks))
out=Path(__file__).with_name('numerical_audit.json')
out.write_text(json.dumps(result,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps(result,ensure_ascii=False,indent=2))
