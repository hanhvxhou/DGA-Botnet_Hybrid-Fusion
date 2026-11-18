"""Independent-benign FPR and recall at fixed FPR (Table 10).
usage: python ben2.py <root containing rev_out> <analysis dir with all_runs.csv>"""
import sys, json, numpy as np, pandas as pd
from sklearn.metrics import roc_curve
RO=sys.argv[1]+'/rev_out'; AD=sys.argv[2]; runs=pd.read_csv(AD+'/all_runs.csv',sep=';')
yt=pd.read_csv(f'{RO}/sets/test.csv',sep=';').label.values
rows=[]
for m in ['M6','bert','distilbert','string','bilstm','M3','hc_orig']:
  for b in ['test_benign','ben_tranco','ben_umb_reg','ben_umb_fqdn']:
    R=dict(model=m,set=b); acc={'fpr':[],'r01':[],'r1':[],'auc':[]}
    for s in [42,43,44,45,46]:
      P=np.load(f'{RO}/seed{s}/probs.npz')
      if m=='M6':
        r=runs[(runs.model=='M6_LF_search3_orig')&(runs.seed==s)].iloc[0]; w=json.loads(r.weights); t=r.thr
        f=lambda st: sum(w[k]*P[f'{k}|{st}'] for k in w if w[k]>0)/sum(v for v in w.values() if v>0)
      else:
        t=runs[(runs.model==m)&(runs.seed==s)&(runs.thr_mode=='val')].thr.iloc[0]; f=lambda st: P[f'{m}|{st}']
      pt=f('test'); pos=pt[yt==1]; neg=pt[yt==0] if b=='test_benign' else f(b)
      acc['fpr'].append((neg>=t).mean())
      y=np.r_[np.ones(len(pos)),np.zeros(len(neg))]; sc=np.r_[pos,neg]
      fpr,tpr,_=roc_curve(y,sc)
      acc['r01'].append(tpr[fpr<=0.001].max()); acc['r1'].append(tpr[fpr<=0.01].max())
      from sklearn.metrics import roc_auc_score; acc['auc'].append(roc_auc_score(y,sc))
    for k,v in acc.items(): v=np.array(v); R[k]=f'{v.mean()*100:.2f} ± {v.std(ddof=1)*100:.2f}' if k!='auc' else f'{v.mean():.4f}'
    rows.append(R)
D=pd.DataFrame(rows); D.to_csv(AD+'/benign_v2.csv',sep=';',index=False); print(D.to_string(index=False))
