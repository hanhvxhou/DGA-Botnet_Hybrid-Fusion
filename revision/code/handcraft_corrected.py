"""Handcrafted branch with the corrected consonant set and the BOS/EOS Markov estimator;
retrains XGBoost for seeds 42-46 and saves validation/test probabilities.
Run from the project root (needs M6_v3_Markov.py, revision_experiments.py, DataNew/)."""
import os, numpy as np, time
import M6_v3_Markov as m6, revision_experiments as rx
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
Xtr,Xva,Xte,ytr,yva,yte,_ = m6.load_data_split("DataNew")
m6._load_english_words()
ben=[d for d,y in zip(Xtr,ytr) if y==0]
m6.build_markov_models(ben)
mk=rx.MarkovEOS().fit([rx.markov_text(d,m6) for d in ben])
def feats(doms):
    m6.CONSONANTS=set("bcdfghjklmnopqrstvwxyz"); A=m6.extract_handcrafted_matrix(doms)
    m6.CONSONANTS=set("bcdfghjklmnpqrstvwxyz");  B=m6.extract_handcrafted_matrix(doms)
    C=rx.handcraft_matrix_eos(doms,m6,mk,B)
    return A,B,C
F={}
for n,d in [("train",Xtr),("val",Xva),("test",Xte)]:
    t=time.time(); F[n]=feats(list(d)); print(n,time.time()-t,flush=True)
out={}
for s in [42,43,44,45,46]:
    for k,name in enumerate(["hc_orig_rep","hc_fix","hc_fix_eos"]):
        sc=StandardScaler().fit(F["train"][k])
        x=XGBClassifier(n_estimators=500,max_depth=6,learning_rate=0.1,subsample=0.8,colsample_bytree=0.8,n_jobs=-1,tree_method="hist",random_state=s,eval_metric="logloss",early_stopping_rounds=20,verbosity=0)
        x.fit(sc.transform(F["train"][k]),ytr,eval_set=[(sc.transform(F["val"][k]),yva)],verbose=False)
        for st in ["val","test"]: out[f"{s}|{name}|{st}"]=x.predict_proba(sc.transform(F[st][k]))[:,1].astype(np.float32)
    print("seed",s,flush=True)
np.savez_compressed("hcfix_probs.npz",**out)
np.save("fix_test_feats.npy",F["test"][1]); print("words",len(m6.ENGLISH_WORDS_SET))
