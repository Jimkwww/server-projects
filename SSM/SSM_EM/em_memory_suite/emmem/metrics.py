import numpy as np

def classification(y,p,classes):
    y=np.asarray(y).reshape(-1);p=np.asarray(p).reshape(-1)
    cm=np.bincount(classes*y.astype(int)+p.astype(int),minlength=classes**2).reshape(classes,classes)
    tp=np.diag(cm);den=cm.sum(0)+cm.sum(1)
    f1=np.divide(2*tp,den,out=np.zeros(classes,float),where=den>0)
    recall=np.divide(tp,cm.sum(1),out=np.zeros(classes,float),where=cm.sum(1)>0)
    iou=np.divide(tp,den-tp,out=np.zeros(classes,float),where=den-tp>0)
    return dict(oa=float(tp.sum()/max(1,cm.sum())),macro_f1=float(f1.mean()),miou=float(iou.mean()),
                per_class_f1=f1.tolist(),per_class_recall=recall.tolist(),confusion=cm.tolist())

def regression(y,p):
    y=np.asarray(y);p=np.asarray(p);e=p-y
    return dict(mae=np.abs(e).mean(0).tolist(),rmse=np.sqrt((e*e).mean(0)).tolist(),mse=float((e*e).mean()))

def binary_auc(y,s):
    # Mann-Whitney with tie-aware average ranks; higher score = anomaly.
    y=np.asarray(y).astype(bool);s=np.asarray(s);order=np.argsort(s,kind='stable');r=np.empty(len(s),float)
    i=0
    while i<len(s):
        j=i+1
        while j<len(s) and s[order[j]]==s[order[i]]:j+=1
        r[order[i:j]]=(i+j+1)/2;i=j
    n=y.sum();m=len(y)-n
    return float((r[y].sum()-n*(n+1)/2)/(n*m)) if n and m else None

def paired_bootstrap(a,b,seed=17,n=10000):
    # Resample independent seed pairs, NOT correlated per-sample predictions.
    d=np.array(a)-np.array(b);rng=np.random.default_rng(seed)
    means=d[rng.integers(len(d),size=(n,len(d)))].mean(1)
    return dict(n_pairs=len(d),mean=float(d.mean()),std=float(d.std(ddof=1)) if len(d)>1 else None,
                ci95=np.quantile(means,[.025,.975]).tolist(),paired_differences=d.tolist(),
                caveat='Few seeds give uncertain bootstrap CIs; report all paired differences.')
