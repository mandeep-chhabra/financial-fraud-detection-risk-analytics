import pandas as pd, numpy as np, json, time
p='data/raw/paysim.csv'
head=pd.read_csv(p,nrows=3)
print("COLUMNS:",list(head.columns)); print(head.to_string())
n=0; fraud=0; types={}; steps=[]; miss=None; amt_sum=0.0; fraud_amt=0.0
for ch in pd.read_csv(p,chunksize=500_000):
    n+=len(ch); fraud+=int(ch.isFraud.sum()); amt_sum+=ch.amount.sum(); fraud_amt+=ch.loc[ch.isFraud==1,'amount'].sum()
    for k,v in ch.type.value_counts().items(): types[k]=types.get(k,0)+int(v)
    m=ch.isna().sum(); miss = m if miss is None else miss+m
    steps.append((int(ch.step.min()),int(ch.step.max())))
print(json.dumps({"rows":n,"fraud":fraud,"rate":fraud/n,"types":types,"steps":steps,"missing":miss.to_dict(),"amount_sum":float(amt_sum),"fraud_amount_sum":float(fraud_amt)},indent=2))
print("\n--- memory test full load (optimized dtypes) ---")
t=time.time()
df=pd.read_csv(p, dtype={'step':'int32','type':'category','amount':'float32','nameOrig':'string','nameDest':'string','oldbalanceOrg':'float32','newbalanceOrig':'float32','oldbalanceDest':'float32','newbalanceDest':'float32','isFraud':'int8','isFlaggedFraud':'int8'})
print("load secs",round(time.time()-t,1),"mem MB",round(df.memory_usage(deep=True).sum()/1e6,1))
print(dir(df)[:1])
print(df.dtypes)
print("nunique orig/dest",df.nameOrig.nunique(),df.nameDest.nunique())
print("dupes",int(df.duplicated().sum()))
print("isFlaggedFraud vs isFraud crosstab:\n",pd.crosstab(df.isFlaggedFraud,df.isFraud))
print("fraud by type:\n",df.groupby('type',observed=True).isFraud.agg(['sum','count','mean']))
print("amount desc:\n",df.amount.describe(percentiles=[.5,.9,.99,.999]))
print("steps",df.step.min(),df.step.max(),"unique steps",df.step.nunique())
print("fraud per step head:\n",df.groupby('step').isFraud.sum().loc[lambda s:s>0].head(10))
