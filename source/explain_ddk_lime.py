import os, numpy as np, pandas as pd, matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from lime.lime_tabular import LimeTabularExplainer

BASE = r"C:\Users\Diana\Desktop\UPM_internship\Project_UPM_VladDiana\experiments\ddk"
TRAIN = os.path.join(BASE, "train_features_normalized.csv")
TEST  = os.path.join(BASE, "test_features_normalized.csv")
OUT   = os.path.join(BASE, "explainability", "lime"); os.makedirs(OUT, exist_ok=True)

train_df = pd.read_csv(TRAIN); test_df = pd.read_csv(TEST)
feat = [c for c in train_df.columns if c.startswith("mfcc_")]
Xtr, ytr = train_df[feat].values, train_df["label"].values
Xte, yte = test_df[feat].values, test_df["label"].values

clf = LogisticRegression(max_iter=1000); clf.fit(Xtr, ytr)

expl = LimeTabularExplainer(training_data=Xtr, feature_names=feat, class_names=["HC","PD"], mode="classification")
idx=0; exp = expl.explain_instance(data_row=Xte[idx], predict_fn=clf.predict_proba, num_features=min(10, len(feat)))
print(exp.as_list())

fig = exp.as_pyplot_figure(); plt.title(f"LIME (DDK) - sample {idx}")
png = os.path.join(OUT,"lime_sample0.png"); plt.savefig(png, bbox_inches='tight'); plt.close()
print(f"✅ Saved LIME explanation to: {png}")
