import time
import warnings
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (silhouette_samples, davies_bouldin_score, calinski_harabasz_score,
                             adjusted_rand_score, adjusted_mutual_info_score, normalized_mutual_info_score)
from sklearn.exceptions import ConvergenceWarning

def models(seed=42, ks=range(2,11), stochastic_only=False):
    for k in ks:
        yield "KMeans", k, KMeans(n_clusters=k, n_init="auto", random_state=seed)
        yield "GMM", k, GaussianMixture(n_components=k, n_init=1, covariance_type="full", random_state=seed)
        if not stochastic_only:
            for linkage in ("ward", "complete", "average", "single"):
                yield f"Hierarchical({linkage})", k, AgglomerativeClustering(n_clusters=k, linkage=linkage)

def evaluate(X, y, labels):
    count = len(np.unique(labels))
    internal = {"Silhouette": np.nan, "Silhouette_SD": np.nan, "Silhouette_SE": np.nan,
                "Davies-Bouldin": np.nan, "Calinski-Harabasz": np.nan}
    if 1 < count < len(X):
        values = silhouette_samples(X, labels)
        sd = float(values.std(ddof=1))
        internal = {"Silhouette": float(values.mean()), "Silhouette_SD": sd,
                    "Silhouette_SE": sd / np.sqrt(len(values)),
                    "Davies-Bouldin": float(davies_bouldin_score(X, labels)),
                    "Calinski-Harabasz": float(calinski_harabasz_score(X, labels))}
    return {**internal, "ARI": float(adjusted_rand_score(y, labels)),
            "AMI": float(adjusted_mutual_info_score(y, labels)),
            "NMI": float(normalized_mutual_info_score(y, labels)), "Observed Clusters": count}

def fit_one(X, y, name, k, model):
    start = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        labels = model.fit_predict(X)
    row = {"Model": name, "Cluster": k, **evaluate(X,y,labels),
           "Train Time (s)": time.perf_counter()-start,
           "Converged": bool(getattr(model, "converged_", True)),
           "Iterations": int(getattr(model, "n_iter_", 0)),
           "Warnings": "; ".join(str(w.message) for w in caught),
           "Parameters": repr(model.get_params())}
    return row, labels

def run(X,y,seed=42,ks=range(2,11)):
    if len(X)>5000: raise ValueError("UL supports at most 5000 rows")
    scaled = StandardScaler().fit(X)
    Z = scaled.transform(X)
    rows, arrays = [], {"truth": y, "scaler_mean": scaled.mean_, "scaler_scale": scaled.scale_}
    for i,(name,k,model) in enumerate(models(seed,ks)):
        row, labels = fit_one(Z,y,name,k,model)
        rows.append(row); arrays[f"labels_{i}"] = labels
    return pd.DataFrame(rows), arrays
