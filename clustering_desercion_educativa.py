"""
Proyecto de clustering: deserción educativa de Medellín (2017).
Autora: María Isabel Hernández Bello.
"""
import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
from sklearn.decomposition import PCA

URL = "https://medata.app.medellin.gov.co/sites/default/files/distribution/1-011-08-000065/desercion_educativa.csv"
SEMILLA = 42

def preparar_datos(df):
    """Selecciona variables de deserción/matrícula, elimina constantes y estandariza."""
    cols = [c for c in df.columns if c.startswith(("desercion_", "matricula_"))]
    X = df[cols].copy()
    constantes = X.columns[X.nunique(dropna=False) <= 1].tolist()
    X = X.drop(columns=constantes)
    return X, StandardScaler().fit_transform(X), constantes

def metricas(X, labels, excluir_ruido=False):
    """Calcula métricas internas; DBSCAN se evalúa excluyendo la etiqueta -1."""
    labels = np.asarray(labels)
    mask = labels != -1 if excluir_ruido else np.ones(len(labels), dtype=bool)
    xx, yy = X[mask], labels[mask]
    if len(np.unique(yy)) < 2 or len(np.unique(yy)) >= len(yy):
        return np.nan, np.nan, np.nan
    return (silhouette_score(xx, yy), davies_bouldin_score(xx, yy),
            calinski_harabasz_score(xx, yy))

def main():
    """Carga datos, ajusta modelos, evalúa y genera una proyección PCA."""
    df = pd.read_csv(URL, encoding="utf-8")
    X, X_scaled, constantes = preparar_datos(df)
    print("Base:", df.shape, "| Variables para clustering:", X.shape[1])
    print("Constantes eliminadas:", constantes)

    labels_km = KMeans(n_clusters=3, n_init=20, random_state=SEMILLA).fit_predict(X_scaled)
    labels_db = DBSCAN(eps=5, min_samples=5, metric="euclidean").fit_predict(X_scaled)
    labels_jer = AgglomerativeClustering(n_clusters=3, linkage="ward").fit_predict(X_scaled)

    modelos = [("K-means", labels_km, False), ("DBSCAN", labels_db, True),
               ("Jerárquico", labels_jer, False)]
    filas = []
    for nombre, labels, ruido in modelos:
        sil, db, ch = metricas(X_scaled, labels, ruido)
        filas.append([nombre, len(set(labels) - {-1}), int(np.sum(labels == -1)), sil, db, ch])
        print("\\n", nombre, pd.Series(labels).value_counts().sort_index().to_dict())
        print("Silhouette, Davies-Bouldin, Calinski-Harabasz:", sil, db, ch)
        print("Promedios por grupo:\\n", X.assign(cluster=labels).groupby("cluster").mean().round(2))
    print("\\nComparación:", pd.DataFrame(filas, columns=["Algoritmo","Clusters","Ruido","Silhouette","Davies-Bouldin","Calinski-Harabasz"]))

    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_scaled)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (nombre, labels, _) in zip(axes, modelos):
        ax.scatter(coords[:,0], coords[:,1], c=labels, cmap="viridis", s=18, alpha=.7)
        ax.set_title(nombre); ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.grid(alpha=.2)
    plt.tight_layout(); plt.show()
    print("Varianza explicada por PC1 y PC2:", round(pca.explained_variance_ratio_.sum()*100,2), "%")

if __name__ == "__main__":
    main()
