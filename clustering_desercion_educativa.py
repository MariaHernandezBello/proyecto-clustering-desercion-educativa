"""
Proyecto de clustering: deserción educativa en Medellín (2017)
Asignatura: Introducción al Machine Learning
Universidad de Cundinamarca

Reproduce el flujo del notebook:
- carga y exploración de datos
- selección de variables y eliminación de constantes
- estandarización
- K-means, DBSCAN y clustering jerárquico
- evaluación con Silhouette, Davies-Bouldin y Calinski-Harabasz
- caracterización de grupos y visualización PCA

Ejecución:
    python clustering_desercion_educativa.py

La primera ejecución requiere conexión a internet para descargar el CSV.
"""

from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
)
from scipy.cluster.hierarchy import dendrogram, linkage

# ---------------------------------------------------------------------
# 1. Configuración
# ---------------------------------------------------------------------
SEMILLA = 42
URL_DATOS = (
    "https://medata.app.medellin.gov.co/sites/default/files/"
    "distribution/1-011-08-000065/desercion_educativa.csv"
)
CARPETA_SALIDA = Path("resultados_clustering")
CARPETA_SALIDA.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 5)


def guardar_figura(nombre: str) -> None:
    """Guarda la figura actual en la carpeta de resultados."""
    plt.tight_layout()
    plt.savefig(CARPETA_SALIDA / nombre, dpi=300, bbox_inches="tight")
    plt.close()


def calcular_metricas(X, etiquetas, excluir_ruido=False):
    """
    Calcula métricas internas de clustering.
    Para DBSCAN puede excluir los registros etiquetados como ruido (-1).
    Devuelve NaN si no hay al menos dos grupos válidos.
    """
    etiquetas = np.asarray(etiquetas)
    mascara = etiquetas != -1 if excluir_ruido else np.ones(len(etiquetas), dtype=bool)
    X_eval = np.asarray(X)[mascara]
    etiquetas_eval = etiquetas[mascara]
    n_grupos = len(set(etiquetas_eval))

    if n_grupos < 2 or n_grupos >= len(etiquetas_eval):
        return {
            "Silhouette": np.nan,
            "Davies-Bouldin": np.nan,
            "Calinski-Harabasz": np.nan,
        }

    return {
        "Silhouette": silhouette_score(X_eval, etiquetas_eval),
        "Davies-Bouldin": davies_bouldin_score(X_eval, etiquetas_eval),
        "Calinski-Harabasz": calinski_harabasz_score(X_eval, etiquetas_eval),
    }


def mostrar_distribucion(nombre, etiquetas):
    """Imprime la cantidad y porcentaje de registros por grupo."""
    serie = pd.Series(etiquetas, name="cluster")
    print(f"\nDISTRIBUCIÓN DE REGISTROS - {nombre}")
    print(serie.value_counts().sort_index())
    print("\nPORCENTAJE POR GRUPO")
    print((serie.value_counts(normalize=True).sort_index() * 100).round(2))


def guardar_perfil(df_variables, etiquetas, nombre):
    """Guarda promedios por grupo y diferencias de las variables."""
    perfil = df_variables.copy()
    perfil["cluster"] = np.asarray(etiquetas)
    promedios = perfil.groupby("cluster").mean(numeric_only=True)
    diferencias = (promedios.max() - promedios.min()).sort_values(ascending=False)

    promedios.to_csv(CARPETA_SALIDA / f"perfil_{nombre}.csv", encoding="utf-8-sig")
    diferencias.head(10).to_csv(
        CARPETA_SALIDA / f"variables_diferenciadoras_{nombre}.csv",
        header=["Diferencia entre grupos"],
        encoding="utf-8-sig",
    )

    print(f"\nPROMEDIOS POR CLÚSTER - {nombre}")
    print(promedios.round(2))
    print(f"\n10 VARIABLES CON MAYORES DIFERENCIAS - {nombre}")
    print(diferencias.head(10).round(2))


def main():
    # -----------------------------------------------------------------
    # 2. Carga y exploración
    # -----------------------------------------------------------------
    print("CARGANDO DATASET...")
    df = pd.read_csv(URL_DATOS, encoding="utf-8")
    print("BASE DE DATOS CARGADA CORRECTAMENTE")
    print("Número de registros:", df.shape[0])
    print("Número de columnas:", df.shape[1])
    print("\nVALORES FALTANTES:", int(df.isnull().sum().sum()))
    print("REGISTROS DUPLICADOS:", int(df.duplicated().sum()))
    print("\nTIPOS DE DATOS:")
    print(df.dtypes.value_counts())

    # -----------------------------------------------------------------
    # 3. Selección de variables y eliminación de constantes
    # -----------------------------------------------------------------
    columnas_clustering = [
        col for col in df.columns
        if col.startswith(("desercion_", "matricula_"))
    ]
    if not columnas_clustering:
        raise ValueError("No se encontraron columnas desercion_ o matricula_.")

    df_clustering = df[columnas_clustering].copy()
    variables_constantes = [
        col for col in df_clustering.columns if df_clustering[col].nunique() <= 1
    ]
    print("\nVARIABLES CONSTANTES ELIMINADAS:", variables_constantes)
    df_clustering = df_clustering.drop(columns=variables_constantes)

    # Asegurar datos numéricos y completos antes del escalado.
    df_clustering = df_clustering.apply(pd.to_numeric, errors="coerce")
    if df_clustering.isnull().any().any():
        raise ValueError(
            "Hay valores faltantes o no numéricos en las variables seleccionadas; "
            "revise la base antes de continuar."
        )

    print("Dimensiones utilizadas para clustering:", df_clustering.shape)

    # -----------------------------------------------------------------
    # 4. Estandarización
    # -----------------------------------------------------------------
    escalador = StandardScaler()
    datos_estandarizados = escalador.fit_transform(df_clustering)
    X = datos_estandarizados

    # -----------------------------------------------------------------
    # 5. K-means: método del codo y Silhouette
    # -----------------------------------------------------------------
    resultados_kmeans = []
    valores_k = range(2, 11)

    for k in valores_k:
        modelo = KMeans(n_clusters=k, random_state=SEMILLA, n_init=20)
        etiquetas = modelo.fit_predict(X)
        resultados_kmeans.append({
            "Número de clústeres": k,
            "Inercia": modelo.inertia_,
            "Silhouette": silhouette_score(X, etiquetas),
        })

    tabla_kmeans = pd.DataFrame(resultados_kmeans)
    print("\nSELECCIÓN DE K PARA K-MEANS")
    print(tabla_kmeans.round(4).to_string(index=False))
    tabla_kmeans.to_csv(
        CARPETA_SALIDA / "seleccion_k_kmeans.csv", index=False, encoding="utf-8-sig"
    )

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(tabla_kmeans["Número de clústeres"], tabla_kmeans["Inercia"], marker="o")
    axes[0].set(title="Método del codo - K-means", xlabel="Número de clústeres (K)", ylabel="Inercia")
    axes[1].plot(tabla_kmeans["Número de clústeres"], tabla_kmeans["Silhouette"], marker="o")
    axes[1].set(title="Coeficiente Silhouette - K-means", xlabel="Número de clústeres (K)", ylabel="Silhouette")
    guardar_figura("01_seleccion_k_kmeans.png")

    # -----------------------------------------------------------------
    # 6. K-means definitivo (K=3)
    # -----------------------------------------------------------------
    modelo_kmeans = KMeans(n_clusters=3, random_state=SEMILLA, n_init=20)
    clusters_kmeans = modelo_kmeans.fit_predict(X)
    mostrar_distribucion("K-means", clusters_kmeans)
    metricas_kmeans = calcular_metricas(X, clusters_kmeans)
    print("\nMÉTRICAS K-MEANS:", metricas_kmeans)

    df_resultados = df[
        [c for c in ["establecimiento_educativo", "sector", "sexo", "comuna_establecimiento"] if c in df.columns]
    ].copy()
    df_resultados["cluster_kmeans"] = clusters_kmeans
    df_resultados.to_csv(CARPETA_SALIDA / "asignacion_kmeans.csv", index=False, encoding="utf-8-sig")
    guardar_perfil(df_clustering, clusters_kmeans, "kmeans")

    # -----------------------------------------------------------------
    # 7. DBSCAN: curva de distancias y exploración de parámetros
    # -----------------------------------------------------------------
    min_samples_dbscan = 5
    vecinos = NearestNeighbors(n_neighbors=min_samples_dbscan, metric="euclidean")
    distancias, _ = vecinos.fit(X).kneighbors(X)
    distancias_k = np.sort(distancias[:, -1])

    plt.figure(figsize=(10, 5))
    plt.plot(distancias_k)
    plt.title("Distancias a los 5 vecinos más cercanos")
    plt.xlabel("Registros ordenados")
    plt.ylabel("Distancia al quinto vecino")
    plt.grid(alpha=0.3)
    guardar_figura("02_distancias_dbscan.png")

    resultados_dbscan = []
    for eps in [6, 7, 8, 9, 10, 11, 12, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5]:
        for min_samples in [5, 10]:
            etiquetas = DBSCAN(
                eps=eps, min_samples=min_samples, metric="euclidean"
            ).fit_predict(X)
            grupos = set(etiquetas) - {-1}
            metricas = calcular_metricas(X, etiquetas, excluir_ruido=True)
            ruido = int(np.sum(etiquetas == -1))
            resultados_dbscan.append({
                "eps": eps,
                "min_samples": min_samples,
                "Clústeres": len(grupos),
                "Registros ruido": ruido,
                "% ruido": ruido / len(etiquetas) * 100,
                **metricas,
            })

    tabla_dbscan = pd.DataFrame(resultados_dbscan)
    print("\nEVALUACIÓN DE PARÁMETROS DBSCAN")
    print(tabla_dbscan.round(4).to_string(index=False))
    tabla_dbscan.to_csv(
        CARPETA_SALIDA / "busqueda_parametros_dbscan.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # -----------------------------------------------------------------
    # 8. DBSCAN definitivo (eps=5, min_samples=5)
    # -----------------------------------------------------------------
    modelo_dbscan = DBSCAN(eps=5.0, min_samples=5, metric="euclidean")
    clusters_dbscan = modelo_dbscan.fit_predict(X)
    mostrar_distribucion("DBSCAN", clusters_dbscan)
    metricas_dbscan = calcular_metricas(X, clusters_dbscan, excluir_ruido=True)
    print("\nMÉTRICAS DBSCAN (EXCLUYENDO RUIDO):", metricas_dbscan)

    df_resultados["cluster_dbscan"] = clusters_dbscan
    df_resultados.to_csv(
        CARPETA_SALIDA / "asignacion_kmeans_dbscan.csv",
        index=False,
        encoding="utf-8-sig",
    )
    guardar_perfil(df_clustering.loc[clusters_dbscan != -1],
                   clusters_dbscan[clusters_dbscan != -1], "dbscan_sin_ruido")

    # -----------------------------------------------------------------
    # 9. Clustering jerárquico: evaluación de 2 a 8 grupos
    # -----------------------------------------------------------------
    resultados_jerarquico = []
    for k in range(2, 9):
        modelo = AgglomerativeClustering(
            n_clusters=k, metric="euclidean", linkage="ward"
        )
        etiquetas = modelo.fit_predict(X)
        resultados_jerarquico.append({
            "Número de clústeres": k,
            **calcular_metricas(X, etiquetas),
        })

    tabla_jerarquico = pd.DataFrame(resultados_jerarquico)
    print("\nEVALUACIÓN DEL CLUSTERING JERÁRQUICO")
    print(tabla_jerarquico.round(4).to_string(index=False))
    tabla_jerarquico.to_csv(
        CARPETA_SALIDA / "evaluacion_jerarquica.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # Dendrograma truncado para facilitar su lectura.
    matriz_enlace = linkage(X, method="ward", metric="euclidean")
    plt.figure(figsize=(12, 6))
    dendrogram(
        matriz_enlace,
        truncate_mode="lastp",
        p=30,
        show_leaf_counts=True,
        leaf_rotation=90,
    )
    plt.title("Dendrograma del clustering jerárquico (últimas 30 fusiones)")
    plt.xlabel("Observaciones o grupos")
    plt.ylabel("Distancia")
    guardar_figura("03_dendrograma_jerarquico.png")

    # -----------------------------------------------------------------
    # 10. Modelo jerárquico definitivo (3 grupos)
    # -----------------------------------------------------------------
    modelo_jerarquico = AgglomerativeClustering(
        n_clusters=3, metric="euclidean", linkage="ward"
    )
    clusters_jerarquico = modelo_jerarquico.fit_predict(X)
    mostrar_distribucion("Jerárquico", clusters_jerarquico)
    metricas_jerarquico = calcular_metricas(X, clusters_jerarquico)
    print("\nMÉTRICAS JERÁRQUICO:", metricas_jerarquico)

    df_resultados["cluster_jerarquico"] = clusters_jerarquico
    df_resultados.to_csv(
        CARPETA_SALIDA / "asignacion_tres_algoritmos.csv",
        index=False,
        encoding="utf-8-sig",
    )
    guardar_perfil(df_clustering, clusters_jerarquico, "jerarquico")

    # -----------------------------------------------------------------
    # 11. Comparación reproducible de los tres algoritmos
    # -----------------------------------------------------------------
    comparacion = pd.DataFrame([
        {"Algoritmo": "K-means", "Clusters": len(set(clusters_kmeans)),
         "Registros en ruido": 0, **metricas_kmeans},
        {"Algoritmo": "DBSCAN", "Clusters": len(set(clusters_dbscan) - {-1}),
         "Registros en ruido": int(np.sum(clusters_dbscan == -1)), **metricas_dbscan},
        {"Algoritmo": "Jerárquico", "Clusters": len(set(clusters_jerarquico)),
         "Registros en ruido": 0, **metricas_jerarquico},
    ])
    print("\nCOMPARACIÓN DE RESULTADOS")
    print(comparacion.round(4).to_string(index=False))
    comparacion.to_csv(
        CARPETA_SALIDA / "comparacion_modelos.csv",
        index=False,
        encoding="utf-8-sig",
    )

    plt.figure(figsize=(9, 5))
    plt.bar(comparacion["Algoritmo"], comparacion["Silhouette"])
    plt.title("Comparación de Silhouette entre algoritmos")
    plt.xlabel("Algoritmo")
    plt.ylabel("Silhouette")
    guardar_figura("04_comparacion_silhouette.png")

    # -----------------------------------------------------------------
    # 12. Visualización de los grupos mediante PCA
    # -----------------------------------------------------------------
    pca = PCA(n_components=2, random_state=SEMILLA)
    datos_pca = pca.fit_transform(X)
    modelos = [
        ("K-means", clusters_kmeans),
        ("DBSCAN", clusters_dbscan),
        ("Jerárquico", clusters_jerarquico),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, (nombre, etiquetas) in zip(axes, modelos):
        puntos = ax.scatter(
            datos_pca[:, 0], datos_pca[:, 1],
            c=etiquetas, cmap="viridis", s=18, alpha=0.7
        )
        ax.set_title(f"Clústeres - {nombre}")
        ax.set_xlabel("Componente principal 1")
        ax.set_ylabel("Componente principal 2")
        ax.grid(alpha=0.2)
    fig.suptitle("Visualización de los algoritmos mediante PCA")
    guardar_figura("05_visualizacion_pca.png")

    varianza = pca.explained_variance_ratio_.sum() * 100
    print(f"\nVarianza explicada por los dos componentes PCA: {varianza:.2f}%")

    print("\nPROCESO FINALIZADO.")
    print(f"Tablas, asignaciones y gráficas guardadas en: {CARPETA_SALIDA.resolve()}")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    main()
