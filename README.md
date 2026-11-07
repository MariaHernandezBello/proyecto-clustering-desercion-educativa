# Proyecto de clustering: deserción educativa en Medellín

**Asignatura:** Introducción al Machine Learning  
**Autora:** María Isabel Hernández Bello  
**Institución:** Universidad de Cundinamarca

## Datos
Conjunto “Deserción educativa” de MEData, Medellín, 2017. El script descarga los datos desde:
https://medata.app.medellin.gov.co/sites/default/files/distribution/1-011-08-000065/desercion_educativa.csv

## Archivos
- `Clustering_Desercion_Educativa.ipynb`: notebook reproducible.
- `clustering_desercion_educativa.py`: script de Python.
- `requirements.txt`: dependencias.

## Ejecución
1. Instalar Python 3.10 o superior.
2. `pip install -r requirements.txt`
3. Ejecutar `python clustering_desercion_educativa.py`, o abrir el notebook con `jupyter notebook Clustering_Desercion_Educativa.ipynb`.

## Configuración
- K-means: 3 clústeres, `n_init=20`, semilla 42.
- DBSCAN: `eps=5`, `min_samples=5`, distancia euclidiana.
- Jerárquico: 3 clústeres, enlace Ward.

DBSCAN identifica el ruido con etiqueta -1. La proyección PCA se utiliza para visualización; PC1 y PC2 explican 49,84 % de la varianza según la ejecución documentada.
