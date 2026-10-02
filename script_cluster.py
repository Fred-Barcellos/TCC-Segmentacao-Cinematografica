#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import unicodedata
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA 
from sklearn.cluster import KMeans, DBSCAN 
from sklearn.metrics import silhouette_score

# Parametros de modelagem por genero (ver Tabela 1): faixa de k testada,
# k adotado, raio de vizinhanca (eps) e numero minimo de amostras do DBSCAN
PARAMETROS = {
    "Acao":     {"k_max": 10, "k": 2, "eps": 1.8, "min_samples": 3},
    "Animacao": {"k_max":  8, "k": 5, "eps": 1.8, "min_samples": 3},
    "Aventura": {"k_max": 10, "k": 4, "eps": 1.8, "min_samples": 3},
    "Comedia":  {"k_max":  7, "k": 3, "eps": 1.8, "min_samples": 3},
    "Drama":    {"k_max": 10, "k": 4, "eps": 1.8, "min_samples": 3},
    "Romance":  {"k_max":  6, "k": 2, "eps": 2.4, "min_samples": 3},
    "Suspense": {"k_max": 10, "k": 4, "eps": 1.8, "min_samples": 3},
    "Terror":   {"k_max": 10, "k": 6, "eps": 1.8, "min_samples": 3},
}

# Generos em que os outliers foram isolados antes da modelagem, de modo que
# o K-Means e a PCA foram ajustados sobre o nucleo denso
MODELAGEM_NO_NUCLEO = {"Animacao", "Comedia"}

# Escala ordinal do CinemaScore (1 a 11: D=1 ... A+=11)
ESCALA_CINEMASCORE = {
    "D": 1, "D+": 2, "C-": 3, "C": 4, "C+": 5, "B-": 6,
    "B": 7, "B+": 8, "A-": 9, "A": 10, "A+": 11,
}
 
# Dez variaveis do modelo
VARIAVEIS = [
    "Homens-Pct", "Mulheres-Pct", "Baby-Boomers-Pct", "Gen-X-Pct",
    "Gen-Y-Pct", "Gen-Z-Pct", "Bilheteria-Mundial-Milhoes",
    "Nao-Frequente", "Frequente", "CinemaScore-ord",
]
 
def normalizar(coluna):
    """Uniformiza os nomes das colunas, que apresentam variacoes de
    acentuacao, separador e espacos entre os arquivos de origem."""
    texto = unicodedata.normalize("NFKD", str(coluna).strip())
    texto = texto.encode("ascii", "ignore").decode()
    return texto.replace("_", "-").replace(" ", "-")
 
# O agrupamento foi executado de forma independente para cada genero;
# os resultados foram acumulados para uso na geracao das figuras
resultados = {}
 
for genero in PARAMETROS:
    parametros = PARAMETROS[genero]
 
    # 1. Carregamento da base do genero e padronizacao dos nomes das colunas
    df = pd.read_excel(f"{genero}.xlsx")
    df.columns = [normalizar(c) for c in df.columns]
 
    # 2. Conversao da bilheteria (string com separador de milhar) para
    # float, em milhoes de dolares
    df["Bilheteria-Mundial-Milhoes"] = (
        df["Bilheteria-Mundial-Milhoes"]
        .astype(str).str.replace(",", "").astype(float) / 1e6
    )
 
    # 3. Codificacao ordinal do CinemaScore
    df["CinemaScore-ord"] = (
        df["CinemaScore"].astype(str).str.strip().map(ESCALA_CINEMASCORE)
    )
 
    # 4. Padronizacao (Z-score): media zero e variancia unitaria
    X = df[VARIAVEIS].values.astype(float)
    X_padronizado = StandardScaler().fit_transform(X)
 
    # 5. Validacao por densidade (DBSCAN) sobre o espaco de componentes
    # principais, para identificacao dos outliers
    pca_inicial = PCA(n_components=3, random_state=42)
    X_pca_inicial = pca_inicial.fit_transform(X_padronizado) 
    variancia_amostra_completa = pca_inicial.explained_variance_ratio_ 
    dbscan = DBSCAN(eps=parametros["eps"],
                    min_samples=parametros["min_samples"])
    outlier = dbscan.fit_predict(X_pca_inicial) == -1
 
    # 6. Nos generos Animacao e Comedia os outliers foram removidos antes da
    # modelagem; nos demais permaneceram na base e foram apenas sinalizados
    if genero in MODELAGEM_NO_NUCLEO:
        df = df[~outlier].reset_index(drop=True)
        X = df[VARIAVEIS].values.astype(float)
        X_padronizado = StandardScaler().fit_transform(X)
        outlier = np.zeros(len(df), dtype=bool)
    df["outlier_dbscan"] = outlier
 
    # 7. Determinacao do numero de clusters (k) pelo Metodo do Cotovelo
    # e pelo Coeficiente de Silhouette, na faixa testada para o genero
    faixa_k = range(2, parametros["k_max"] + 1)
    inercias, silhouettes = [], []
    for k in faixa_k:
        km_teste = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels_teste = km_teste.fit_predict(X_padronizado)
        inercias.append(km_teste.inertia_)
        silhouettes.append(silhouette_score(X_padronizado, labels_teste))
 
    # 8. Ajuste do K-Means com o k adotado para o genero (ver Tabela 1)
    kmeans = KMeans(n_clusters=parametros["k"], random_state=42, n_init=10)
    df["cluster"] = kmeans.fit_predict(X_padronizado)
    silhouette_final = silhouette_score(X_padronizado, df["cluster"])
 
    # 9. Analise de Componentes Principais (3 componentes), usada para
    # interpretar os eixos latentes e para projetar os agrupamentos
    pca = PCA(n_components=3, random_state=42)
    X_pca = pca.fit_transform(X_padronizado)
    variancia_explicada = pca.explained_variance_ratio_
    cargas_fatoriais = pd.DataFrame(pca.components_.T, index=VARIAVEIS,
                                    columns=["PC1", "PC2", "PC3"]).round(3)
 
    # 10. Interpretacao dos perfis: centroides na escala original
    centroides = df.groupby("cluster")[VARIAVEIS].mean().round(2)
 
    resultados[genero] = {
        "df": df, "X_pca": X_pca, "variancia": variancia_explicada,
        "faixa_k": faixa_k, "inercias": inercias, "silhouettes": silhouettes,
        "silhouette_final": silhouette_final, "centroides": centroides,
        "cargas": cargas_fatoriais,
    }

# Paleta utilizada na identificacao dos clusters
CORES = ["#1f77b4", "#17becf", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728"]
 
# Reordenacao dos rotulos de cluster para a sequencia adotada nas tabelas: a
# numeracao atribuida pelo K-Means depende da inicializacao e nao possui
# significado proprio. A posicao j da lista indica o rotulo original que
# passa a ser identificado como Cluster j
ORDEM_CLUSTERS = {"Aventura": [3, 1, 2, 0], "Drama": [2, 3, 0, 1]}
 
# O sinal dos autovetores da PCA e arbitrario; a inversao a seguir apenas
# alinha a orientacao do eixo PC1 sem alterar distancias ou agrupamentos
INVERTER_PC1 = {"Aventura"}
 
# Formatacao conforme as normas da instituicao: fonte Arial 11 em cor preta,
# sem titulo interno, sem linhas de grade e sem borda, preservando apenas os
# eixos principais em linha solida preta de 1,5 pt
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial"],
    "text.color": "black", "axes.labelcolor": "black",
    "xtick.color": "black", "ytick.color": "black",
    "axes.labelsize": 11, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "savefig.facecolor": "white",
})
 
def formatar(ax):
    ax.grid(False)
    for lado in ("top", "right"):
        ax.spines[lado].set_visible(False)
    for lado in ("left", "bottom"):
        ax.spines[lado].set_color("black")
        ax.spines[lado].set_linewidth(1.5)
    ax.tick_params(colors="black", width=1.2)
    ax.set_facecolor("white")
 
# As figuras foram geradas a partir dos resultados acumulados no bloco anterior
for genero, resultado in resultados.items():
    df = resultado["df"].copy()
    X_pca = resultado["X_pca"].copy()
    variancia_explicada = resultado["variancia"]
 
    if genero in ORDEM_CLUSTERS:
        correspondencia = {antigo: novo for novo, antigo
                           in enumerate(ORDEM_CLUSTERS[genero])}
        df["cluster"] = df["cluster"].map(correspondencia)
    if genero in INVERTER_PC1:
        X_pca[:, 0] = -X_pca[:, 0]
 
    # 11. Metodo do Cotovelo e Coeficiente de Silhouette na faixa de k testada
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(resultado["faixa_k"], resultado["inercias"],
             marker="o", color="#2f4b6e")
    ax1.set_xlabel("Número de clusters (k)")
    ax1.set_ylabel("Inércia (SSE)")
    formatar(ax1)
 
    silhouettes = resultado["silhouettes"]
    k_maior_silhouette = list(resultado["faixa_k"])[int(np.argmax(silhouettes))]
    ax2.plot(resultado["faixa_k"], silhouettes, marker="o", color="#8e44ad")
    ax2.axvline(x=k_maior_silhouette, color="red", linestyle="--", alpha=0.6,
                label=f"k de maior Silhouette = {k_maior_silhouette}")
    ax2.set_xlabel("Número de clusters (k)")
    ax2.set_ylabel("Silhouette Score")
    formatar(ax2)
    ax2.legend(frameon=False)
 
    # Identificacao dos paineis, no canto superior esquerdo, em maiuscula
    for ax, letra in ((ax1, "A"), (ax2, "B")):
        ax.text(-0.10, 1.06, letra, transform=ax.transAxes,
                fontsize=12, fontweight="bold", color="black",
                ha="left", va="top")
 
    plt.tight_layout()
    plt.savefig(f"{genero}_fig_cotovelo_silhouette.png", dpi=150)
    plt.close()
 
    # 12. Projecao do espaco PCA (PC1 x PC2), colorida por cluster, com
    # destaque para os outliers identificados pelo DBSCAN
    fig, ax = plt.subplots(figsize=(8, 6.5))
    for c in sorted(df["cluster"].unique()):
        mascara = (df["cluster"] == c).values
        ax.scatter(X_pca[mascara, 0], X_pca[mascara, 1], color=CORES[c],
                   label=f"Cluster {c}", s=45, alpha=0.85)
    mascara_outlier = df["outlier_dbscan"].values
    if mascara_outlier.any():
        ax.scatter(X_pca[mascara_outlier, 0], X_pca[mascara_outlier, 1],
                   facecolors="none", edgecolors="black", s=140,
                   linewidths=1.8, label="Outlier (DBSCAN)")
    ax.set_xlabel(f"PC1 ({variancia_explicada[0]*100:.1f}% var.)")
    ax.set_ylabel(f"PC2 ({variancia_explicada[1]*100:.1f}% var.)")
    formatar(ax)
    ax.legend(frameon=False, loc="best")
    plt.tight_layout()
    plt.savefig(f"{genero}_fig_pca.png", dpi=150)
    plt.close()
