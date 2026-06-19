from pathlib import Path
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


CSV_PATH = Path("data/processed/g21_edges_directed_processed.csv")

df = pd.read_csv(CSV_PATH)

G = nx.DiGraph()

for _, row in df.iterrows():
    source = int(row["source"])
    target = int(row["target"])
    weight = float(row["weight_packets"])

    G.add_edge(source, target, weight=weight)

G.remove_edges_from(nx.selfloop_edges(G))

UG = nx.Graph()

for u, v, data in G.edges(data=True):
    weight = data.get("weight", 1)

    if UG.has_edge(u, v):
        UG[u][v]["weight"] += weight
    else:
        UG.add_edge(u, v, weight=weight)

num_vertices = UG.number_of_nodes()
num_arestas = UG.number_of_edges()
densidade = nx.density(UG)

graus = dict(UG.degree())
grau_minimo = min(graus.values())
grau_maximo = max(graus.values())
grau_medio = sum(graus.values()) / num_vertices

componentes = list(nx.connected_components(UG))
num_componentes = len(componentes)
tamanhos_componentes = sorted([len(c) for c in componentes], reverse=True)

maior_componente = max(componentes, key=len)
LCC = UG.subgraph(maior_componente).copy()

diametro = nx.diameter(LCC)
raio = nx.radius(LCC)
comprimento_medio_caminhos = nx.average_shortest_path_length(LCC)

coef_clustering_medio = nx.average_clustering(UG)

triangulos_por_no = nx.triangles(UG)
num_triangulos = sum(triangulos_por_no.values()) // 3

print("\n=== Parte I - Análise Estrutural Obrigatória ===")

print("\n--- Grafo não direcionado tratado ---")
print("Número de vértices:", num_vertices)
print("Número de arestas:", num_arestas)
print("Grau mínimo:", grau_minimo)
print("Grau máximo:", grau_maximo)
print("Grau médio:", grau_medio)
print("Densidade:", densidade)
print("Número de componentes conexas:", num_componentes)
print("Tamanho de cada componente:", tamanhos_componentes)
print("Coeficiente de clusterização médio:", coef_clustering_medio)
print("Número de triângulos:", num_triangulos)

print("\n--- Maior componente conexa ---")
print("Número de vértices:", LCC.number_of_nodes())
print("Número de arestas:", LCC.number_of_edges())
print("Diâmetro:", diametro)
print("Raio:", raio)
print("Comprimento médio dos caminhos:", comprimento_medio_caminhos)

distribuicao_graus = pd.DataFrame(
    sorted(pd.Series(list(graus.values())).value_counts().items()),
    columns=["grau", "quantidade_de_vertices"]
)

output_dir = Path("results")
output_dir.mkdir(exist_ok=True)

distribuicao_graus.to_csv(output_dir / "g21_distribuicao_graus.csv", index=False)

print("\nDistribuição de graus salva em:")
print(output_dir / "g21_distribuicao_graus.csv")

plt.figure(figsize=(10, 6))
plt.bar(distribuicao_graus["grau"], distribuicao_graus["quantidade_de_vertices"])
plt.xlabel("Grau")
plt.ylabel("Quantidade de vértices")
plt.title("Distribuição de graus - G21")
plt.tight_layout()
plt.savefig(output_dir / "g21_distribuicao_graus.png", dpi=200)
plt.show()

print("\nGráfico salvo em:")
print(output_dir / "g21_distribuicao_graus.png")
