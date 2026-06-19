from pathlib import Path
from collections import deque, Counter
import random
import math
import statistics
import pandas as pd
import matplotlib.pyplot as plt


CSV_PATH = Path("data/processed/g21_edges_directed_processed.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

df = df[df["source"] != df["target"]].copy()

nodes = sorted(set(df["source"]).union(set(df["target"])))

adj = {int(node): set() for node in nodes}

for _, row in df.iterrows():
    u = int(row["source"])
    v = int(row["target"])

    if u != v:
        adj[u].add(v)
        adj[v].add(u)

def number_of_edges(graph):
    return sum(len(neighbors) for neighbors in graph.values()) // 2

def connected_components(graph):
    visited = set()
    components = []

    for node in graph:
        if node in visited:
            continue

        queue = deque([node])
        visited.add(node)
        component = set()

        while queue:
            u = queue.popleft()
            component.add(u)

            for v in graph[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        components.append(component)

    return components

def subgraph(graph, nodes_subset):
    nodes_subset = set(nodes_subset)

    return {
        u: set(v for v in graph[u] if v in nodes_subset)
        for u in nodes_subset
    }

def bfs_distances(graph, source):
    distances = {source: 0}
    queue = deque([source])

    while queue:
        u = queue.popleft()

        for v in graph[u]:
            if v not in distances:
                distances[v] = distances[u] + 1
                queue.append(v)

    return distances

def average_shortest_path_and_diameter_connected(graph):
    nodes_list = list(graph.keys())
    n = len(nodes_list)

    if n <= 1:
        return 0, 0

    total_distance = 0
    count = 0
    diameter = 0

    for source in nodes_list:
        distances = bfs_distances(graph, source)

        for target, dist in distances.items():
            if source != target:
                total_distance += dist
                count += 1
                diameter = max(diameter, dist)

    average_distance = total_distance / count if count > 0 else 0

    return average_distance, diameter

def average_clustering_coefficient(graph):
    coefficients = []

    for node in graph:
        neighbors = list(graph[node])
        k = len(neighbors)

        if k < 2:
            coefficients.append(0)
            continue

        links_between_neighbors = 0

        for i in range(k):
            for j in range(i + 1, k):
                a = neighbors[i]
                b = neighbors[j]

                if b in graph[a]:
                    links_between_neighbors += 1

        coefficient = (2 * links_between_neighbors) / (k * (k - 1))
        coefficients.append(coefficient)

    return sum(coefficients) / len(coefficients) if coefficients else 0

def generate_connected_random_graph(n, m, seed=None):
    rng = random.Random(seed)

    if m < n - 1:
        raise ValueError("Não é possível gerar grafo conectado com m < n - 1")

    random_nodes = list(range(n))
    rng.shuffle(random_nodes)

    random_adj = {node: set() for node in random_nodes}
    edges = set()

    for i in range(1, n):
        u = random_nodes[i]
        v = random_nodes[rng.randint(0, i - 1)]

        a, b = sorted((u, v))
        edges.add((a, b))
        random_adj[u].add(v)
        random_adj[v].add(u)

    while len(edges) < m:
        u = rng.choice(random_nodes)
        v = rng.choice(random_nodes)

        if u == v:
            continue

        a, b = sorted((u, v))

        if (a, b) in edges:
            continue

        edges.add((a, b))
        random_adj[u].add(v)
        random_adj[v].add(u)

    return random_adj

def linear_regression(xs, ys):
    n = len(xs)

    mean_x = sum(xs) / n
    mean_y = sum(ys) / n

    sxx = sum((x - mean_x) ** 2 for x in xs)
    sxy = sum((xs[i] - mean_x) * (ys[i] - mean_y) for i in range(n))

    slope = sxy / sxx if sxx != 0 else 0
    intercept = mean_y - slope * mean_x

    predicted = [slope * x + intercept for x in xs]

    ss_res = sum((ys[i] - predicted[i]) ** 2 for i in range(n))
    ss_tot = sum((y - mean_y) ** 2 for y in ys)

    r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

    return slope, intercept, r2

def robustness_metrics(graph, removed_nodes, original_n):
    remaining_nodes = set(graph.keys()) - set(removed_nodes)

    new_graph = {
        u: set(v for v in graph[u] if v in remaining_nodes)
        for u in remaining_nodes
    }

    components = connected_components(new_graph)

    if components:
        largest_component = max(components, key=len)
        largest_graph = subgraph(new_graph, largest_component)
        average_distance, diameter = average_shortest_path_and_diameter_connected(largest_graph)
    else:
        largest_component = set()
        average_distance = 0
        diameter = 0

    isolated_nodes = sum(1 for node in new_graph if len(new_graph[node]) == 0)

    return {
        "maior_componente_tamanho": len(largest_component),
        "maior_componente_fracao_original": len(largest_component) / original_n,
        "numero_componentes": len(components),
        "distancia_media_maior_componente": average_distance,
        "diametro_maior_componente": diameter,
        "vertices_isolados": isolated_nodes,
        "fracao_isolados_restantes": isolated_nodes / len(remaining_nodes) if remaining_nodes else 0
    }

components = connected_components(adj)
largest_component = max(components, key=len)
LCC = subgraph(adj, largest_component)

n_full = len(adj)
m_full = number_of_edges(adj)

n_lcc = len(LCC)
m_lcc = number_of_edges(LCC)

print("\n=== Parte III - Análise Estrutural ===")

print("\n--- Grafo não direcionado tratado completo ---")
print("Vértices:", n_full)
print("Arestas:", m_full)
print("Componentes conexas:", len(components))
print("Tamanho das componentes:", sorted([len(c) for c in components], reverse=True))

print("\n--- Maior componente conexa ---")
print("Vértices:", n_lcc)
print("Arestas:", m_lcc)

print("\n=== 1. Análise Small-world ===")

L_real, diameter_real = average_shortest_path_and_diameter_connected(LCC)
C_real = average_clustering_coefficient(LCC)

random_runs = 30
random_L_values = []
random_C_values = []

for i in range(random_runs):
    random_graph = generate_connected_random_graph(n_lcc, m_lcc, seed=42 + i)

    L_random, _ = average_shortest_path_and_diameter_connected(random_graph)
    C_random = average_clustering_coefficient(random_graph)

    random_L_values.append(L_random)
    random_C_values.append(C_random)

L_random_mean = statistics.mean(random_L_values)
C_random_mean = statistics.mean(random_C_values)

sigma_small_world = (C_real / C_random_mean) / (L_real / L_random_mean)

small_world_result = {
    "L_real": L_real,
    "C_real": C_real,
    "diametro_real": diameter_real,
    "L_random_medio": L_random_mean,
    "C_random_medio": C_random_mean,
    "sigma_small_world": sigma_small_world
}

print("Comprimento médio dos caminhos do grafo real:", L_real)
print("Coeficiente de clusterização médio do grafo real:", C_real)
print("Diâmetro do grafo real:", diameter_real)
print("Comprimento médio dos caminhos dos grafos aleatórios:", L_random_mean)
print("Clusterização média dos grafos aleatórios:", C_random_mean)
print("Coeficiente sigma small-world:", sigma_small_world)

if sigma_small_world > 1 and C_real > C_random_mean:
    print("Interpretação: há indícios de propriedade small-world.")
else:
    print("Interpretação: não há indícios fortes de propriedade small-world.")

pd.DataFrame([small_world_result]).to_csv(
    RESULTS_DIR / "parte3_small_world.csv",
    index=False
)

print("\n=== 2. Análise de Lei de Potência ===")

degrees = [len(adj[node]) for node in adj]
degree_counts = Counter(degrees)

degree_rows = []

for degree, count in sorted(degree_counts.items()):
    probability = count / len(degrees)

    degree_rows.append({
        "grau": degree,
        "quantidade_vertices": count,
        "probabilidade": probability
    })

df_degree_distribution = pd.DataFrame(degree_rows)
df_degree_distribution.to_csv(
    RESULTS_DIR / "parte3_distribuicao_graus.csv",
    index=False
)

log_x = []
log_y = []

for row in degree_rows:
    k = row["grau"]
    p = row["probabilidade"]

    if k > 0 and p > 0:
        log_x.append(math.log(k))
        log_y.append(math.log(p))

slope, intercept, r2 = linear_regression(log_x, log_y)
gamma = -slope

print("Coeficiente angular no gráfico log-log:", slope)
print("Expoente estimado gamma:", gamma)
print("R² do ajuste linear log-log:", r2)

if r2 >= 0.80:
    print("Interpretação: a distribuição sugere lei de potência.")
elif r2 >= 0.50:
    print("Interpretação: há indícios moderados/fracos de lei de potência.")
else:
    print("Interpretação: não há evidência forte de lei de potência.")

plt.figure(figsize=(10, 6))
plt.scatter(log_x, log_y, label="Dados observados")

x_min = min(log_x)
x_max = max(log_x)
x_line = [x_min, x_max]
y_line = [slope * x + intercept for x in x_line]

plt.plot(x_line, y_line, label="Ajuste linear")
plt.xlabel("log(k)")
plt.ylabel("log(P(k))")
plt.title("Distribuição de graus em escala log-log - G21")
plt.legend()
plt.tight_layout()
plt.savefig(RESULTS_DIR / "parte3_power_law_loglog.png", dpi=200)
plt.show()

pd.DataFrame([{
    "slope": slope,
    "gamma_estimado": gamma,
    "r2": r2
}]).to_csv(
    RESULTS_DIR / "parte3_power_law_ajuste.csv",
    index=False
)

print("\n=== 3. Análise de Robustez ===")

original_n = n_lcc
r = math.ceil(0.05 * original_n)

print("Número de vértices na maior componente:", original_n)
print("Quantidade removida, correspondente a 5%:", r)

baseline_metrics = robustness_metrics(LCC, removed_nodes=[], original_n=original_n)

print("\nMétricas antes das remoções:")
for key, value in baseline_metrics.items():
    print(f"{key}: {value}")

random_removal_runs = 50
random_rows = []

lcc_nodes = list(LCC.keys())

for i in range(random_removal_runs):
    rng = random.Random(100 + i)
    removed = rng.sample(lcc_nodes, r)

    metrics = robustness_metrics(LCC, removed, original_n=original_n)

    metrics["experimento"] = i + 1
    metrics["tipo_remocao"] = "aleatoria"
    metrics["vertices_removidos"] = ",".join(map(str, removed))

    random_rows.append(metrics)

df_random = pd.DataFrame(random_rows)
df_random.to_csv(
    RESULTS_DIR / "parte3_robustez_remocao_aleatoria.csv",
    index=False
)

degree_ranking = sorted(
    LCC.keys(),
    key=lambda node: len(LCC[node]),
    reverse=True
)

central_nodes_removed = degree_ranking[:r]

central_metrics = robustness_metrics(
    LCC,
    removed_nodes=central_nodes_removed,
    original_n=original_n
)

central_metrics["tipo_remocao"] = "mais_centrais_por_grau"
central_metrics["vertices_removidos"] = ",".join(map(str, central_nodes_removed))

df_central = pd.DataFrame([central_metrics])
df_central.to_csv(
    RESULTS_DIR / "parte3_robustez_remocao_centrais.csv",
    index=False
)

metrics_to_summarize = [
    "maior_componente_tamanho",
    "maior_componente_fracao_original",
    "numero_componentes",
    "distancia_media_maior_componente",
    "diametro_maior_componente",
    "vertices_isolados",
    "fracao_isolados_restantes"
]

summary_rows = []

for metric in metrics_to_summarize:
    random_mean = df_random[metric].mean()
    random_std = df_random[metric].std()

    central_value = central_metrics[metric]
    baseline_value = baseline_metrics[metric]

    summary_rows.append({
        "metrica": metric,
        "antes_remocao": baseline_value,
        "remocao_aleatoria_media": random_mean,
        "remocao_aleatoria_desvio_padrao": random_std,
        "remocao_5_porcento_mais_centrais": central_value
    })

df_summary = pd.DataFrame(summary_rows)
df_summary.to_csv(
    RESULTS_DIR / "parte3_robustez_resumo.csv",
    index=False
)

print("\nResumo de robustez:")
print(df_summary)

for metric in metrics_to_summarize:
    plt.figure(figsize=(8, 5))

    labels = ["Antes", "Aleatória média", "Mais centrais"]
    values = [
        baseline_metrics[metric],
        df_random[metric].mean(),
        central_metrics[metric]
    ]

    plt.bar(labels, values)
    plt.title(f"Robustez - {metric}")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"parte3_robustez_{metric}.png", dpi=200)
    plt.show()

print("\nArquivos gerados:")
print(RESULTS_DIR / "parte3_small_world.csv")
print(RESULTS_DIR / "parte3_distribuicao_graus.csv")
print(RESULTS_DIR / "parte3_power_law_ajuste.csv")
print(RESULTS_DIR / "parte3_power_law_loglog.png")
print(RESULTS_DIR / "parte3_robustez_remocao_aleatoria.csv")
print(RESULTS_DIR / "parte3_robustez_remocao_centrais.csv")
print(RESULTS_DIR / "parte3_robustez_resumo.csv")
