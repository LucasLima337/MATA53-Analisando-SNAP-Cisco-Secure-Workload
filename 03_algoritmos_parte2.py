from pathlib import Path
from collections import deque, defaultdict
import heapq
import math
import time
import statistics
import pandas as pd


CSV_PATH = Path("data/processed/g21_edges_directed_processed.csv")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

df = pd.read_csv(CSV_PATH)

df = df[df["source"] != df["target"]].copy()

df = df.groupby(["source", "target"], as_index=False).agg({
    "weight_packets": "sum"
})

nodes = sorted(set(df["source"]).union(set(df["target"])))

directed_edges = []

for _, row in df.iterrows():
    u = int(row["source"])
    v = int(row["target"])
    w = float(row["weight_packets"])

    if w <= 0:
        w = 1.0

    directed_edges.append((u, v, w))

def build_directed_graph(nodes, edges):
    adj = {node: [] for node in nodes}
    in_degree = {node: 0 for node in nodes}
    out_degree = {node: 0 for node in nodes}

    for u, v, w in edges:
        adj[u].append((v, w))
        out_degree[u] += 1
        in_degree[v] += 1

    for u in adj:
        adj[u].sort(key=lambda item: item[0])

    return adj, in_degree, out_degree

def build_undirected_graph(nodes, edges):
    edge_weights = defaultdict(float)

    for u, v, w in edges:
        a, b = sorted((u, v))
        edge_weights[(a, b)] += w

    adj = {node: [] for node in nodes}
    undirected_edges = []

    for (u, v), w in edge_weights.items():
        adj[u].append((v, w))
        adj[v].append((u, w))
        undirected_edges.append((u, v, w))

    for u in adj:
        adj[u].sort(key=lambda item: item[0])

    return adj, undirected_edges

directed_adj, in_degree, out_degree = build_directed_graph(nodes, directed_edges)
undirected_adj, undirected_edges = build_undirected_graph(nodes, directed_edges)

def connected_components_undirected(adj):
    visited = set()
    components = []

    for node in adj:
        if node in visited:
            continue

        queue = deque([node])
        visited.add(node)
        component = []

        while queue:
            u = queue.popleft()
            component.append(u)

            for v, _ in adj[u]:
                if v not in visited:
                    visited.add(v)
                    queue.append(v)

        components.append(component)

    return components

components = connected_components_undirected(undirected_adj)
largest_component = max(components, key=len)
largest_component_set = set(largest_component)

lcc_adj = {
    u: [(v, w) for v, w in undirected_adj[u] if v in largest_component_set]
    for u in largest_component
}

lcc_edges = [
    (u, v, w)
    for u, v, w in undirected_edges
    if u in largest_component_set and v in largest_component_set
]

source_directed = max(
    nodes,
    key=lambda n: in_degree[n] + out_degree[n]
)

source_undirected = max(
    nodes,
    key=lambda n: len(undirected_adj[n])
)

def bfs(adj, source):
    visited = set([source])
    queue = deque([source])
    order = []
    distance = {source: 0}

    while queue:
        u = queue.popleft()
        order.append(u)

        for v, _ in adj[u]:
            if v not in visited:
                visited.add(v)
                distance[v] = distance[u] + 1
                queue.append(v)

    return {
        "visited_count": len(order),
        "max_level": max(distance.values()) if distance else 0
    }

def dfs(adj, source):
    visited = set()
    stack = [source]
    order = []

    while stack:
        u = stack.pop()

        if u in visited:
            continue

        visited.add(u)
        order.append(u)

        for v, _ in reversed(adj[u]):
            if v not in visited:
                stack.append(v)

    return {
        "visited_count": len(order)
    }

def is_weakly_connected_for_directed(nodes, edges, in_degree, out_degree):
    nonzero_nodes = [
        n for n in nodes
        if in_degree[n] + out_degree[n] > 0
    ]

    if not nonzero_nodes:
        return True

    weak_adj = {n: [] for n in nonzero_nodes}
    nonzero_set = set(nonzero_nodes)

    for u, v, _ in edges:
        if u in nonzero_set and v in nonzero_set:
            weak_adj[u].append((v, 1))
            weak_adj[v].append((u, 1))

    start = nonzero_nodes[0]
    visited = set([start])
    queue = deque([start])

    while queue:
        u = queue.popleft()

        for v, _ in weak_adj[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)

    return len(visited) == len(nonzero_nodes)

def eulerian_directed(nodes, edges, in_degree, out_degree):
    connected = is_weakly_connected_for_directed(
        nodes,
        edges,
        in_degree,
        out_degree
    )

    if not connected:
        return {
            "type": "Não euleriano",
            "reason": "Os vértices com grau positivo não pertencem a uma única componente fracamente conexa."
        }

    start_candidates = 0
    end_candidates = 0

    for n in nodes:
        diff = out_degree[n] - in_degree[n]

        if diff == 1:
            start_candidates += 1
        elif diff == -1:
            end_candidates += 1
        elif diff != 0:
            return {
                "type": "Não euleriano",
                "reason": "Há vértices com diferença entre grau de saída e entrada incompatível."
            }

    if start_candidates == 0 and end_candidates == 0:
        return {
            "type": "Circuito euleriano",
            "reason": "Todos os vértices possuem grau de entrada igual ao grau de saída."
        }

    if start_candidates == 1 and end_candidates == 1:
        return {
            "type": "Caminho euleriano",
            "reason": "Existe um vértice inicial e um vértice final compatíveis."
        }

    return {
        "type": "Não euleriano",
        "reason": "Quantidade de vértices iniciais/finais incompatível."
    }

def eulerian_undirected(adj):
    nonzero_nodes = [
        n for n in adj
        if len(adj[n]) > 0
    ]

    if not nonzero_nodes:
        return {
            "type": "Circuito euleriano",
            "reason": "Grafo sem arestas."
        }

    start = nonzero_nodes[0]
    visited = set([start])
    queue = deque([start])

    while queue:
        u = queue.popleft()

        for v, _ in adj[u]:
            if v not in visited:
                visited.add(v)
                queue.append(v)

    if len(visited) != len(nonzero_nodes):
        return {
            "type": "Não euleriano",
            "reason": "O grafo não é conexo considerando os vértices com grau positivo."
        }

    odd_vertices = [
        n for n in adj
        if len(adj[n]) % 2 == 1
    ]

    if len(odd_vertices) == 0:
        return {
            "type": "Circuito euleriano",
            "reason": "Todos os vértices possuem grau par."
        }

    if len(odd_vertices) == 2:
        return {
            "type": "Caminho euleriano",
            "reason": "Exatamente dois vértices possuem grau ímpar."
        }

    return {
        "type": "Não euleriano",
        "reason": f"Existem {len(odd_vertices)} vértices de grau ímpar."
    }

def dijkstra(adj, source):
    dist = {node: math.inf for node in adj}
    dist[source] = 0

    heap = [(0, source)]

    while heap:
        current_dist, u = heapq.heappop(heap)

        if current_dist > dist[u]:
            continue

        for v, w in adj[u]:
            new_dist = current_dist + w

            if new_dist < dist[v]:
                dist[v] = new_dist
                heapq.heappush(heap, (new_dist, v))

    reachable = [d for d in dist.values() if d < math.inf]

    return {
        "reachable_count": len(reachable),
        "max_distance": max(reachable) if reachable else None
    }

def bellman_ford(nodes, edges, source):
    dist = {node: math.inf for node in nodes}
    dist[source] = 0

    for _ in range(len(nodes) - 1):
        changed = False

        for u, v, w in edges:
            if dist[u] != math.inf and dist[u] + w < dist[v]:
                dist[v] = dist[u] + w
                changed = True

        if not changed:
            break

    has_negative_cycle = False

    for u, v, w in edges:
        if dist[u] != math.inf and dist[u] + w < dist[v]:
            has_negative_cycle = True
            break

    reachable = [d for d in dist.values() if d < math.inf]

    return {
        "reachable_count": len(reachable),
        "max_distance": max(reachable) if reachable else None,
        "has_negative_cycle": has_negative_cycle
    }

def floyd_warshall(nodes_subset, edges_subset):
    nodes_list = sorted(nodes_subset)
    index = {node: i for i, node in enumerate(nodes_list)}
    n = len(nodes_list)

    dist = [[math.inf] * n for _ in range(n)]

    for i in range(n):
        dist[i][i] = 0

    for u, v, w in edges_subset:
        i = index[u]
        j = index[v]

        if w < dist[i][j]:
            dist[i][j] = w
            dist[j][i] = w

    for k in range(n):
        for i in range(n):
            if dist[i][k] == math.inf:
                continue

            for j in range(n):
                new_dist = dist[i][k] + dist[k][j]

                if new_dist < dist[i][j]:
                    dist[i][j] = new_dist

    finite_distances = []

    for i in range(n):
        for j in range(n):
            if i != j and dist[i][j] < math.inf:
                finite_distances.append(dist[i][j])

    return {
        "nodes_count": n,
        "finite_pairs": len(finite_distances),
        "max_distance": max(finite_distances) if finite_distances else None,
        "average_distance": sum(finite_distances) / len(finite_distances) if finite_distances else None
    }

def tarjan_scc(adj):
    index_counter = [0]
    stack = []
    on_stack = set()
    indices = {}
    lowlink = {}
    sccs = []

    def strongconnect(v):
        indices[v] = index_counter[0]
        lowlink[v] = index_counter[0]
        index_counter[0] += 1

        stack.append(v)
        on_stack.add(v)

        for w, _ in adj[v]:
            if w not in indices:
                strongconnect(w)
                lowlink[v] = min(lowlink[v], lowlink[w])
            elif w in on_stack:
                lowlink[v] = min(lowlink[v], indices[w])

        if lowlink[v] == indices[v]:
            component = []

            while True:
                w = stack.pop()
                on_stack.remove(w)
                component.append(w)

                if w == v:
                    break

            sccs.append(component)

    for v in adj:
        if v not in indices:
            strongconnect(v)

    largest = max(len(c) for c in sccs)

    return {
        "scc_count": len(sccs),
        "largest_scc_size": largest
    }

class DSU:
    def __init__(self, nodes):
        self.parent = {n: n for n in nodes}
        self.rank = {n: 0 for n in nodes}

    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])

        return self.parent[x]

    def union(self, a, b):
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False

        if self.rank[root_a] < self.rank[root_b]:
            self.parent[root_a] = root_b
        elif self.rank[root_a] > self.rank[root_b]:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1

        return True

def kruskal_mst(nodes_subset, edges_subset):
    dsu = DSU(nodes_subset)
    mst_weight = 0
    mst_edges = 0

    for u, v, w in sorted(edges_subset, key=lambda item: item[2]):
        if dsu.union(u, v):
            mst_weight += w
            mst_edges += 1

    return {
        "mst_edges": mst_edges,
        "mst_weight": mst_weight
    }

def prim_mst(adj, start):
    visited = set([start])
    heap = []

    for v, w in adj[start]:
        heapq.heappush(heap, (w, start, v))

    mst_weight = 0
    mst_edges = 0

    while heap and len(visited) < len(adj):
        w, u, v = heapq.heappop(heap)

        if v in visited:
            continue

        visited.add(v)
        mst_weight += w
        mst_edges += 1

        for next_node, next_weight in adj[v]:
            if next_node not in visited:
                heapq.heappush(heap, (next_weight, v, next_node))

    return {
        "mst_edges": mst_edges,
        "mst_weight": mst_weight,
        "visited_count": len(visited)
    }

T_CRITICAL_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045
}

def critical_value_95(n):
    if n >= 30:
        return 1.96

    df = n - 1
    return T_CRITICAL_95.get(df, 2.262)

def benchmark(fn, runs):
    times = []
    result = None

    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        end = time.perf_counter()

        times.append(end - start)

    mean_time = statistics.mean(times)
    std_time = statistics.stdev(times) if len(times) > 1 else 0
    critical = critical_value_95(len(times))
    margin_error = critical * std_time / math.sqrt(len(times))

    return result, {
        "runs": len(times),
        "mean_seconds": mean_time,
        "std_seconds": std_time,
        "ci95_lower": mean_time - margin_error,
        "ci95_upper": mean_time + margin_error
    }

def summarize_result(result):
    return "; ".join(
        f"{key}={value}"
        for key, value in result.items()
    )

RUNS_FAST = 30
RUNS_SLOW = 10

experiments = [
    {
        "algorithm": "BFS",
        "graph": "Grafo direcionado tratado",
        "complexity": "O(V + E)",
        "runs": RUNS_FAST,
        "fn": lambda: bfs(directed_adj, source_directed)
    },
    {
        "algorithm": "DFS",
        "graph": "Grafo direcionado tratado",
        "complexity": "O(V + E)",
        "runs": RUNS_FAST,
        "fn": lambda: dfs(directed_adj, source_directed)
    },
    {
        "algorithm": "Eulerianidade direcionada",
        "graph": "Grafo direcionado tratado",
        "complexity": "O(V + E)",
        "runs": RUNS_FAST,
        "fn": lambda: eulerian_directed(nodes, directed_edges, in_degree, out_degree)
    },
    {
        "algorithm": "Eulerianidade não direcionada",
        "graph": "Grafo não direcionado tratado",
        "complexity": "O(V + E)",
        "runs": RUNS_FAST,
        "fn": lambda: eulerian_undirected(undirected_adj)
    },
    {
        "algorithm": "Dijkstra",
        "graph": "Grafo direcionado ponderado",
        "complexity": "O((V + E) log V)",
        "runs": RUNS_FAST,
        "fn": lambda: dijkstra(directed_adj, source_directed)
    },
    {
        "algorithm": "Bellman-Ford",
        "graph": "Grafo direcionado ponderado",
        "complexity": "O(VE)",
        "runs": RUNS_FAST,
        "fn": lambda: bellman_ford(nodes, directed_edges, source_directed)
    },
    {
        "algorithm": "Floyd-Warshall",
        "graph": "Maior componente conexa não direcionada",
        "complexity": "O(V³)",
        "runs": RUNS_SLOW,
        "fn": lambda: floyd_warshall(largest_component, lcc_edges)
    },
    {
        "algorithm": "Tarjan",
        "graph": "Grafo direcionado tratado",
        "complexity": "O(V + E)",
        "runs": RUNS_FAST,
        "fn": lambda: tarjan_scc(directed_adj)
    },
    {
        "algorithm": "Kruskal",
        "graph": "Maior componente conexa não direcionada",
        "complexity": "O(E log E)",
        "runs": RUNS_FAST,
        "fn": lambda: kruskal_mst(largest_component, lcc_edges)
    },
    {
        "algorithm": "Prim",
        "graph": "Maior componente conexa não direcionada",
        "complexity": "O(E log V)",
        "runs": RUNS_FAST,
        "fn": lambda: prim_mst(lcc_adj, largest_component[0])
    }
]

print("\n=== Parte II - Algoritmos da Disciplina ===")
print(f"Vértice inicial no grafo direcionado: {source_directed}")
print(f"Vértice inicial no grafo não direcionado: {source_undirected}")
print(f"Número de vértices no grafo direcionado: {len(nodes)}")
print(f"Número de arestas direcionadas: {len(directed_edges)}")
print(f"Número de vértices na maior componente conexa: {len(largest_component)}")
print(f"Número de arestas na maior componente conexa: {len(lcc_edges)}")

results_rows = []
timing_rows = []

for exp in experiments:
    print(f"\nExecutando {exp['algorithm']}...")

    result, timing = benchmark(exp["fn"], exp["runs"])

    print("Resultado:", summarize_result(result))
    print(
        f"Tempo médio: {timing['mean_seconds']:.8f}s | "
        f"Desvio padrão: {timing['std_seconds']:.8f}s | "
        f"IC 95%: [{timing['ci95_lower']:.8f}, {timing['ci95_upper']:.8f}]"
    )

    results_rows.append({
        "Algoritmo": exp["algorithm"],
        "Computado": "Sim",
        "Grafo usado": exp["graph"],
        "Complexidade teórica": exp["complexity"],
        "Resultado resumido": summarize_result(result)
    })

    timing_rows.append({
        "Algoritmo": exp["algorithm"],
        "Execuções": timing["runs"],
        "Tempo médio (s)": timing["mean_seconds"],
        "Desvio padrão (s)": timing["std_seconds"],
        "IC 95% inferior (s)": timing["ci95_lower"],
        "IC 95% superior (s)": timing["ci95_upper"],
        "Grafo usado": exp["graph"],
        "Complexidade teórica": exp["complexity"]
    })

df_results = pd.DataFrame(results_rows)
df_timings = pd.DataFrame(timing_rows)

df_results.to_csv(RESULTS_DIR / "parte2_resultados_algoritmos.csv", index=False)
df_timings.to_csv(RESULTS_DIR / "parte2_tempos_algoritmos.csv", index=False)

print("\nArquivos gerados:")
print(RESULTS_DIR / "parte2_resultados_algoritmos.csv")
print(RESULTS_DIR / "parte2_tempos_algoritmos.csv")
