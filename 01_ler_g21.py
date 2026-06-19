from pathlib import Path
import gzip
import networkx as nx
import matplotlib.pyplot as plt
import pandas as pd


BASE_DIR = Path("Cisco_22_networks/dir_g21_small_workload_with_gt")

EDGE_DIR = BASE_DIR / "dir_includes_packets_and_other_nodes"

if not EDGE_DIR.exists():
    print("Pasta não encontrada:", EDGE_DIR)
    print("Subpastas disponíveis em dir_g21_small_workload_with_gt:")
    for p in BASE_DIR.iterdir():
        if p.is_dir():
            print("-", p.name)
    raise SystemExit

def open_file(path):
    if path.suffix == ".gz":
        return gzip.open(path, "rt", encoding="utf-8", errors="ignore")
    return open(path, "r", encoding="utf-8", errors="ignore")

def parse_services(services_text):
    total_packets = 0
    services = []

    if not services_text:
        return 1, services

    for item in services_text.split(","):
        item = item.strip()

        if "-" not in item:
            continue

        port_proto, packets = item.split("-", 1)

        try:
            packets = int(packets)
        except ValueError:
            packets = 0

        services.append(port_proto)
        total_packets += packets

    if total_packets == 0:
        total_packets = 1

    return total_packets, services

def read_g21(edge_dir):
    G = nx.DiGraph()

    files = sorted(edge_dir.rglob("*.txt.gz")) + sorted(edge_dir.rglob("*.txt"))

    print(f"Arquivos encontrados: {len(files)}")

    for file_path in files:
        with open_file(file_path) as f:
            for line in f:
                line = line.strip()

                if not line or line.startswith("#"):
                    continue

                parts = line.split()

                if len(parts) < 3:
                    continue

                graph_id = parts[0]

                if graph_id != "g21":
                    continue

                source = int(parts[1])
                target = int(parts[2])
                services_text = parts[3] if len(parts) >= 4 else ""

                packets, services = parse_services(services_text)

                if G.has_edge(source, target):
                    G[source][target]["weight"] += packets
                    G[source][target]["observations"] += 1
                    G[source][target]["services"].update(services)
                else:
                    G.add_edge(
                        source,
                        target,
                        weight=packets,
                        observations=1,
                        services=set(services)
                    )

    return G

def to_weighted_undirected(G):
    UG = nx.Graph()

    for u, v, data in G.edges(data=True):
        weight = data.get("weight", 1)

        if UG.has_edge(u, v):
            UG[u][v]["weight"] += weight
        else:
            UG.add_edge(u, v, weight=weight)

    return UG

G = read_g21(EDGE_DIR)

print("\n=== Grafo direcionado G21 ===")
print("Número de vértices:", G.number_of_nodes())
print("Número de arestas direcionadas:", G.number_of_edges())
print("Número de auto-loops:", nx.number_of_selfloops(G))

G_sem_loops = G.copy()
G_sem_loops.remove_edges_from(nx.selfloop_edges(G_sem_loops))

print("\n=== Grafo direcionado sem auto-loops ===")
print("Número de vértices:", G_sem_loops.number_of_nodes())
print("Número de arestas:", G_sem_loops.number_of_edges())

UG = to_weighted_undirected(G_sem_loops)

print("\n=== Grafo não direcionado ===")
print("Número de vértices:", UG.number_of_nodes())
print("Número de arestas não direcionadas:", UG.number_of_edges())
print("Número de componentes conexas:", nx.number_connected_components(UG))

largest_component_nodes = max(nx.connected_components(UG), key=len)
LCC = UG.subgraph(largest_component_nodes).copy()

print("\n=== Maior componente conexa ===")
print("Número de vértices:", LCC.number_of_nodes())
print("Número de arestas:", LCC.number_of_edges())

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

edges_data = []

for u, v, data in G_sem_loops.edges(data=True):
    edges_data.append({
        "source": u,
        "target": v,
        "weight_packets": data.get("weight", 1),
        "observations": data.get("observations", 1),
        "num_services": len(data.get("services", []))
    })

df_edges = pd.DataFrame(edges_data)
df_edges.to_csv(output_dir / "g21_edges_directed_processed.csv", index=False)

print("\nArquivo salvo em:")
print(output_dir / "g21_edges_directed_processed.csv")

plt.figure(figsize=(14, 10))

pos = nx.spring_layout(LCC, seed=42)

node_sizes = [
    20 + 3 * LCC.degree(node)
    for node in LCC.nodes()
]

nx.draw_networkx_nodes(
    LCC,
    pos,
    node_size=node_sizes,
    alpha=0.8
)

nx.draw_networkx_edges(
    LCC,
    pos,
    alpha=0.15,
    width=0.5
)

plt.title("Visualização do G21 - Maior Componente Conexa")
plt.axis("off")
plt.tight_layout()

output_img = results_dir / "g21_visualizacao.png"
plt.savefig(output_img, dpi=200)
plt.show()

print("\nImagem salva em:")
print(output_img)
