import random
from pathlib import Path

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.io import mmread
from scipy.sparse import csr_matrix


MTX_PATH = "soc-douban.mtx"
OUT_DIR = Path("outputs_douban")
OUT_DIR.mkdir(exist_ok=True)

RANDOM_SEED = 42
NULL_REALISATIONS = 30          
PATH_SAMPLE_NODES = 1500       
PLOT_MAX_K = 5000              


def set_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)

def load_mtx(mtx_path: str) -> nx.Graph:
    A = mmread(mtx_path)
    if not hasattr(A, "tocsr"):
        A = csr_matrix(A)
    else:
        A = A.tocsr()

    A.setdiag(0)
    A.eliminate_zeros()
    G = nx.from_scipy_sparse_array(A)  # Graph by default
    return G

def degree_distribution(degrees: np.ndarray):
    max_k = int(degrees.max()) if degrees.size else 0
    counts = np.bincount(degrees, minlength=max_k + 1)
    Pk = counts / counts.sum()
    k_values = np.arange(len(Pk))
    return k_values, Pk

def ccdf_from_pk(k_values: np.ndarray, Pk: np.ndarray):
    ccdf = np.flip(np.cumsum(np.flip(Pk)))
    return k_values, ccdf

def average_shortest_path(G: nx.Graph, sample_nodes: int, seed: int) -> float:

    if G.number_of_nodes() == 0:
        return float("nan")

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc).copy()
    else:
        H = G

    n = H.number_of_nodes()
    if n <= 1:
        return 0.0

    rng = random.Random(seed)
    nodes = list(H.nodes())
    s = min(sample_nodes, n)
    sampled = rng.sample(nodes, s)

    dists = []
    for src in sampled:
        lengths = nx.single_source_shortest_path_length(H, src)
        if len(lengths) > 1:
            dists.extend([d for v, d in lengths.items() if d > 0])

    return float(np.mean(dists)) if dists else float("nan")


def diameter_upper_bound(G: nx.Graph, seed: int) -> int:
    if G.number_of_nodes() == 0:
        return 0

    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        H = G.subgraph(largest_cc).copy()
    else:
        H = G

    rng = random.Random(seed)
    start = rng.choice(list(H.nodes()))

    dist1 = nx.single_source_shortest_path_length(H, start)
    u = max(dist1, key=dist1.get)
    dist2 = nx.single_source_shortest_path_length(H, u)
    v = max(dist2, key=dist2.get)
    return dist2[v]


def compute_knnk(G: nx.Graph):
    degrees = dict(G.degree())
    # k_nn(i)
    knn_i = {}
    for i in G.nodes():
        ki = degrees[i]
        if ki == 0:
            knn_i[i] = 0.0
            continue
        knn_i[i] = sum(degrees[j] for j in G.neighbors(i)) / ki

    # Group by k
    buckets = {}
    for i, ki in degrees.items():
        buckets.setdefault(ki, []).append(knn_i[i])

    k_vals = np.array(sorted(buckets.keys()))
    knn_vals = np.array([np.mean(buckets[k]) for k in k_vals])
    return k_vals, knn_vals


def configuration_model_graph(deg_seq, seed: int) -> nx.Graph:
    MG = nx.configuration_model(deg_seq, seed=seed)
    G = nx.Graph(MG)  # Merge parallel edges
    G.remove_edges_from(nx.selfloop_edges(G))
    return G

def compute_metrics(G: nx.Graph, path_sample_nodes: int, seed: int):
    avg_clust = float(nx.average_clustering(G))
    assort = float(nx.degree_assortativity_coefficient(G))
    asp = float(average_shortest_path(G, sample_nodes=path_sample_nodes, seed=seed))
    diam = int(diameter_upper_bound(G, seed=seed))
    return {
        "avg_clustering": avg_clust,
        "assortativity_r": assort,
        "avg_shortest_path_est": asp,
        "diameter_ub": diam,
    }

def main():
    set_seeds(RANDOM_SEED)

    G = load_mtx(MTX_PATH)
    print("Basic Information")
    print("Nodes:", G.number_of_nodes())
    print("Edges:", G.number_of_edges())
    print("Density:", nx.density(G))
    print("Is directed:", G.is_directed())
    print("Is connected:", nx.is_connected(G))
    print("Self-loops:", nx.number_of_selfloops(G))
    print("Graph type:", type(G))


    emp = compute_metrics(G, path_sample_nodes=PATH_SAMPLE_NODES, seed=RANDOM_SEED)
    degrees = np.array([d for _, d in G.degree()], dtype=int)
    k_vals, Pk = degree_distribution(degrees)
    k_ccdf, ccdf = ccdf_from_pk(k_vals, Pk)

    # Plot P(k)
    plt.figure()
    mask = (k_vals > 0) & (k_vals <= PLOT_MAX_K) & (Pk > 0)
    plt.loglog(k_vals[mask], Pk[mask], marker="o", linestyle="None")
    plt.xlabel("Degree k")
    plt.ylabel("P(k)")
    plt.title("Degree distribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "degree_distribution_pk.png", dpi=200)
    plt.close()

    # Plot CCDF
    plt.figure()
    mask = (k_ccdf > 0) & (k_ccdf <= PLOT_MAX_K) & (ccdf > 0)
    plt.loglog(k_ccdf[mask], ccdf[mask], marker="o", linestyle="None")
    plt.xlabel("Degree k")
    plt.ylabel("P(K ≥ k)")
    plt.title("Degree CCDF")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "degree_ccdf.png", dpi=200)
    plt.close()

    # ---- k_nn(k) plot
    k_knn, knn = compute_knnk(G)
    plt.figure()
    mask = (k_knn > 0) & (k_knn <= PLOT_MAX_K) & (knn > 0)
    plt.loglog(k_knn[mask], knn[mask], marker="o", linestyle="None")
    plt.xlabel("Degree k")
    plt.ylabel(r"$k_{nn}(k)$")
    plt.title(r"Average nearest-neighbour degree $k_{nn}(k)$")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "knn_k.png", dpi=200)
    plt.close()

    # ---- Configuration Model
    deg_seq = degrees.tolist()
    null_vals = {
        "avg_clustering": [],
        "assortativity_r": [],
        "avg_shortest_path_est": [],
        "diameter_ub": [],
    }

    for t in range(NULL_REALISATIONS):
        seed_t = RANDOM_SEED + 1000 + t
        Gn = configuration_model_graph(deg_seq, seed=seed_t)
        m = compute_metrics(Gn, path_sample_nodes=PATH_SAMPLE_NODES, seed=seed_t)
        for key in null_vals:
            null_vals[key].append(m[key])


    # Compare empirical vs null
    def plot_emp_vs_null(metric: str, ylabel: str, filename: str):
        arr = np.array(null_vals[metric], dtype=float)
        mu = float(np.mean(arr))
        sigma = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

        plt.figure()
        plt.errorbar([0], [mu], yerr=[sigma], fmt="o")
        plt.scatter([0], [emp[metric]], marker="x")
        plt.xticks([0], ["Null (mean±std) vs Emp (x)"])
        plt.ylabel(ylabel)
        plt.title(f"{metric}: empirical vs null")
        plt.tight_layout()
        plt.savefig(OUT_DIR / filename, dpi=200)
        plt.close()

    plot_emp_vs_null("avg_clustering", "Average clustering ⟨C⟩", "compare_avg_clustering.png")
    plot_emp_vs_null("assortativity_r", "Assortativity r", "compare_assortativity.png")
    plot_emp_vs_null("avg_shortest_path_est", "Estimated ⟨l⟩", "compare_path_length.png")
    plot_emp_vs_null("diameter_ub", "Diameter", "compare_diameter.png")



if __name__ == "__main__":
    main()