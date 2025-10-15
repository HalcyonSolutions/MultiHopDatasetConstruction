import argparse
import os

import numpy as np
from tqdm import tqdm
from collections import defaultdict, Counter

import warnings

from utils.basic import load_triplets, load_pandas, extract_literals

def parse_args() -> argparse.Namespace:
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Provide the Random Walk statistics utilities.")
    
    parser.add_argument('--triplet-path', type=str, default='./data/link_prediction/MetaQA/',
                        help='')
    parser.add_argument('--question-path', type=str, default='./data/qa/MetaQA/metaqa_nhop.csv',
                        help='')
    
    parser.add_argument('--num-rollout-steps', type=int, default=3,
                        help='Number of random-walk steps to simulate.')

    parser.add_argument('--test-only', action='store_true',
                        help='If set, only evaluate test questions.')
    parser.add_argument('--use-self-loops', action='store_true',
                        help='If set, add self-loops to the adjacency matrix.')
    parser.add_argument('--use-full-graph', action='store_true',
                        help='If set, use the full graph (train+dev+test triplets) for random-walks.')
    parser.add_argument('--use-ideal-paths', action='store_true',
                        help='If set, compute ideal path probabilities.')

    return parser.parse_args()

# -------------------- Markov-chain utilities --------------------
try:
    import scipy.sparse as sp
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

def _build_transition_matrices(entity_neighbors_set,
                               entity_neighbors_list,
                               num_entities):
    """
    Build two row-stochastic transition matrices:
      - T_uniform: uniform over unique neighbors
      - T_weighted: weighted by multiplicity (number of parallel edges)
    Undirected handling is upstream (you already add both directions).
    Isolated nodes become absorbing (self-loop = 1).
    Returns CSR sparse if scipy is available, else dense numpy arrays.
    """
    if _HAS_SCIPY:
        rows_u, cols_u, data_u = [], [], []
        rows_w, cols_w, data_w = [], [], []

        for u in range(num_entities):
            uniq = entity_neighbors_set.get(u, set())
            lst  = entity_neighbors_list.get(u, [])
            if len(uniq) == 0:
                rows_u.append(u); cols_u.append(u); data_u.append(1.0)
                rows_w.append(u); cols_w.append(u); data_w.append(1.0)
                continue

            # Uniform
            pu = 1.0 / float(len(uniq))
            for v in uniq:
                rows_u.append(u); cols_u.append(int(v)); data_u.append(pu)

            # Weighted by multiplicity
            counts = Counter(lst)
            total  = float(len(lst))
            for v, c in counts.items():
                rows_w.append(u); cols_w.append(int(v)); data_w.append(c / total)

        Tu = sp.coo_matrix((data_u, (rows_u, cols_u)), shape=(num_entities, num_entities)).tocsr()
        Tw = sp.coo_matrix((data_w, (rows_w, cols_w)), shape=(num_entities, num_entities)).tocsr()
        return Tu, Tw
    else:
        Tu = np.zeros((num_entities, num_entities), dtype=np.float64)
        Tw = np.zeros((num_entities, num_entities), dtype=np.float64)
        for u in range(num_entities):
            uniq = entity_neighbors_set.get(u, set())
            lst  = entity_neighbors_list.get(u, [])
            if len(uniq) == 0:
                Tu[u, u] = 1.0
                Tw[u, u] = 1.0
                continue

            pu = 1.0 / float(len(uniq))
            for v in uniq:
                Tu[u, int(v)] = pu

            counts = Counter(lst)
            total  = float(len(lst))
            for v, c in counts.items():
                Tw[u, int(v)] = c / total
        return Tu, Tw

# -------------------- PATCHED utils (robust to sparse outputs) --------------------
def n_step_distribution(start_node, n, T, upto=False):
    """
    Row-vector evolution p_{t+1} = p_t @ T.

    Returns a 1-D np.ndarray always, even when T is scipy.sparse.
    - upto=False: p_n (exactly n steps)
    - upto=True : sum_{k=0..n} p_k (occupancy up to n; not first-hit)
    """
    N = T.shape[0]
    p = np.zeros(N, dtype=np.float64)
    p[start_node] = 1.0

    if upto:
        acc = p.copy()

    for _ in range(n):
        if _HAS_SCIPY and sp.issparse(T):
            # compute (p @ T) via column form (T.T @ p) to avoid csr_matrix outputs
            p = np.asarray(T.transpose().dot(p)).ravel()
        else:
            p = np.asarray(p @ T).ravel()

        if upto:
            acc += p

    return acc if upto else p


def reach_prob(start_node, target_nodes, n, T, upto=False):
    """
    Probability of being at 'target_nodes' after exactly n steps (upto=False),
    or occupancy ≤ n (upto=True). Always returns floats.
    """
    is_single = isinstance(target_nodes, (int, np.integer))
    targets = [target_nodes] if is_single else list(target_nodes)

    dist = n_step_distribution(start_node, n, T, upto=upto)  # guaranteed 1-D np.ndarray
    out = {int(t): float(dist[int(t)]) for t in targets}
    return out[targets[0]] if is_single else out

def _pct(x: float) -> str:
    return f"{x:.4e}"

def _stats(n_eval: int, arr: np.ndarray):
    if n_eval == 0:
        return ("n/a", "n/a", "n/a")
    arr = np.asarray(arr, dtype=float)
    mean = _pct(np.mean(arr))
    median = _pct(np.median(arr))
    std = _pct(np.std(arr))
    return (mean, median, std)

import sys

if __name__ == "__main__":

    args = parse_args()

    if args.use_full_graph:
        # Load full graph (train+dev+test) if available, else load train, dev, test separately
        full_triplet_path = os.path.join(args.triplet_path, 'triplets.txt')
        if os.path.exists(full_triplet_path):
            triplet_df = load_triplets(full_triplet_path, sep='\t')
        else:
            triplet_df = load_triplets([
                os.path.join(args.triplet_path, 'train.txt'),
                os.path.join(args.triplet_path, 'valid.txt'),
                os.path.join(args.triplet_path, 'test.txt'),
            ], sep='\t')
    else:
        # Load only training triplets
        triplet_df = load_triplets(os.path.join(args.triplet_path, 'train.txt'), sep='\t')
        
        if args.use_ideal_paths:
            args.use_ideal_paths = False
            warnings.warn("Ideal paths require the full graph; disabling ideal path computation.")

    qa_df = load_pandas(args.question_path, sep=',')
    
    # Check if "Paths" column exists for ideal path computation
    if args.use_ideal_paths and 'Paths' not in qa_df.columns:
        args.use_ideal_paths = False
        warnings.warn("No 'Paths' column found in questions; disabling ideal path computation.")

    # Check if multiple answers exist and convert it to list if needed
    if all(isinstance(x, list) or (isinstance(x, str) and x.startswith('[') and x.endswith(']')) for x in qa_df['Answer-Entity']):
        multi_answer = True
        qa_df['Answer-Entity'] = extract_literals(qa_df['Answer-Entity'])
    else:
        multi_answer = False

    nodes = set(triplet_df['head']).union(set(triplet_df['tail']))
    rels = set(triplet_df['relation'])

    # convert to integer ids
    ent2id = {n: i for i, n in enumerate(sorted(nodes))}
    id2ent = {i: n for n, i in ent2id.items()}
    rel2id = {r: i for i, r in enumerate(sorted(rels))}
    id2rel = {i: r for r, i in rel2id.items()}

    # Filter questions based on SplitLabel if not test_only
    train_df = qa_df[qa_df['SplitLabel'] == 'train']
    dev_df = qa_df[qa_df['SplitLabel'] == 'dev']
    test_df = qa_df[qa_df['SplitLabel'] == 'test']

    triplet_df['head'] = triplet_df['head'].map(ent2id)
    triplet_df['tail'] = triplet_df['tail'].map(ent2id)
    triplet_df['relation'] = triplet_df['relation'].map(rel2id)

    all_true_triples = set(zip(triplet_df['head'], triplet_df['relation'], triplet_df['tail']))
    nodes = set(ent2id.values())
    rels = set(rel2id.values())

    # Build neighborhood structures
    entity_neighbors_set = defaultdict(set)
    entity_neighbors_list = defaultdict(list)
    entity_neighbor_triples_set = defaultdict(set)
    for h, r, t in all_true_triples:
        entity_neighbors_set[h].add(t)
        entity_neighbors_set[t].add(h)

        entity_neighbors_list[h].append(t)
        entity_neighbors_list[t].append(h)

        entity_neighbor_triples_set[h].add( (r, t) )
        entity_neighbor_triples_set[t].add( (r, h) )

    if args.use_self_loops:
        # include self-loops
        for n in nodes:
            entity_neighbors_set[n].add(n)
            entity_neighbors_list[n].append(n)
            entity_neighbor_triples_set[n].add( (None, n) )

    # Calculate uniform neighbor probabilities
    neighbor_uniform_prob = {k: 1.0 / len(v) for k, v in entity_neighbors_set.items() if len(v) > 0}

    # Calculate weighted neighbor probabilities
    neighbor_weighted_prob = defaultdict(dict)
    for k, neighbors in entity_neighbors_list.items():
        if len(neighbors) == 0:
            continue
        neighborhood = np.array(neighbors)
        unique, counts = np.unique(neighborhood, return_counts=True)
        total_count = np.sum(counts)
        for u, c in zip(unique, counts):
            neighbor_weighted_prob[int(k)][int(u)] = float(c / total_count)

    # Calculate uniform triple probabilities
    neighbor_triple_uniform_prob = {k: 1.0 / len(v) for k, v in entity_neighbor_triples_set.items()}

    # -------- build Markov transition matrices --------
    num_entities = len(nodes)
    T_uniform, T_weighted = _build_transition_matrices(
        entity_neighbors_set=entity_neighbors_set,
        entity_neighbors_list=entity_neighbors_list,
        num_entities=num_entities
    )

    # exact-n random-walk arrival probabilities accumulators
    rw_exact_uniform = []            # P(X_n = answer | uniform)
    rw_exact_weighted = []           # P(X_n = answer | weighted)

    uniform_path_probs = []         # consider the unique neighboring entities (only care if arrived at desired node, disregarding number of links in-between)
    weighted_path_probs = []        # consider the neighboring entities dependant on the number of rels (only care if arrive at desired node, considering number of links in-between)
    ideal_path_probs = []           # consider each triplet separately (for ideal path, caring if it took desired rel and node)
    skipped_paths = 0

    splits = [[train_df, 'train'], [dev_df, 'dev'], [test_df, 'test']] if not args.test_only else [[test_df, 'test']]

    # Evaluate path probabilities for each dataset split
    for eval_df, name in splits:
        for idx in tqdm(range(len(eval_df)), desc={"train": "Training", "dev": "Validation", "test": "Testing"}[name], leave=False):
            # evaluate one question at a time (to avoid large memory usage)
            mini_batch = eval_df[idx : idx + 1]

            questions = mini_batch["Question"].tolist()[0]
            hops = int(mini_batch["Hops"].tolist()[0])
            source_node = mini_batch["Source-Entity"].tolist()[0]
            answer_node = mini_batch["Answer-Entity"].tolist()[0]
            source_node = ent2id.get(source_node, None)
            if multi_answer:
                answer_node = [ent2id.get(ans, None) for ans in answer_node]
            else:
                answer_node = ent2id.get(answer_node, None)

            assert args.num_rollout_steps >= hops, f"Path length {hops} exceeds num_rollout_steps {args.num_rollout_steps}"

            # -------- Random Walk: Only exact-n (last step must be the answer) --------
            if multi_answer:
                # sum up probabilities for multiple answers
                p_exact_uniform  = sum(reach_prob(source_node, ans, args.num_rollout_steps, T_uniform,  upto=False) for ans in answer_node)
                p_exact_weighted = sum(reach_prob(source_node, ans, args.num_rollout_steps, T_weighted, upto=False) for ans in answer_node)
            else:
                p_exact_uniform  = reach_prob(source_node, answer_node, args.num_rollout_steps, T_uniform,  upto=False)
                p_exact_weighted = reach_prob(source_node, answer_node, args.num_rollout_steps, T_weighted, upto=False)
            
            rw_exact_uniform.append(p_exact_uniform)
            rw_exact_weighted.append(p_exact_weighted)

            if not args.use_ideal_paths: continue

            path = mini_batch["Paths"].tolist()[0]

            #--------- Specific Path: Must take specific path to reach the answer ------
            path_prob_uniform = 1.0
            path_prob_weighted = 1.0
            path_prob_uniform_triple = 1.0
            skip_sample = False
            for triple in path:
                head = ent2id.get(triple[0], None)
                rel = rel2id.get(triple[1], None)
                tail = ent2id.get(triple[2], None)

                if (head, rel, tail) not in all_true_triples:
                    skipped_paths += 1
                    skip_sample = True
                    break

                # Multiply previous prob with current edge prob
                path_prob_uniform *= neighbor_uniform_prob.get(head, 0.0)
                path_prob_weighted *= neighbor_weighted_prob.get(head, {}).get(tail, 0.0)
                path_prob_uniform_triple *= neighbor_triple_uniform_prob.get(head, 0.0)

            # If the path length is smaller than the answer node, add the probability of staying in the answer node
            if len(path) < args.num_rollout_steps:
                len_diff = args.num_rollout_steps - len(path)
                path_prob_uniform *= neighbor_uniform_prob.get(answer_node, 0.0)*len_diff
                path_prob_weighted *= neighbor_weighted_prob.get(answer_node, {}).get(answer_node, 0.0)*len_diff
                path_prob_uniform_triple *= neighbor_triple_uniform_prob.get(answer_node, 0.0)*len_diff

            if skip_sample: continue # If there were any invalid paths, skip aggregation

            uniform_path_probs.append(path_prob_uniform)
            weighted_path_probs.append(path_prob_weighted)
            ideal_path_probs.append(path_prob_uniform_triple)

    # -------------------- Reporting --------------------
    # Keep the true number of evaluated samples BEFORE any sentinel fallback
    n_eval = len(rw_exact_uniform)

    # Safe conversion if lists are empty (keeps your current behavior)
    rw_exact_uniform     = np.array(rw_exact_uniform)     if n_eval > 0 else np.array([0.0])
    rw_exact_weighted    = np.array(rw_exact_weighted)    if n_eval > 0 else np.array([0.0])

    rows = [
        ("Arrival@n (weighted random walk)",                *_stats(n_eval, rw_exact_weighted)),
        ("Arrival@n (uniform random walk)",                 *_stats(n_eval, rw_exact_uniform)),
    ]

    if args.use_ideal_paths:
        uniform_path_probs   = np.array(uniform_path_probs)   if n_eval > 0 else np.array([0.0])
        weighted_path_probs  = np.array(weighted_path_probs)  if n_eval > 0 else np.array([0.0])
        ideal_path_probs     = np.array(ideal_path_probs)     if n_eval > 0 else np.array([0.0])
        rows.extend([
            ("Specific Path (weighted neighbors)",           *_stats(n_eval, weighted_path_probs)),
            ("Specific Path (uniform neighbors)",            *_stats(n_eval, uniform_path_probs)),
            ("Specific Path (ideal path, uniform triples)",  *_stats(n_eval, ideal_path_probs)),
        ])

    print("\n" + "=" * 86)
    print(f"Random-walk path statistics (exact n-step = last hop reaches answer)")
    print(f"Evaluated: {n_eval}   |   Skipped: {skipped_paths}")
    print("-" * 86)
    print(f"{'Metric':<46} {'mean':>12} {'median':>12} {'std':>12}")
    print("-" * 86)
    for name, mean, median, std in rows:
        print(f"{name:<46} {mean:>12} {median:>12} {std:>12}")
    print("-" * 86)