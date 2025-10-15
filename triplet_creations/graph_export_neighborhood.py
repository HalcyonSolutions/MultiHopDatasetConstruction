import pandas as pd
import networkx as nx

# TODO: Add argparse for input parameters
# TODO: Add Summary
# TODO: Add type hints
# TODO: move function to utils

def export_subgraph_triplets(
    triplet_file, root_id, depth, output_file, invalid_ids=None, invalid_relations=None, max_triplets=None, min_triplets=None
):
    # Read triplets
    main_df = pd.read_csv(triplet_file, sep='\t', header=None, names=['head', 'rel', 'tail'])
    df = main_df.copy()

    # Remove invalid triplets
    if invalid_ids is not None:
        df = df[~df['head'].isin(invalid_ids) & ~df['tail'].isin(invalid_ids)]
    if invalid_relations is not None:
        df = df[~df['rel'].isin(invalid_relations)]

    # Build undirected graph for BFS
    G = nx.Graph()
    edge_map = {}  # (node1, node2) -> [list of (head, rel, tail)]
    for _, row in df.iterrows():
        h, t = row['head'], row['tail']
        if pd.notna(t) and t != '':
            G.add_edge(h, t)
            # Use tuple sorted to ensure undirected
            key = tuple(sorted([h, t]))
            edge_map.setdefault(key, []).append((row['head'], row['rel'], row['tail']))

    # BFS from root_id up to depth
    bfs_edges = set()
    for parent, child in nx.bfs_edges(G, root_id, depth_limit=depth):
        key = tuple(sorted([parent, child]))
        bfs_edges.add(key)


    # Optionally, add the self-loop for the root if you want (not typical)
    # bfs_edges.add((root_id, root_id))

    # Collect triplets for traversed edges only
    triplets_out = []
    for key in bfs_edges:
        for triplet in edge_map[key]:
            triplets_out.append(triplet)

    # Export to txt
    out_df = pd.DataFrame(triplets_out, columns=['head', 'rel', 'tail'])

    # add invalid relations if specified post BFS
    if invalid_relations is not None:
        invalid_relations = set(invalid_relations)
        nodes = set(out_df['head'].tolist()) | set(out_df['tail'].tolist())
        valid_df = (main_df['head'].isin(nodes) & main_df['rel'].isin(invalid_relations)) | (main_df['tail'].isin(nodes) & main_df['rel'].isin(invalid_relations))
        
        # apped main_df[valid_df] to out_df
        if not valid_df.empty:
            out_df = pd.concat([out_df, main_df[valid_df]], ignore_index=True)

    out_df.drop_duplicates(inplace=True)

    assert max_triplets is not None and len(out_df) < max_triplets, f"Error! The number of triplets exceeds the maximum limit. { len(out_df) } > { max_triplets }"
    assert min_triplets is not None and len(out_df) >= min_triplets, f"Error! The number of triplets is below the minimum limit. { len(out_df) } < { min_triplets }"
    print(f"Exporting {len(out_df)} triplets to {output_file}")
    out_df.to_csv(output_file, sep='\t', header=False, index=False)

if __name__ == "__main__":
    node_data = pd.read_csv('./data/metadata/node_data_family_bodon.csv')  # fixed typo
    num = 10

    # These should match the gender node IDs in your triplets file (length 40, check your actual IDs!)
    invalid_relations = ['gender']

    # select a random row as root
    root_id = node_data.sample(1)['ID'].values[0]

    print(f"Selected root ID: {root_id}")

    export_subgraph_triplets(
        triplet_file='./data/link_prediction/FamilyBodon/train.txt',
        root_id=root_id,
        depth=2,
        output_file=f'./data/link_prediction/FamilyBodon/sub{num}/triplets.txt',
        invalid_ids=None,
        invalid_relations=invalid_relations,
        max_triplets=5000,  # Optional: limit to 1000 triplets
        min_triplets=200  # Optional: ensure at least 100 triplets
    )
