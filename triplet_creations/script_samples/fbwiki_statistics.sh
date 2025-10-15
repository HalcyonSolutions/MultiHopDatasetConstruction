# Shell script to run fbwiki_statistics.py with different datasets

# FJ-Wiki
echo "Running FJ-Wiki..."
python fbwiki_statistics.py \
    --entity-list-path ./data/vocabs/nodes_fj_wiki.txt \
    --entity-data-path ./data/metadata/node_data_fj_wiki.csv \
    --relationship-data-path ./data/metadata/relation_data_wiki.csv \
    --triplets-data-path ./data/link_prediction/FJ-Wiki/triplets.txt
echo

# Subgraph
echo "Running Subgraph..."
python fbwiki_statistics.py \
    --entity-list-path ./data/vocabs/nodes_cherrypicked.txt \
    --entity-data-path ./data/metadata/node_data_cherrypicked.csv \
    --relationship-data-path ./data/metadata/relation_data_wiki.csv \
    --triplets-data-path ./data/triplets_subgraph.txt
echo

# FB-Wiki-V4
echo "Running FB-Wiki-V4..."
python fbwiki_statistics.py \
    --entity-data-path ./data/metadata/node_data_fb_wiki.csv \
    --relationship-data-path ./data/metadata/relation_data_wiki.csv \
    --triplets-data-path ./data/link_prediction/Fb-Wiki/triplets.txt
echo

# FB-Wiki-V2
echo "Running FB-Wiki-V2..."
python fbwiki_statistics.py \
    --relationship-data-path ./data/metadata/relation_data_wiki.csv \
    --triplets-data-path ./data/link_prediction/Fb-Wiki-V2/triplets.txt
echo

# Fb15k-237
echo "Running Fb15k-237..."
python fbwiki_statistics.py \
    --triplets-data-path ./data/link_prediction/FB15k-237/triplets.txt
echo

# WN18RR
echo "Running WN18RR..."
python fbwiki_statistics.py \
    --triplets-data-path ./data/link_prediction/WN18RR/triplets.txt
echo

# FB15k-Wiki
echo "Running FB15k-Wiki..."
python fbwiki_statistics.py \
    --triplets-data-path ./data/temp/triplet_filt_fb_wiki_15k.txt
echo

# MQuaKE
echo "Running MQuaKE..."
python fbwiki_statistics.py \
    --triplets-data-path ./data/mquake/triplets.txt
echo

# FB-Wiki v4.2
echo "Running FB-Wiki v4.2..."
python fbwiki_statistics.py \
    --triplets-data-path ./data/temp/triplet_filt_fb_wiki_v4.txt
echo

# KinshipHinton
echo "Running KinshipHinton..."
python fbwiki_statistics.py \
    --triplets-data-path ./data/link_prediction/KinshipHinton/triplets.txt
echo

# MetaQA
echo "Running MetaQA..."
python fbwiki_statistics.py \
    --triplets-data-path ./data/link_prediction/MetaQA/triplets.txt
echo

echo "All tasks completed!"