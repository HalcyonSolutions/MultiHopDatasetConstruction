@echo off
REM Batch script to run fbwiki_triplte_process_alt.py with different datasets

REM Fb15k to Wiki Equivalent
python fbwiki_triplet_process_alt.py --primary-triplet-path ./data/triplet_creation_fb_wiki_15k.txt --entity-forwarding-path ./data/forwarding_creation_fb_wiki_15k.txt --filtered-triplet-output ./data/triplet_filt_fb_wiki_15k.txt --candidate-nodes-output ./data/nodes_fb15k_wiki_15k.txt --candidate-relationships-output ./data/relationship_fb_wiki_15k.txt
echo.

REM MQuaKE
python fbwiki_triplet_process_alt.py --primary-triplet-path ./data/triplet_creation_mquake.txt --entity-forwarding-path ./data/forwarding_creation_mquake.txt --filtered-triplet-output ./data/triplet_filt_mquake.txt --candidate-nodes-output ./data/nodes_filt_mquake.txt --candidate-relationships-output ./data/relationship_filt_mquake.txt
echo.

REM FB-Wiki v4.2
python fbwiki_triplet_process_alt.py --primary-triplet-path ./data/triplet_creation_fb_wiki_v4.txt --entity-forwarding-path ./data/forwarding_creation_fb_wiki_v4.txt --filtered-triplet-output ./data/triplet_filt_fb_wiki_v4.txt --candidate-nodes-output ./data/nodes_filt_fb_wiki_v4.txt --candidate-relationships-output ./data/relationship_filt_fb_wiki_v4.txt
echo.