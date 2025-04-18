@echo off
REM Batch script to run fbwiki_triplte_cration.py with different datasets

REM FB15k to Wiki Equivalent
python fbwiki_triplet_creation.py --entity-list-path './data/nodes_fb15k.txt' --triplet-output-path ./data/triplet_creation_fb_wiki_15k.txt --qualifier-output-path ./data/qualifier_creation_fb_wiki_15k.txt --forwarding-output-path ./data/forwarding_creation_fb_wiki_15k.txt
echo.

REM MQuake
python fbwiki_triplet_creation.py --entity-list-path './data/nodes_mquake.txt' --triplet-output-path ./data/triplet_creation_mquake.txt --qualifier-output-path ./data/qualifier_creation_mquake.txt --forwarding-output-path ./data/forwarding_creation_mquake.txt
echo.

REM Fb-Wiki-V4.2
python fbwiki_triplet_creation.py --entity-list-path './data/nodes_fb_wiki_v3.txt' --triplet-output-path ./data/triplet_creation_fb_wiki_v4.txt --qualifier-output-path ./data/qualifier_creation_fb_wiki_v4.txt --forwarding-output-path ./data/forwarding_creation_fb_wiki_v4.txt
echo.