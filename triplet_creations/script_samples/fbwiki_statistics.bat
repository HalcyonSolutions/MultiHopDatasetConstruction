@echo off
REM Batch script to run fbwiki_statistics.py with different datasets

REM FJ-Wiki
echo Running FJ-Wiki...
python fbwiki_statistics.py --entity-list-path ./data/nodes_fj_wiki.txt --entity-data-path ./data/node_data_fj_wiki.csv --relationship-data-path ./data/relation_data_wiki.csv --triplets-data-path ./data/triplets_fj_wiki.txt
echo.

REM Subgraph
echo Running Subgraph...
python fbwiki_statistics.py --entity-list-path ./data/nodes_cherrypicked.txt --entity-data-path ./data/node_data_cherrypicked.csv --relationship-data-path ./data/relation_data_wiki.csv --triplets-data-path ./data/triplets_subgraph.txt
echo.

REM FB-Wiki-V3
echo Running FB-Wiki-V3...
python fbwiki_statistics.py --entity-list-path ./data/nodes_fb_wiki.txt --entity-data-path ./data/node_data_fb_wiki.csv --relationship-data-path ./data/relation_data_wiki.csv --triplets-data-path ./data/triplets_fb_wiki_v3.txt
echo.

REM FB-Wiki-V4
echo Running FB-Wiki-V4...
python fbwiki_statistics.py --entity-data-path ./data/node_data_fb_wiki.csv --relationship-data-path ./data/relation_data_wiki.csv --triplets-data-path ./data/triplets_fb_wiki.txt
echo.

REM FB-Wiki-V2
echo Running FB-Wiki-V2...
python fbwiki_statistics.py --relationship-data-path ./data/relation_data_wiki.csv --triplets-data-path ./data/triplets_fb_wiki_v2.txt
echo.

REM Fb15k-237
echo Running Fb15k-237...
python fbwiki_statistics.py --triplets-data-path ./data/triplets_fb15k_237.txt
echo.

REM WN18RR
echo Running WN18RR...
python fbwiki_statistics.py --triplets-data-path ./data/triplets_wn18rr.txt
echo.

echo All tasks completed!
pause

REM FB15k-Wiki
echo Running FB15k-Wiki
python fbwiki_statistics.py --triplets-data-path ./data/triplet_filt_fb_wiki_15k.txt
echo.

REM MQuaKE
echo Running MQuaKE
python fbwiki_statistics.py --triplets-data-path ./data/triplet_filt_mquake.txt
echo.

REM FB-Wiki v4.2
echo Running FB15k-Wiki
python fbwiki_statistics.py --triplets-data-path ./data/triplet_filt_fb_wiki_v4.txt
echo.
