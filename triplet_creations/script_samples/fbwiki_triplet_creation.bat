@echo off
REM Batch script to run fbwiki_triplet_creation.py with different datasets

REM FB15k to Wiki Equivalent
echo Running FB15k to Wiki Equivalent...
python fbwiki_triplet_creation.py ^
    --entity-list-path ./data/vocabs/nodes_fb15k.txt ^
    --triplet-output-path ./data/temp/triplet_creation_fb15k_wiki.txt ^
    --forwarding-output-path ./data/temp/forwarding_creation_fb15k_wiki.txt
echo.

REM Fb-Wiki-V4.2
echo Running Fb-Wiki-V4.2...
python fbwiki_triplet_creation.py ^
    --entity-list-path ./data/vocabs/nodes_fb_wiki.txt ^
    --triplet-output-path ./data/temp/triplet_creation_fb_wiki_v4.txt ^
    --forwarding-output-path ./data/temp/forwarding_creation_fb_wiki_v4.txt
echo.

echo All tasks completed!
pause