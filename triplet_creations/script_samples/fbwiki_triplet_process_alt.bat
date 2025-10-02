@echo off
REM Batch script to run fbwiki_triplet_process_alt.py with different datasets

REM Fb15k to Wiki Equivalent
echo Running Fb15k to Wiki Equivalent...
python fbwiki_triplet_process_alt.py ^
    --primary-triplet-path ./data/temp/triplet_creation_fb15k_wiki.txt ^
    --entity-forwarding-path ./data/temp/forwarding_creation_fb15k_wiki.txt ^
    --filtered-triplet-output ./data/link_prediction/Fb15k-Wiki/triplets.txt ^
    --candidate-nodes-output ./data/vocabs/nodes_fb15k_wiki.txt ^
    --candidate-relationships-output ./data/vocabs/relationship_fb15k_wiki.txt
echo.

REM FB-Wiki v4.2
echo Running FB-Wiki v4.2...
python fbwiki_triplet_process_alt.py ^
    --primary-triplet-path ./data/temp/triplet_creation_fb_wiki_v4.txt ^
    --entity-forwarding-path ./data/temp/forwarding_creation_fb_wiki_v4.txt ^
    --filtered-triplet-output ./data/link_prediction/Fb-Wiki-V4/triplets.txt ^
    --candidate-nodes-output ./data/vocabs/nodes_fb_wiki_v4.txt ^
    --candidate-relationships-output ./data/vocabs/relationship_fb_wiki_v4.txt
echo.

echo All tasks completed!
pause