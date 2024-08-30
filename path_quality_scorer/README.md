### Desciption
Utilizing the OpenAI API to assess the quality of a path within a knowledge graph by assigning a score between 0 and 1.


#### Required input:
- Each row represents a path between nodes.
- All paths in the dataset must have the same number of hops.
    - e.g A -> edge -> B -> edge -> C represents a path with 2 hops
- number of columns in the file should be $2*hop+1$


### How-to-run
- If you want to try your own dataset, put it in the `data/multihop/`
- I recommend using Batch Processing (second option)
- Evaluate dataset of paths directly (you may hit the rate limit)
    1. Run the dataset of paths directly: `python path_quality_scorer.py --input 3_hop_filt_50.csv --output evaluated_3_hop_filt_50_test.csv --model gpt-4o-mini --hop 3 --from_scratch True`
    2. It will create `data/multihop/scores_backup.csv` file, do not deleted it by accident. If the main function completed succefully the results will be merged with the input dataset and `scores_backup.csv` will be deleted.
- Evaluate the dataset of paths using Batch API (lower cost + higher rate limit) 
    - `./execute.sh -d 3_hop_filt_50.csv -i test -o test -m gpt-4o-mini -h 3` it will create `evaluated_filename` in the `data/multihop/` directory
        - -d is the dataset to be used for the quality scoring process
        - -i is the name of the folder where the batch input files will be stored
        - -o is the name of the folder where the batch output files will be stored
        - -m is the pre-processing model to be used
        - -h is the hop value for the quality scoring process 
   
    - Additionally, `path_quality_scorer.py` will display the amount of tokens that was used, and the cost. `execute.sh` will display amount of tokens and cost as well but only for input.


