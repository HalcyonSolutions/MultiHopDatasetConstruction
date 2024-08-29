### Desciption
Utilizing the OpenAI API to assess the quality of a path within a knowledge graph by assigning a score between 0 and 1.


#### Required input:
- A `.csv` file.
- Each row represents a path between nodes.
- All paths in the dataset must have the same number of hops.
    - e.g A -> edge -> B -> edge -> C represents a path with 2 hops
- number of columns in the file should be $2*hop+1$


### How-to-run
- Evaluate dataset of paths directly (you may hit the rate limit)
    1. Run the dataset of paths directly: `python batch_pre_processing.py --input_dataset 2_hop_filt.csv  --model gpt-4o-mini --hop 2`
    2. It will create `scores_backup.csv` file, do not deleted it by accident. If the main function completed succefully the results will be merged with the input dataset and `scores_backup.csv` will be deleted.
- Evaluate the dataset of paths using Batch API (lower cost + higher rate limit) 
    1. Preprocess: `python batch_pre_processing.py --input_dataset 2_hop_filt.csv  --model gpt-4o-mini --hop 2`. The result of this command is a produced file(s) in `data/batch_input/`, since OpenAI Batch API has a token limit (2M), we have to split the batch files into multiple if that limit is exceeded. 
    2. Run the Batch API: `python path_quality_scorer_batch.py --input 2_hop_filt.csv --model gpt-4o-mini --hop 2 --monitor True --output_list_file my_batches.txt`. The result of this command is stored in `data/batch_output/`, the file name should be `batch_file_name + _results`. It also produces the `.txt` file that contains all processed batch file names. 
    3. Running `python batch_output_processing.py --dataset 2_hop_filt_10.csv --results my_batches.txt --model gpt-4o-mini` will add a new column to a multihop dataset with path evaluation scores.
    4. Note that using `--monitor` True will display the status of your batch every 60 seconds. You can stop monitoring at any time by pressing **CTRL+C** or setting `--monitor False`. The batch process itself runs on OpenAI's servers. Be sure to write down the batch id so you can check the status of your batch at any time. Keep in mind that the batch will be automatically canceled after 24 hours.

- Additionally, `path_quality_scorer.py` will display the amount of tokens that was used, and the cost. `batch_pre_processing.py` will display amount of tokens and cost as well but only for input.

If you want to try your own dataset, put it in the `data/multihop/`