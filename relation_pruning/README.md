### Description:
to run:

`python relation_classification.py --input Head3K.csv --method 2 --row 0` -- pruning will from row 0 and onwards 

- `use python pruning --help` to learn more about arguments
- `--method 1` is written for `Head23K.csv` and `--method 2` is for `Head3K.csv`

Every successful row "pruning" will save the data into `Head3K.csv`.

`gpt-4o-mini` sometimes can't prune a row, so I give it `th = 3` attemps, after which I skip it. Skipped rows are written in the `failed_heads.txt`

- If you're done with main pruning, run `python --input Head3K.csv --method 2 --th 5 --run_failed True`, though I think right now this is unnecessary since code is improved

### Add openai token
you can create a `.env` file locally and paste `OPENAI_API_KEY=abc123` key there. Follow this tutorial https://platform.openai.com/docs/quickstart
