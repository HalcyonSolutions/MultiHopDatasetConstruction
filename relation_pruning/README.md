### How-to Run
to run in `terminal`:

`python relation_classification.py --input datasets/Head3K.csv --method 2 --row 0` -- pruning will from row 0 and onwards 

- `use python pruning.py --help` to learn more about arguments
- `--method 1` is written for `Head23K.csv` and `--method 2` is for `Head3K.csv`

Every successful "pruning" of the row will save the data into `Head3K.csv`.

`gpt-4o-mini` sometimes can't prune a row, so I give it `th = 3` attemps, after which I skip it. Skipped rows are written in the `failed_heads.txt`.

- If you're done with main pruning, run `python pruning.py --input Head3K.csv --method 2 --th 5 --run_failed True`, though I think right now this is unnecessary since code is improved.

to run in `VSCode`:
- Follow use `.vscode/launch.json`, and same logic as if you'd run it from the terminal.

### Add OpenAI Token
you can create a `.env` file locally and paste `OPENAI_API_KEY=abc123` key there. Follow this tutorial https://platform.openai.com/docs/quickstart
