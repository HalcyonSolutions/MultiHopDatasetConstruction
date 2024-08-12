### How to Run
To run the script in the **terminal**: `python relation_classification.py --input datasets/Head3K.csv --method 2 --row 0`

This will start the pruning process from row 0 onwards.
- Use `python pruning.py --help` to learn more about the available arguments.
- `--method 1` is designed for `Head23K.csv`, while `--method 2` is for `Head3K.csv`.

Each successful row pruning will save the updated data back into `Head3K.csv`.

> Note: The model **gpt-4o-mini** sometimes fails to prune a row. In such cases, it will attempt pruning up to 3 times if `th = 3`. If it still fails, the row will be skipped and recorded in `failed_heads.txt`.

If you have completed the main pruning process, you can run:
`python pruning.py --input Head3K.csv --method 2 --th 5 --run_failed True`

However, this step is likely unnecessary now due to recent code improvements.

To run the script in **Visual Studio Code**:
- Use the `DaRealMultiHop/.vscode/launch.json` configuration, following the same logic as running it from the terminal.

### Add OpenAI Token
To add your OpenAI API token, create a `.env` file in your project directory and paste the following line: `OPENAI_API_KEY=abc123`, where `abc123` is your token.

For more details, follow the OpenAI quickstart guide: https://platform.openai.com/docs/quickstart
