### Description:
to run:

`python relation_classification.py --input dataset.csv --row 4000` -- pruning will start from row 4000 and onwards

Every successful row `pruning` will save the data into `dataset.csv`.

`gpt-4o-mini` sometimes can't prune a row, so I give it 3 attemps, after which I skip it!


### Add openai token
you can create a `.env` file locally and paste `OPENAI_API_KEY=abc123` key there. Follow this tutorial https://platform.openai.com/docs/quickstart
