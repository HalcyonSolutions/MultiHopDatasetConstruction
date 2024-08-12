# Introduction

Preparation for submission of multihop traversal

# Datasets
- Using Freebase and WikiData triplets, the **KG** **fbwiki** was created. You can find a **KG** for **Neo4j** in `datasets/free-wiki.dump`, and follow https://github.com/Nurassyl-lab/neo4j_graph_traversal.git repository to load the `.dump` file into a **Neo4j**.
- **free-wiki** on itself is a more refined version of the triplets from Freebase and WikiData. We cleaned the triplets, reduced the number of edges, and made graph directed.
- Using the **KG** we have derived datasets of *paths*, you can find those in the `datasets/#_hops.csv`
