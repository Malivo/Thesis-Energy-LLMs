# Prompts

The prompt sets used in the experiments, in all eight languages.

## Layout

prompts/
<language>/
mscore.txt
easy2hard.txt
compost_dbpedia.txt

Languages: english, french, german, polish, portuguese_pt, russian, spanish, ukrainian.

Each file holds one prompt per line. The three files per language group prompts by the source that inspired them:

- `mscore.txt` general knowledge and reasoning tasks
- `easy2hard.txt` progressively harder mathematical and reasoning problems
- `compost_dbpedia.txt` structured knowledge and factual retrieval tasks

## Difficulty tiers

Within each file the prompts are ordered by difficulty and split into three equal tiers by line range: easy (lines 1 to 30), medium (lines 31 to 60), and hard (lines 61 to 90). The benchmarking system tags each prompt with its language and difficulty tier from this ordering.

## Provenance

These prompts are our own. They were written for this work, drawing on the style and task types of the MScore, Easy2Hard, and COMPOST/DBpedia benchmarks, but they are not copies of those datasets. The English prompts are the originals. All other languages were produced by translating the English prompts with ChatGPT, aiming for semantic equivalence across languages.

Because the translations are machine generated and were not independently verified by native speakers, small differences in phrasing or naturalness between languages are possible. This is noted as a limitation in the dissertation.
