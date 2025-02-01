# SQUAB: **SQ**l-**U**nanswerable-and-**A**mbiguous-**B**enchmarking

This repository is the official implementation
of [Evaluating LLM’s robustness to Ambiguous and Unanswerable Questions
in Semantic Parsing]()

# 🔥 Updates

# 🏴󠁶󠁵󠁭󠁡󠁰󠁿 Overview
* ***What is SQUAB?*** **SQ**l-**U**nanswerable-and-**A**mbiguous-**B**enchmarking (SQUAB) is a benchmarking tool for generating Unanswerable and Ambiguous Text2SQL questions.
* ***How does it work?*** Given a proprietary database as input, it employs a blend of script and LLMs during the generation.
* ***More specifically?*** The tool is based on a three interface architecture, built to maximize number of test while minimizing cost.
* ***Where is processed the data?*** The data is processed by external provider like OpenAi. However, it is possible to change one line of code to include AzureOpenAi for privacy constraints.

## Prompts for reproducibility:
All the prompts used in the main paper are in the following module
```shell
|--squab
    | -- models  # contains wrapper for using different LLM logics and their prompts
        | -- prompts.py  # contains all the prompts used for the generations of the tests
```
The following table, associate the used prompt with the key of the dictionary in the prompts.py module


| **Usage**                        | **Prompts Key**                  | **Prompt Description**                                         |
|----------------------------------|----------------------------------|----------------------------------------------------------------|
| Inference                        | ambrosia-text2sql                | Prompt used to test LLMs over _Ambiguous_ questions            |
| Inference                        | ambrosia-text-2-sql-unanswerable | Prompt used to test LLMs over _Unanswerable_ questions         |
| Generation Ambiguous Question    | question_variability             | Base prompt for Test Generation for _Ambiguous_ questions      |
| Generation Unanswerable Question | sql-to-text                      | Base prompt for Test Generation for _Unanswerable_ questions   |
| Column Ambiguity                 | ambiguity-col_generator          | Used to generate the hypernym in the Column Ambiguity Category |
| Scope                            | scope_pattern_semantic           | Used to find and classify the entity-component relationship    |
| Column Unanswerable              | unanswerable-column_generation   | Generate a new Column aligned with the intent of the table     |
| Calculation Unanswerable         | unanswerable-udf_generation      | Generate a new UDF executable in SQL                           |
| Out-Of-Scope                     | unanswerable-udf_generation_oos  | Generate a new UDF not executable in SQL                       |


To find a prompt, simply look for the prompts key in the module [prompts.py](squab/models/prompts.py).



## Project

```shell
|--squab
    |-- generate_datasets  # 
         | -- dataset_generator.py  # abstract dataset generator to implement for a new test category
         | -- generators  # contains ambiguous and unanswerable tests generator
             | -- ambiguity_generators  # contains ambiguous tests generator
                 | -- attachment_generator.py  # logic for building attachment ambiguity tests
                 | -- column_ambiguity_generator.py  # logic for building column ambiguity tests
                 | -- scope_generator.py  # logic for building scope ambiguity tests
             | -- unanswerable_generators  # contains unanswerable tests generator
                 | -- calculation_unanswerable.py  # logic for building calculation unanswerable tests
                 | -- column_unanswerable.py  # logic for building column unanswerable tests
                 | -- out_of_scope.py  # logic for building out of scope unanswerable tests
    |-- evaluate_datasets  
         | -- evaluate.py  # logic to calculate precision-recall-accuracy for generated tests 
    | -- models  # contains wrapper for using different LLM logics and their prompts
        | -- prompts.py  # contains all the prompts used for the generations of the tests
```








## Citation

# ⚡️ Reproducibility

## Installation

After cloning the repository, the environment can be created with [uv](https://docs.astral.sh/uv/)

```console
uv install
```

## Generation of the dataset

SQUAB facilitates the automatic creation of ambiguous and unanswerable test cases using any database as input.
To replicate the experiments described in the paper, the benchmarks [AMBROSIA](https://ambrosia-benchmark.github.io/) and [BEAVER](https://github.com/peterbaile/beaver/tree/main) were utilized.
These benchmarks assess both SQUAB’s dataset generation capabilities and model performance in various settings, including enterprise environments.

Once downloaded, insert the database in the _data_ folder 
```shell
| -- squab
| -- data
    | -- ambrosia
    | -- beaver
```
The generation of the tests is based on a blend of script and openai call. Remember to define `OPENAI_API_KEY` in the **.env** file

- To generate the datasets for AMBROSIA:

```shell
python ./main_generate_dataset.py --test_category_to_generate attachment --dataset_path data/ambrosia/ambrosia.csv
```

- To generate the datasets for BEAVER:

```shell
python ./main_generate_dataset.py --test_category_to_generate attachment --dataset_path data/beaver
```

To create different test categories, change `test_category_to_generate` accordingly.  

