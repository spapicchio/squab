import argparse

import pandas as pd

from squab import DatasetInput
from squab.generate_datasets.generators.ambiguity_generators import (
    AttachmentGenerator,
    ScopeGenerator,
    ColumnAmbiguityGenerator
)
from squab.generate_datasets.generators.unanswerable_generators import ColumnUnanswerableGenerator, \
    CalculationUnanswerableGenerator, \
    OutOfScopeGenerator
from utils import read_db_tbl_ambrosia_ambig, read_db_tbl_beaver, read_db_tbl_amrbosia_unans

GENERATORS = {
    'attachment': AttachmentGenerator,
    'scope': ScopeGenerator,
    'column_ambiguity': ColumnAmbiguityGenerator,
    'column_unanswerable': ColumnUnanswerableGenerator,
    'calculation_unanswerable': CalculationUnanswerableGenerator,
    'out_of_scope': OutOfScopeGenerator
}


def read_db_tbl(db_path, ambig_type):
    if 'ambrosia' in db_path.lower():
        if ambig_type in ['attachment', 'scope', 'column_ambiguity']:
            ambig_type = ambig_type if ambig_type != 'column_ambiguity' else 'vague'
            return read_db_tbl_ambrosia_ambig(db_path, ambig_type)
        else:
            return read_db_tbl_amrbosia_unans(db_path)

    elif 'beaver' in db_path.lower():
        return read_db_tbl_beaver(db_path)
    else:
        raise ValueError("the db_path must contain either 'ambrosia' or 'beaver'")


def generate(dataset_path, test_category_to_generate, generator_name):
    db_paths2list_tbl_names = read_db_tbl(dataset_path, test_category_to_generate)
    dfs = []
    generator = GENERATORS[generator_name]()
    for db_path, tbls in db_paths2list_tbl_names:
        fun_input = DatasetInput(
            relative_sqlite_db_path=db_path,
            tbl_in_db_to_analyze=tbls,
            max_patterns_for_tbl=10,
            max_num_metadata_for_pattern=2,
            max_questions_for_metadata=7,
        )
        df = generator.generate_dataset(fun_input=fun_input)
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)


def main():
    args = parse_args()
    generate(args.dataset_path,
             args.test_type,
             args.test_category_to_generate)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset and save as JSON file")
    parser.add_argument('--test_category_to_generate',
                        type=str,
                        help=f'the test_category to generate: one in {list(GENERATORS.keys())}')

    parser.add_argument('--dataset_path',
                        type=str,
                        help=f'the dataset path where to fetch the databases. For Ambrosia `data/ambrosia/ambrosia.csv`,'
                             f'for BEAVER `data/beaver`')

    return parser.parse_args()
