import argparse
import logging

import pandas as pd
import sqlalchemy.exc
from dotenv import load_dotenv
from tqdm import tqdm

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

load_dotenv(override=True)

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
        elif ambig_type in ['column_unanswerable', 'calculation_unanswerable', 'out_of_scope']:
            return read_db_tbl_amrbosia_unans(db_path)
        else:
            raise ValueError(f'test_category_to_generate must be in {list(GENERATORS.keys())}')

    elif 'beaver' in db_path.lower():
        return read_db_tbl_beaver(db_path)
    else:
        raise ValueError("the db_path must contain either 'ambrosia' or 'beaver'")


def generate(dataset_path, test_category_to_generate,
             max_patterns_for_tbl,
             max_num_metadata_for_pattern,
             max_questions_for_metadata):
    db_paths2list_tbl_names = read_db_tbl(dataset_path, test_category_to_generate)
    dfs = []
    generator = GENERATORS[test_category_to_generate]()
    for db_path, tbls in tqdm(db_paths2list_tbl_names):
        fun_input = DatasetInput(
            relative_sqlite_db_path=db_path,
            tbl_in_db_to_analyze=list(tbls),
            max_patterns_for_tbl=max_patterns_for_tbl,
            max_num_metadata_for_pattern=max_num_metadata_for_pattern,
            max_questions_for_metadata=max_questions_for_metadata,
        )
        try:
            df = generator.generate_dataset(fun_input)
        except sqlalchemy.exc.NoSuchTableError as e:
            # this error arises with AMBROSIA databases
            logging.warning(f'{db_path}\n{e}')
            continue
        dfs.append(df)

    return pd.concat(dfs, ignore_index=True) if len(dfs) > 0 else pd.DataFrame()


def main():
    args = parse_args()
    df = generate(args.dataset_path,
                  args.test_category_to_generate,
                  args.max_patterns_for_tbl,
                  args.max_num_metadata_for_pattern,
                  args.max_questions_for_metadata)
    dataset = 'beaver' if 'beaver' in args.dataset_path.lower() else 'ambrosia'
    df.to_json(f'generated_dataset_{dataset}_{args.test_category_to_generate}.json', orient='records', indent=2)


def parse_args():
    parser = argparse.ArgumentParser(description="Generate dataset and save as JSON file")
    parser.add_argument('--test_category_to_generate',
                        type=str,
                        help=f'the test_category to generate: one in {list(GENERATORS.keys())}')

    parser.add_argument('--dataset_path',
                        type=str,
                        help=f'the dataset path where to fetch the databases. For Ambrosia `data/ambrosia/ambrosia.csv`,'
                             f'for BEAVER `data/beaver`')

    parser.add_argument('--max_patterns_for_tbl',
                        type=int,
                        default=1,
                        help='the maximum number of patterns to generate for each table')

    parser.add_argument('--max_num_metadata_for_pattern',
                        type=int,
                        default=1,
                        help='the maximum number of metadata to generate for each pattern')
    parser.add_argument('--max_questions_for_metadata',
                        type=int,
                        default=1,
                        help='the maximum number of questions to generate for each metadata')

    return parser.parse_args()


if __name__ == '__main__':
    main()
