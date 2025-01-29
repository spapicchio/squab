import logging

import sqlalchemy
from func_timeout import func_timeout, FunctionTimedOut
from qatch.connectors import SqliteConnector
from qatch.evaluate_dataset import OrchestratorEvaluator as QatchEvaluator
from typing_extensions import Literal


class BaseEvaluator:
    """Provides base evaluation functionality for ambiguous and unanswerable queries."""

    def __init__(self):
        """
        Initializes a BaseEvaluator instance.

        - Creates an internal QatchEvaluator configured to measure 'execution_accuracy'.
        - Initializes a connector placeholder and database path attribute.
        """

        self.qatch_evaluator = QatchEvaluator(evaluator_names=['execution_accuracy'])
        self._connector: SqliteConnector | None = None
        self.db_path = None

    @property
    def connector(self):
        """
        Returns a SqliteConnector instance based on the current db_path.

        If the internal connector is None or was previously bound to a different db_path,
        a new connector will be created. Otherwise, the existing connector is returned.
        """
        if self._connector is None or self._connector.db_path != self.db_path:
            return SqliteConnector(relative_db_path=self.db_path,
                                   db_name=self.db_path.split('/')[-1].replace('.sqlite', ''))
        return self._connector

    def evaluate(self,
                 target_sql: list[str],
                 predicted_sql: list[str],
                 test_type: str,
                 string_in_unans_prediction: str,
                 db_path: str) -> dict:
        """
        Main evaluation method for queries.

        Depending on the test_type, dispatches to different specialized methods.

        :param target_sql: List of target SQL queries.
        :param predicted_sql: List of predicted SQL queries.
        :param test_type: The type of test (e.g., 'unans' for unanswerable queries).
        :param string_in_unans_prediction: String indicating unanswerability in the predicted query.
        :param db_path: Path to the SQLite database file.
        :return: A dictionary containing evaluation metrics.
        """

        self.db_path = db_path
        if 'unans' in test_type.lower():
            return self.evaluate_unanswerable(predicted_sql[0],
                                              string_in_unans_prediction=string_in_unans_prediction)
        else:
            return self.evaluate_ambig_queries(target_sql, predicted_sql)

    def evaluate_ambig_queries(self,
                               target_queries: list[str],
                               predicted_queries: list[str],
                               ) -> dict[Literal['precision', 'recall', 'f1'], float]:
        """
        Evaluates ambiguous queries by comparing each target query to each predicted query.

        The method keeps track of whether target or predicted queries have matches
        based on whether the execution accuracy surpasses a threshold.

        :param target_queries: List of target SQL queries.
        :param predicted_queries: List of predicted SQL queries.
        :return: A dictionary with precision, recall, and f1 scores.
        """
        # Initialize match tracking
        predictions2match = {pred: False for pred in predicted_queries}
        target2match = {target: False for target in target_queries}

        # Compare each predicted query to each target query
        for prediction in predicted_queries:
            for target in target_queries:
                execution_accuracy = self.run_qatch_metrics(target, prediction, self.connector)
                if execution_accuracy > 0.5:
                    # Mark both queries as matched
                    predictions2match[prediction] = True
                    target2match[target] = True

        # Calculate precision, capping at 1
        precision = sum(predictions2match.values()) / len(predicted_queries) if len(predicted_queries) > 0 else 0
        precision = 1 if precision > 1 else precision
        # Calculate recall, capping at 1
        recall = sum(target2match.values()) / len(target_queries)
        recall = 1 if recall > 1 else recall
        # Calculate F1
        f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
        return {'precision': precision, 'recall': recall, 'f1': f1}

    def evaluate_unanswerable(self,
                              predicted_sql: str,
                              string_in_unans_prediction: str | None = 'is unanswerable',
                              ) -> dict:
        """
        Evaluates whether the predicted SQL query is recognized as unanswerable.

        :param predicted_sql: The predicted SQL query string for unanswerable checks.
        :param string_in_unans_prediction: The substring to look for in predicted_sql indicating unanswerable.
        :return: A dictionary with the 'accuracy' metric.
        """

        if string_in_unans_prediction.lower() in predicted_sql.lower():
            return {'accuracy': 1.0}
        return {'accuracy': 0.0}

    def run_qatch_metrics(self, target_sql: str, predicted_sql: str, connector: SqliteConnector) -> float:
        """
        Runs the Qatch metrics to measure execution accuracy for a single query pair.

        :param target_sql: A single target SQL query.
        :param predicted_sql: A single predicted SQL query.
        :param connector: A SqliteConnector instance for executing queries.
        :return: The execution accuracy as a float.
        """
        # Disallow queries with CRUD operations
        if any(
                op in predicted_sql.lower()
                for op in
                ['insert into ', 'update ', 'delete ', 'create table ', 'drop table ', 'alter table ']
        ):
            return 0.0

        try:
            evaluation = func_timeout(10,
                                      self.qatch_evaluator.evaluate_single_test,
                                      args=(target_sql, predicted_sql, connector))
            return evaluation['execution_accuracy']
        except FunctionTimedOut:
            logging.warning("Evaluation timed out")
            return 0.0
        except sqlalchemy.exc.ResourceClosedError:
            logging.warning(f"Resource closed error: {predicted_sql}")
            return 0.0
