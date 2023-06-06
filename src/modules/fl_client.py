import flwr as fl
import json
import pandas as pd
import numpy as np

from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedKFold, train_test_split
from sktime.classification.interval_based import TimeSeriesForestClassifier
from src.modules.typology import Typology
from src.modules.utils import compute_causality, smote_rebalance, IS_BASELINE_SCENARIO, DATASET_NAME


class FLClient(fl.client.NumPyClient):
    def __init__(self, node):
        self._parent = node
        self.matrix = None
        self.model = TimeSeriesForestClassifier(n_jobs=-1)

    def get_parameters(self, config):
        params = [
            json.dumps(self._parent.name),
            json.dumps(self._parent.nominal_attributes),
            json.dumps(self._parent.X.tolist()),
            json.dumps(self._parent.y.tolist()),
            json.dumps(self._parent.columns.tolist()) if len(self._parent.name) >= 2 and self._parent.name != 'CN' else json.dumps(
                self._parent.columns),
        ]
        return params

    def fit(self, parameters, config):
        params = self.get_parameters(config={})
        return params, len(self._parent.X), {}

    def evaluate(self, parameters, config):
        if IS_BASELINE_SCENARIO:
            ate = 0
            df_typo = pd.DataFrame(
                self._parent.X,
                columns=self._parent.columns[:-1]
            )
            df_typo["label"] = self._parent.y
            typology = Typology(
                path="./temp/",
                dataframe=df_typo,
                temp_file=f"temp_{self._parent.name}"
            )
            safe, borderline, rare, outlier = typology.typo_percentage
            num_safe, num_borderline, num_rare, num_outlier = typology.typo_nums
        else:
            df_smote = pd.DataFrame(
                self._parent.X,
                columns=self._parent.columns[:-1]
            )
            df_smote["label"] = self._parent.y
            self._parent.X, self._parent.y = smote_rebalance(df_smote, return_df=False)
            df_typo = pd.DataFrame(
                self._parent.X,
                columns=self._parent.columns[:-1]
            )
            df_typo["label"] = self._parent.y
            typology = Typology(
                path="./temp/",
                dataframe=df_typo,
                temp_file=f"temp_{self._parent.name}"
            )
            safe, borderline, rare, outlier = typology.typo_percentage
            num_safe, num_borderline, num_rare, num_outlier = typology.typo_nums
            ate = compute_causality(
                df_typo,
                dag_file_path=f"./results/causality/dag/model_dag_{DATASET_NAME}.pkl",
                dot_file_path=f"./temp/graph_{self._parent.name}.dot"
            )
            self._parent.ma7_ate.append(ate)
            if len(self._parent.ma7_ate) > 7:
                self._parent.ma7_ate.pop(0)
            new_df_typo = df_typo.sample(frac=0.9)
            ate = compute_causality(
                new_df_typo,
                dag_file_path=f"./results/causality/dag/model_dag_{DATASET_NAME}.pkl",
                dot_file_path=f"./temp/graph_{self._parent.name}.dot"
            )
            if ate > np.mean(self._parent.ma7_ate):
                self._parent.X = new_df_typo.drop(columns=["label"]).to_numpy()
                self._parent.y = new_df_typo["label"].to_numpy()
        X_train, X_test, y_train, y_test = train_test_split(self._parent.X, self._parent.y, test_size=0.2)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        cr = classification_report(y_pred, y_test, output_dict=True)
        accuracy = cr['accuracy']
        macro_precision = cr['macro avg']['precision']
        macro_recall = cr['macro avg']['recall']
        macro_f1 = cr['macro avg']['f1-score']
        weighted_precision = cr['weighted avg']['precision']
        weighted_recall = cr['weighted avg']['recall']
        weighted_f1 = cr['weighted avg']['f1-score']
        self._parent.result = {
            "node": self._parent.name,
            "instances": len(self._parent.y),
            "min_instances": int(num_safe) + int(num_borderline) + int(num_rare) + int(num_outlier),
            "S": safe,
            "B": borderline,
            "R": rare,
            "O": outlier,
            "num_S": num_safe,
            "num_B": num_borderline,
            "num_R": num_rare,
            "num_O": num_outlier,
            "ate": ate,
            "ma7_ate": np.array(self._parent.ma7_ate).mean() if len(self._parent.ma7_ate) > 0 else 0,
            "accuracy": accuracy,
            "macro_precision": macro_precision,
            "macro_recall": macro_recall,
            "macro_f1": macro_f1,
            "weighted_precision": weighted_precision,
            "weighted_recall": weighted_recall,
            "weighted_f1": weighted_f1
        }
        self._parent.send_result(self._parent.result)
        return float(0), len(self._parent.X), self._parent.result
