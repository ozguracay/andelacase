from typing import Tuple

import pandas as pd


class DatasetGenerator:
    def __init__(self, aplication_record_path: str, credit_record_path: str) -> None:
        self.application_record_path = aplication_record_path
        self.credit_record_path = credit_record_path

        self.raw_application_data = self.read_data(self.application_record_path)
        self.raw_credit_record_data = self.read_data(self.credit_record_path)

        self.label_data = self.create_label_data()
        self.dataset = self.create_dataset()

    @staticmethod
    def read_data(data_path: str) -> pd.DataFrame:
        data = pd.read_csv(data_path)
        data.columns = data.columns.str.lower()
        return data

    @staticmethod
    def status_encoder(status: str) -> int:
        if status in ("1", "2", "3", "4", "5"):
            return 1
        else:
            return 0

    def create_label_data(self) -> pd.DataFrame:
        self.raw_credit_record_data["label"] = self.raw_credit_record_data.apply(
            lambda x: self.status_encoder(x["status"]), axis=1
        )
        return self.raw_credit_record_data.groupby("id")["label"].max().reset_index()

    def create_dataset(self) -> pd.DataFrame:
        dataset = self.raw_application_data.merge(self.label_data, how="inner", on="id")
        return dataset.drop("id", axis=1)

    def get_x_and_y(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        feature_columns = [i for i in self.dataset.columns if i != "label"]
        x = self.dataset[feature_columns]
        y = self.dataset[["label"]]
        return (x, y)
