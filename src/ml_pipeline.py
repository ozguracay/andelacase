import pickle

import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import make_pipeline
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

EDUCATION_DICT = {
    "Lower secondary": 1,
    "Secondary / secondary special": 2,
    "Incomplete higher": 3,
    "Higher education": 4,
    "Academic degree": 5,
}
# NOTE remove gender code for model fairness

SELECTED_COLUMNS = [
    # "code_gender",
    "flag_own_car",
    "flag_own_realty",
    "cnt_children",
    "amt_income_total",
    "name_income_type",
    "name_education_type",
    "name_family_status",
    "name_housing_type",
    "days_birth",
    "days_employed",
    "flag_mobil",
    "flag_work_phone",
    "flag_phone",
    "flag_email",
    "occupation_type",
    "cnt_fam_members",
    "education",
]

SELECTED_COLUMNS.remove("flag_mobil")
SELECTED_COLUMNS.remove("name_education_type")


class PrepProcesor(BaseEstimator, TransformerMixin):
    """
    This processor used for feature engineering tasks

    - drop feature
    - fillna
    - feature encoding
    - feature grouping
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):

        X.fillna({"occupation_type": "unknown"}, inplace=True)
        X = X.assign(
            education=lambda x: EDUCATION_DICT.get(x.name_education_type.str, 0)
        )

        return X[SELECTED_COLUMNS]


class MLPipeline:
    def __init__(
        self, x: pd.DataFrame, y: pd.DataFrame, pipleline_hpo=None, pipeline_path=None
    ) -> None:
        self.x = x
        self.y = y
        self.pipeline_hpo = pipleline_hpo

        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(
            self.x, self.y, test_size=0.2
        )

        self.cat_columns = [
            i
            for i in self.x.select_dtypes(include=["object"]).columns.tolist()
            if i in SELECTED_COLUMNS
        ]
        self.num_columns = [
            i
            for i in self.x.select_dtypes(exclude=["object"]).columns.tolist()
            if i in SELECTED_COLUMNS
        ]

        self.pipeline = self.create_pipeline()
        self.set_hp()

    def create_pipeline(self):
        pre_processor = PrepProcesor()
        numeric_pipeline = Pipeline([("Scaler", StandardScaler())])
        categorical_pipeline = Pipeline(
            [("OneHot", OneHotEncoder(handle_unknown="ignore"))]
        )
        transformer = ColumnTransformer(
            [
                ("num", numeric_pipeline, self.num_columns),
                ("cat", categorical_pipeline, self.cat_columns),
            ]
        )
        enn = EditedNearestNeighbours()
        smote = SMOTE(random_state=0)
        xgb = XGBClassifier(scale_pos_weight=2)
        pipeline = make_pipeline(pre_processor, transformer, enn, smote, xgb)
        return pipeline

    def make_grid_search_cv(self, pipeline_params: dict):
        gs = GridSearchCV(self.pipeline, pipeline_params)
        gs.fit(self.x_train, self.y_train)
        self.pipeline = gs

    def fit(self):
        self.pipeline.fit(self.x_train, self.y_train)

    def set_hp(self):
        if self.pipeline_hpo:
            self.pipeline.set_params(self.pipeline_hpo)

    def save_pipeline(self, file_name="storage/models/last_pipeline.pkl"):
        with open(file_name, "wb") as f:
            pickle.dump(self.pipeline, f)

    def store_model_performance(
        self, file_name="storage/artifacts/model_performance.md"
    ):
        y_pred = self.pipeline.predict(self.x_test)
        y_train_pred = self.pipeline.predict(self.x_train)
        with open(file_name, "w") as f:
            print("# test set performance", file=f)
            print("## confusin matrix", file=f)
            print(confusion_matrix(self.y_test, y_pred), file=f)

            print("## classification report", file=f)
            print(classification_report(self.y_test, y_pred), file=f)

            print("# train set performance", file=f)
            print("## confusin matrix", file=f)
            print(confusion_matrix(self.y_train, y_train_pred), file=f)

            print("## classification report", file=f)
            print(classification_report(self.y_train, y_train_pred), file=f)
