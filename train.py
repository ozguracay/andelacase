import json

from src.dataset import DatasetGenerator
from src.ml_pipeline import MLPipeline

APPLICATION_PATH = "storage/data/application_record.csv"
CREDIT_PATH = "storage/data/credit_record.csv"

HPO_SPACE_PATH = "storage/config/hpo_space.json"

with open(HPO_SPACE_PATH, "r") as f:
    hpo_space_params = json.load(f)


dg = DatasetGenerator(
    aplication_record_path=APPLICATION_PATH, credit_record_path=CREDIT_PATH
)

x, y = dg.get_x_and_y()


mlpipeline = MLPipeline(x=x, y=y)
mlpipeline.make_grid_search_cv(pipeline_params=hpo_space_params)
mlpipeline.save_pipeline()
mlpipeline.store_model_performance()
