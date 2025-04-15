from zenml.config import DockerSettings
from zenml.integrations.constants import MLFLOW
from zenml.pipelines import pipeline
from steps.config import ModelNameConfig
from steps.clean_data import clean_data
from steps.ingest_data import ingest_data
from steps.model_train import train_model
from steps.evaluation import evaluation

docker_settings = DockerSettings(required_integrations=[MLFLOW])

@pipeline(enable_cache=False, settings={"docker": docker_settings})
def train_pipeline():
    """
    Args:
        ingest_data: Step
        clean_data: Step
        train_model: Step
        evaluation: Step
    Returns:
        mse: float
        rmse: float
    """
    df = ingest_data()
    x_train, x_test, y_train, y_test = clean_data(df)
  
    model = train_model(
    x_train, x_test, y_train, y_test, 
    config=ModelNameConfig(model_name="lightgbm", fine_tuning=False)  # or False
    )
    mse, rmse = evaluation(model, x_test, y_test)