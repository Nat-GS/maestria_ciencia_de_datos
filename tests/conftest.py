import pytest
from src.integration import load_project_data
from src.modeling import run_experiment


@pytest.fixture(scope="session")
def bundle():
    return load_project_data()


@pytest.fixture(scope="session")
def experiment(bundle):
    return run_experiment(bundle.raw, integrated=True)
