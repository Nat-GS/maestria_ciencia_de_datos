import numpy as np
import pandas as pd

from src.data_processing import FEATURES, NUMERIC, make_preprocessor
from src.integration import load_project_data, prepare_integrated


def test_imputation_learns_only_training_values_and_preserves_rows():
    frame = prepare_integrated(load_project_data().raw).frame[FEATURES]
    train = frame.iloc[:100].copy()
    train["TotalCharges"] = np.arange(100, dtype=float)
    test = frame.iloc[100:102].copy()
    test["TotalCharges"] = [np.nan, 9999999]
    preprocessor = make_preprocessor(scale=False)
    preprocessor.fit(train)
    result = preprocessor.transform(test)
    assert preprocessor.named_transformers_["numericas"]["imputacion"].statistics_[2] == 49.5
    assert result[0, 2] == 49.5
    assert result[1, 2] == 9999999
    assert result.shape[0] == 2
    assert np.isfinite(result).all()
    assert "customerID" not in preprocessor.get_feature_names_out()
    assert not any(name.startswith("Churn") for name in preprocessor.get_feature_names_out())


def test_scaling_unknown_categories_and_no_mutation():
    frame = prepare_integrated(load_project_data().raw).frame[FEATURES]
    original = frame.copy(deep=True)
    preprocessor = make_preprocessor().fit(frame.iloc[:100])
    unseen = frame.iloc[[101]].copy()
    unseen["PaymentMethod"] = "Categoría futura"
    result = preprocessor.transform(unseen)
    assert np.isfinite(result).all()
    training_numeric = frame.iloc[:100][NUMERIC]
    expected_mean = training_numeric.fillna(training_numeric.median()).mean()
    assert np.allclose(preprocessor.named_transformers_["numericas"]["escalado"].mean_, expected_mean)
    pd.testing.assert_frame_equal(frame, original)
