from copy import copy
import numpy as np
import pytest
from sklearn.metrics import confusion_matrix,f1_score
from sklearn.model_selection import train_test_split
from src.integration import load_project_data
from src.data_processing import FEATURES
from src.modeling import BASELINE,MODEL_NAMES,compare_experiments,model_frame,predict_customer,prepared_view,run_experiment,validate_customer


@pytest.fixture(scope="module")
def experiment():
    return run_experiment(load_project_data().raw,integrated=True)


def test_split_reproducible_disjoint_and_stratified(experiment):
    frame=model_frame(load_project_data().raw,True)
    train,test=train_test_split(frame.index,test_size=.2,stratify=frame.Churn,random_state=42)
    assert experiment.train_index==train.tolist()
    assert experiment.test_index==test.tolist()
    assert len(train)==5634 and len(test)==1409
    assert not set(train)&set(test)
    assert not set(experiment.train_ids)&set(experiment.test_ids)


def test_metrics_match_predictions_and_reference(experiment):
    for name,probability in experiment.probabilities.items():
        assert np.isfinite(probability).all()
        assert ((probability>=0)&(probability<=1)).all()
        matrix=confusion_matrix(experiment.y_test,probability>=.5)
        np.testing.assert_array_equal(matrix,experiment.confusion[name])
        assert matrix.sum()==1409
        row=experiment.metrics.set_index("Modelo").loc[name]
        assert row["F1-score"]==pytest.approx(f1_score(experiment.y_test,probability>=.5))
        assert row.between(0,1).all()
    assert experiment.metrics.set_index("Modelo").loc[BASELINE,"Recall"]==0


def test_selection_depends_on_validation_and_comparison_checks_clients(experiment):
    selected=experiment.validation.sort_values("F1 medio (validación)",ascending=False).iloc[0].Modelo
    assert experiment.selected==selected
    assert compare_experiments(experiment,experiment)["Δ Recall"].eq(0).all()
    other=copy(experiment);other.test_ids=experiment.test_ids[::-1]
    with pytest.raises(ValueError): compare_experiments(experiment,other)


def test_prepared_view_and_individual_predictions(experiment):
    raw=load_project_data().raw
    view=prepared_view(experiment,raw,MODEL_NAMES[0])
    assert len(view)==7043 and view.customerID.is_unique
    assert view.isna().sum().sum()==0
    customer=view.iloc[0][FEATURES].to_dict()
    for name in MODEL_NAMES:
        probability=predict_customer(experiment.models[name],customer)
        assert 0<=probability<=1
        direct=experiment.models[name].predict_proba(validate_customer(customer))[0,1]
        assert probability==direct
    customer["InternetService"]="No"
    with pytest.raises(ValueError): validate_customer(customer)


def test_incomplete_experiments_are_detected(experiment):
    from src.modeling import experiment_complete, experiment_signature
    assert experiment_complete(experiment)
    signature = experiment_signature()
    assert signature == experiment_signature() and len(signature) == 64
    stale = copy(experiment)
    stale.models = {k: v for k, v in experiment.models.items() if k != 'Naive Bayes'}
    assert not experiment_complete(stale)
    stale = copy(experiment)
    stale.metrics = experiment.metrics.loc[experiment.metrics.Modelo.ne('Naive Bayes')]
    assert not experiment_complete(stale)
