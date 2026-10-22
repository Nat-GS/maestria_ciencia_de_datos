import numpy as np
import pytest
from sklearn.metrics import f1_score

from src.data_processing import FEATURES, NUMERIC
from src.modeling import MODEL_NAMES, model_frame
from src.prediction import (training_importance, run_reduced_experiment,
                            validate_reduced_customer, risk_level, profile_associations)


@pytest.fixture(scope='module')
def reduced(bundle, experiment):
    relevance=training_importance(bundle.raw,experiment)
    return run_reduced_experiment(bundle.raw,experiment,relevance)


def test_reduced_uses_actual_subset_same_clients_and_training_statistics(bundle,experiment,reduced):
    result=reduced.experiment
    assert len(reduced.features)==8 and set(reduced.features)<set(FEATURES)
    assert result.train_ids==experiment.train_ids and result.test_ids==experiment.test_ids
    frame=model_frame(bundle.raw,True)
    for name in MODEL_NAMES:
        model=result.models[name]
        assert set(model.feature_names_in_)==set(reduced.features)
        pp=model['preparacion']; numeric=pp.named_transformers_['numericas']
        columns=[c for c in NUMERIC if c in reduced.features]
        train=numeric['atipicos'].transform(frame.loc[result.train_index,columns])
        np.testing.assert_allclose(numeric['imputacion'].statistics_,np.nanmedian(train,axis=0))
        assert result.metrics.set_index('Modelo').loc[name,'F1-score']==pytest.approx(
            f1_score(result.y_test,result.probabilities[name]>=.5))
    # La explicación de grupos consulta exclusivamente entrenamiento.
    customer=frame.loc[result.train_index].dropna(subset=reduced.features).iloc[0][reduced.features].to_dict()
    associations=profile_associations(bundle.raw,result,customer)
    assert associations['Clientes de entrenamiento'].le(len(result.train_index)).all()
    x=validate_reduced_customer(customer,reduced.features)
    assert x.shape==(1,8)
    for name in MODEL_NAMES:
        assert 0<=result.models[name].predict_proba(x)[0,1]<=1
    with pytest.raises(ValueError): validate_reduced_customer({'tenure':12},reduced.features)
    bad=customer.copy();bad['MonthlyCharges']=-1
    with pytest.raises(ValueError): validate_reduced_customer(bad,reduced.features)


def test_risk_bands_are_explicit_and_service_inputs_consistent():
    assert [risk_level(p) for p in [0,.2999,.30,.5999,.60,1]]==['Bajo','Bajo','Medio','Medio','Alto','Alto']
    customer={'tenure':12,'MonthlyCharges':70,'InternetService':'No','TechSupport':'Yes'}
    with pytest.raises(ValueError): validate_reduced_customer(customer,list(customer))
