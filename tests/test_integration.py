import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from src.data_processing import FEATURES, MonthlyOutlierFilter
from src.integration import SOURCE_PATHS,load_project_data,prepare_integrated
from src.interpretation import category_exhibit,contract_exhibit,numeric_exhibit,numeric_statistics,frequency_table
from src.modeling import MODEL_NAMES,model_frame,prepared_view,run_experiment


@pytest.fixture(scope="module")
def bundle():
    return load_project_data()


@pytest.fixture(scope="module")
def experiment(bundle):
    return run_experiment(bundle.raw,integrated=True)


def test_vertical_integration_provenance_and_immutable_inputs(bundle):
    assert bundle.inventory["SHA-256"].tolist()==[
        "4699d2054e8c740c6899aef71f6e5374fc631c8b2152cecff254f292d775da56",
        "64ff0b1da5c5ba5d180f3b5169f007c0b37f09f5c3d4045fa3712919d2172e3d",
        "def8c8739d9c5adc4a9e05efb9cc410f8cd027d7002d672cfc1daa726b9e37f3"]
    assert bundle.inventory.Filas.tolist()==[2395,2394,2514]
    assert bundle.raw.shape==(7303,21)
    assert len(bundle.provenance)==7303
    assert bundle.provenance.iloc[2395].to_dict()=={"Archivo":"telco_csv_2.csv","Fila CSV":2}
    assert bundle.raw.customerID.nunique()==7043
    assert bundle.raw.duplicated().sum()==222
    assert bundle.raw.customerID.duplicated().sum()==260
    before=bundle.raw.copy(deep=True)
    preparation=prepare_integrated(bundle.raw)
    assert_frame_equal(before,bundle.raw)
    assert preparation.frame.customerID.is_unique
    assert len(preparation.frame)==7043
    assert len(preparation.conflicts)==36
    assert preparation.frame.MonthlyCharges.lt(0).sum()==0
    assert preparation.frame.Contract.dropna().isin(["Month-to-month","One year","Two year"]).all()


def test_optional_legacy_generator_preserves_sources():
    generator=pytest.importorskip("src.generate_dirty_data", reason="Generador histórico opcional no disponible.")
    generate_file=generator.generate_file
    for path in SOURCE_PATHS:
        with pytest.raises(ValueError,match="original"):
            generate_file(destination=path)


def test_conflicting_labels_are_not_silently_resolved(bundle):
    raw=bundle.raw.iloc[:2].copy()
    raw.loc[raw.index[1],"customerID"]=raw.iloc[0].customerID
    raw.loc[raw.index[1],"Churn"]="Yes" if raw.iloc[0].Churn=="No" else "No"
    with pytest.raises(ValueError,match="contradictorias"):
        prepare_integrated(raw)


def test_outlier_limit_is_learned_from_training_only():
    train=pd.DataFrame({"tenure":[1,2,3,4],"MonthlyCharges":[20,30,40,50],"TotalCharges":[20,60,120,200]})
    filter=MonthlyOutlierFilter().fit(train)
    assert filter.upper_bound_==65
    heldout=pd.DataFrame({"tenure":[1],"MonthlyCharges":[99999],"TotalCharges":[1]})
    assert np.isnan(filter.transform(heldout)[0,1])
    assert filter.upper_bound_==65
    assert heldout.MonthlyCharges.iloc[0]==99999


def test_models_share_clients_and_imputation_has_no_leakage(bundle,experiment):
    assert len(experiment.train_ids)==5634
    assert len(experiment.test_ids)==1409
    assert not set(experiment.train_ids)&set(experiment.test_ids)
    frame=model_frame(bundle.raw,True)
    train=frame.loc[experiment.train_index,FEATURES]
    test=frame.loc[experiment.test_index,FEATURES]
    for name in MODEL_NAMES:
        pipeline=experiment.models[name]["preparacion"]
        numeric=pipeline.named_transformers_["numericas"]
        monthly=train.MonthlyCharges.dropna()
        q1,q3=monthly.quantile([.25,.75])
        assert numeric["atipicos"].upper_bound_==pytest.approx(q3+1.5*(q3-q1))
        transformed=numeric["atipicos"].transform(train[["tenure","MonthlyCharges","TotalCharges"]])
        np.testing.assert_allclose(numeric["imputacion"].statistics_,np.nanmedian(transformed,axis=0))
        assert np.isfinite(pipeline.transform(test)).all()
        assert not any(n.startswith("customerID") or n.startswith("Churn") for n in pipeline.get_feature_names_out())
    view=prepared_view(experiment,bundle.raw,MODEL_NAMES[0])
    assert view.isna().sum().sum()==0
    assert view.customerID.is_unique


def test_notebook_tables_and_automatic_interpretations(bundle):
    from src.data_processing import clean_data
    frame=clean_data(bundle.raw)
    assert numeric_statistics(frame).shape==(3,9)
    assert frequency_table(frame,"Contract").Frecuencia.sum()==7303
    for col in ["tenure","MonthlyCharges","TotalCharges"]:
        exhibit=numeric_exhibit(frame,col)
        assert exhibit.table.Registros.sum()==frame[col].notna().sum()
        assert len(exhibit.table)==25
        assert "abandono" in exhibit.interpretation
    exhibit=contract_exhibit(frame)
    assert exhibit.table.Total.sum()==len(frame)-frame.Contract.isna().sum()
    assert exhibit.table.Abandonaron.sum()==frame.loc[frame.Contract.notna()].Churn.eq("Yes").sum()
    toy=frame.iloc[:2].copy()
    toy["Churn"]=["Yes","No"];toy["Contract"]=["One year","Two year"]
    assert "100,0 %" in contract_exhibit(toy).interpretation
    assert "100,0 %" in category_exhibit(toy,"Contract").interpretation
