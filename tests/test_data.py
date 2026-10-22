import pytest
from src.integration import load_project_data,prepare_integrated
from src.data_processing import clean_data,consistency_report,quality_report,validate_for_model
from src.exploration import abandonment_table,monthly_by_churn,summary


def test_source_counts_and_schema():
    raw=load_project_data().raw
    assert raw.shape==(7303,21)
    assert raw.customerID.nunique()==7043
    assert raw.Churn.value_counts().to_dict()=={"No":5357,"Yes":1946}


def test_quality_does_not_hide_missing_or_invalid_values():
    raw=load_project_data().raw
    frame=clean_data(raw)
    report=quality_report(raw)
    assert report.Ausentes.sum()==1534
    assert frame.MonthlyCharges.lt(0).sum()==95
    assert report["Valores inválidos"].sum()==191
    assert consistency_report(frame).Incidencias.iloc[3:].sum()==0
    prepared=validate_for_model(prepare_integrated(raw).frame)
    assert len(prepared)==7043
    assert prepared.TotalCharges.isna().sum()==362


def test_kpi_denominators_include_missing_groups():
    frame=clean_data(load_project_data().raw)
    assert summary(frame)["tasa"]==pytest.approx(100*1946/7303)
    for col in ["Contract","InternetService","TramoAntiguedad","TramoCargo"]:
        table=abandonment_table(frame,col)
        assert table.Clientes.sum()==7303
        assert table.Abandonos.sum()==1946
        assert table["Abandono (%)"].between(0,100).all()
    assert monthly_by_churn(frame)["Clientes con cargo válido"].sum()==6816


def test_invalid_values_and_conflicting_ids_block_modeling():
    frame=prepare_integrated(load_project_data().raw).frame
    frame.loc[frame.index[0],"Churn"]="Otro"
    with pytest.raises(ValueError): validate_for_model(frame)
    frame=prepare_integrated(load_project_data().raw).frame
    frame.loc[frame.index[0],"customerID"]=frame.iloc[1].customerID
    with pytest.raises(ValueError): validate_for_model(frame)
