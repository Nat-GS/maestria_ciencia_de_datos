import numpy as np

from src.cleaning import cleaning_steps, field_checks, final_checks
from src.modeling import MODEL_NAMES, prepared_view


def test_walkthrough_matches_model_and_measures_real_final_controls(bundle, experiment):
    steps = cleaning_steps(bundle.raw, experiment)
    assert len(steps.initial) == 7303
    assert len(steps.deduplicated) == len(steps.cleaned) == 7043
    assert steps.deduplicated.customerID.is_unique
    assert steps.initial.isna().sum().sum() == 1534
    assert steps.marked.isna().sum().sum() > steps.deduplicated.isna().sum().sum()
    assert steps.cleaned.equals(prepared_view(experiment, bundle.raw, MODEL_NAMES[0]).drop(columns="Partición"))
    assert final_checks(steps.cleaned, steps.upper_bound).Cantidad.eq(0).all()
    assert field_checks(steps.cleaned, steps.upper_bound, completed=True)[["Nulos", "Valores incorrectos"]].eq(0).all().all()


def test_final_controls_do_not_hide_problems_or_count_valid_category_repetitions(bundle, experiment):
    steps = cleaning_steps(bundle.raw, experiment)
    damaged = steps.cleaned.copy()
    damaged.loc[damaged.index[0], "Contract"] = "Contrato mal escrito"
    damaged.loc[damaged.index[1], "MonthlyCharges"] = -99
    damaged.loc[damaged.index[2], "TotalCharges"] = np.nan
    checks = final_checks(damaged, steps.upper_bound).set_index("Comprobación").Cantidad
    assert checks["Celdas nulas"] == 1
    assert checks["Valores incorrectos según las reglas definidas"] == 2
    assert checks["Clientes repetidos por identificador"] == 0
    # Género y contrato repetidos entre distintos clientes son valores válidos.
    assert steps.cleaned.gender.duplicated().any()
    assert final_checks(steps.cleaned, steps.upper_bound).Cantidad.eq(0).all()
