import pandas as pd

from src.cleaning import cleaning_steps
from src.cleaning_comparison import cleaning_comparison
from src.integration import prepare_integrated


def test_comparison_matches_pipeline_without_mutation(bundle, experiment):
    raw = bundle.raw.copy(deep=True)
    preparation = prepare_integrated(raw)
    steps = cleaning_steps(raw, experiment)
    before = steps.cleaned.copy(deep=True)
    items, summary, dimensions = cleaning_comparison(raw, steps, preparation, experiment)
    assert len(items) == 6
    assert items[0].table['Entrada inicial'].tolist() == [7303, 260, 222]
    assert items[1].table['Entrada inicial'].tolist() == [555, 487, 373, 119]
    assert items[1].table['Vista final'].eq(0).all()
    assert items[2].table['Celdas faltantes'].tolist() == [1534, 1493, 1626, 1741, 0]
    assert items[5].table[['Antes de normalizar', 'Después de normalizar']].sum().tolist() == [7303, 7303]
    assert summary.set_index('Indicador').loc['Registros'].tolist() == [7303, 7043]
    assert summary.set_index('Indicador').loc['Columnas originales'].tolist() == [21, 21]
    assert dimensions == 46
    for index in [3, 4]:
        pd.testing.assert_series_equal(
            pd.Series(items[index].chart.data[0].x), steps.initial.MonthlyCharges.dropna().reset_index(drop=True),
            check_names=False)
        pd.testing.assert_series_equal(
            pd.Series(items[index].chart.data[1].x), steps.cleaned.MonthlyCharges.reset_index(drop=True),
            check_names=False)
    pd.testing.assert_frame_equal(raw, bundle.raw)
    pd.testing.assert_frame_equal(steps.cleaned, before)
