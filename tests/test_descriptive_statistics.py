import numpy as np
import pandas as pd
import pytest
from src.interpretation import descriptive_statistics,spanish_statistics,descriptive_interpretation
from src.data_processing import clean_data
from src.integration import load_project_data


def test_statistics_orientation_missing_values_and_sample_deviation():
    data=pd.DataFrame({'tenure':[0,2,np.nan], 'MonthlyCharges':[1000,3000,np.nan],
                       'TotalCharges':[np.nan,np.nan,np.nan]})
    stats=descriptive_statistics(data)
    assert stats.shape==(8,3)
    assert stats.columns.tolist()==['tenure','MonthlyCharges','TotalCharges']
    assert stats.loc['Cantidad disponible'].tolist()==[2,2,0]
    assert stats.loc['Desviación estándar','tenure']==pytest.approx(np.sqrt(2))
    assert stats.loc['Primer cuartil (25 %)','MonthlyCharges']==1500
    assert stats.loc['Mediana (50 %)','MonthlyCharges']==2000
    assert stats.loc['Tercer cuartil (75 %)','MonthlyCharges']==2500
    formatted=spanish_statistics(stats)
    assert formatted.loc['Cantidad disponible','tenure']=='2'
    assert formatted.loc['Media','MonthlyCharges']=='2.000,00'
    assert formatted.loc['Media','TotalCharges']=='Sin datos'
    assert stats.loc['Media','MonthlyCharges']==2000  # El formato no altera los cálculos.


def test_integrated_sources_counts_and_dynamic_interpretation():
    frame=clean_data(load_project_data().raw)
    stats=descriptive_statistics(frame)
    assert stats.loc['Cantidad disponible'].tolist()==[6748,6816,6930]
    assert spanish_statistics(stats).loc['Cantidad disponible','tenure']=='6.748'
    assert 'negativos' in descriptive_interpretation(stats)
    modified=frame.copy()
    modified['tenure']=10
    changed=descriptive_statistics(modified)
    assert '10,00 meses' in descriptive_interpretation(changed)
