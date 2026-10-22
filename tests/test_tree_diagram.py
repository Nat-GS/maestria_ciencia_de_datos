import re

import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from src.data_processing import make_preprocessor, FEATURES
from src.integration import prepare_integrated
from src.tree_diagram import decision_tree_diagram


def test_tree_diagram_reports_real_counts_and_truncation(bundle):
    frame=prepare_integrated(bundle.raw).frame.iloc[:300]
    x=frame[FEATURES];y=frame.Churn.eq('Yes').astype(int)
    model=Pipeline([('preparacion',make_preprocessor(False,True)),
                    ('modelo',DecisionTreeClassifier(max_depth=3,class_weight='balanced',random_state=42))]).fit(x,y)
    predictions=model.predict_proba(x).copy()
    dot=decision_tree_diagram(model,x,y,max_depth=1)
    assert 'Muestras = 300' in dot
    assert f'Clientes = [{int(y.eq(0).sum())}, {int(y.eq(1).sum())}]' in dot
    assert 'Continúa en niveles inferiores' in dot
    assert 'label="Sí"' in dot and 'label="No"' in dot
    assert len(re.findall(r'^\d+ \[label=',dot,re.MULTILINE))==3
    full=decision_tree_diagram(model,x,y,max_depth=3)
    assert 'Continúa en niveles inferiores' not in full
    assert len(re.findall(r'^\d+ \[label=',full,re.MULTILINE))==model['modelo'].tree_.node_count
    np.testing.assert_array_equal(predictions,model.predict_proba(x))
