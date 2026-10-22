from streamlit.testing.v1 import AppTest


def test_navigation_filters_preparation_and_prediction():
    app=AppTest.from_file("app.py",default_timeout=60).run()
    assert not app.exception
    assert app.sidebar.radio(key="navigation").options==["Inicio","Exploración","Limpieza y preparación","Modelado","Predicción","Conclusiones"]
    assert any(v.value=="Objetivo general" for v in app.subheader)
    assert any(v.value=="Diccionario de datos" for v in app.subheader)
    app.selectbox(key="source_preview").set_value("telco_csv_2.csv").run()
    assert not app.exception
    sections=["Exploración","Limpieza y preparación","Modelado","Predicción","Conclusiones"]
    for section in sections:
        app.sidebar.radio(key="navigation").set_value(section).run()
        assert not app.exception,section
        assert not app.error,section
        if section=="Limpieza y preparación":
            assert len(app.code)>=4
            assert any(v.value=="Comparación de datos antes y después de la limpieza" for v in app.subheader)
            assert any(v.value=="Control final por campo" for v in app.subheader)
            assert any(v.value=="Control final de la tabla completa" for v in app.subheader)
            assert any("Verificado: 0 nulos" in v.value for v in app.success)
        if section=="Exploración":
            assert any(v.value=="Estadísticas descriptivas de las variables numéricas" for v in app.subheader)
            descriptive=[v.value for v in app.dataframe if v.value.index.name=="Estadístico"]
            assert len(descriptive)==1 and descriptive[0].shape==(8,3)
            assert descriptive[0].loc["Cantidad disponible","tenure"]=="6.748"
            assert any(v.label=="Calidad de datos" for v in app.expander)
            assert any(v.value=="Dónde falta información" for v in app.subheader)
            app.multiselect(key="filter_contract").set_value([])
            app.checkbox(key="filter_missing").set_value(False).run()
            assert any("No hay registros" in v.value for v in app.info)
            assert not app.exception
            app.selectbox(key="exploration_version").set_value("Datos limpios · valores faltantes completados").run()
            assert not app.exception
        if section=="Modelado":
            assert [tab.label for tab in app.tabs]==["Regresión logística","Árbol de decisión","Naive Bayes","Comparación"]
            assert any(v.value=="Diagrama del árbol de decisión" for v in app.subheader)
            assert len(app.get('graphviz_chart'))==1
            assert any('Descargar árbol en PNG' in item.proto.srcdoc for item in app.get('iframe'))
            app.slider(key='tree_display_depth').set_value(2).run()
            assert not app.exception
            assert any(v.value=="Modelo específico para predicción simplificada" for v in app.subheader)
            assert any(v.value=="Evidencia para evaluación y propuesta de implementación" for v in app.subheader)
        if section=="Predicción":
            app.button[0].click().run()
            assert not app.exception
            assert any(v.label=="Probabilidad estimada de abandono" for v in app.metric)
            app.selectbox(key="predict_internet").set_value("No").run()
            assert app.selectbox(key="predict_TechSupport").options==["Sin internet"]
            app.selectbox(key="predict_model").set_value("Árbol de decisión").run()
            app.button[0].click().run()
            assert not app.exception
            assert not app.error
            app.selectbox(key="predict_model").set_value("Naive Bayes").run()
            app.button[0].click().run()
            assert not app.exception
            assert not app.error


def test_old_two_model_cache_is_retrained(monkeypatch, experiment):
    from copy import copy
    import streamlit as st
    import src.modeling as modeling

    stale=copy(experiment)
    stale.models={k:v for k,v in experiment.models.items() if k!='Naive Bayes'}
    calls=[]
    def simulate_old_experiment(raw, *, integrated=False, features=None):
        if features is not None:
            return original_run(raw,integrated=integrated,features=features)
        calls.append(integrated)
        return stale if len(calls)==1 else experiment
    original_run=modeling.run_experiment
    st.cache_resource.clear()
    monkeypatch.setattr(modeling,'run_experiment',simulate_old_experiment)
    app=AppTest.from_file('app.py',default_timeout=60).run()
    app.sidebar.radio(key='navigation').set_value('Modelado').run()
    assert not app.exception and not app.error
    assert len(calls)==2
    assert [tab.label for tab in app.tabs]==['Regresión logística','Árbol de decisión','Naive Bayes','Comparación']
    app.run()
    assert not app.exception and len(calls)==2
    st.cache_resource.clear()
