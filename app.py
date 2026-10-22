"""Presentación del trabajo académico. La lógica analítica reside en src/."""
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.business import OBJECTIVE
from src.data_processing import CATEGORIES, FEATURES, NUMERIC, clean_data, consistency_report, quality_report
from src.exploration import LABELS, data_dictionary, display_value
from src.integration import SOURCE_PATHS, load_project_data, prepare_integrated, quality_summary
from src.interpretation import (TEAL, CORAL, category_exhibit, cleaning_exhibit, contract_exhibit,
                                frequency_table, missing_exhibit, model_conclusions, number,
                                numeric_exhibit, numeric_statistics, styled)
from src.modeling import (MODEL_NAMES, confusion_chart, importance_table, predict_customer,
                          prepared_view, roc_chart, run_experiment, experiment_signature, experiment_complete)

st.set_page_config(page_title="Telco | Proyecto de maestría", page_icon="◈", layout="wide")
st.markdown("""<style>
.block-container{max-width:1380px;padding-top:2.2rem;padding-bottom:3rem}
[data-testid="stSidebar"]{border-right:1px solid #e1e6ea;background:#eef2f4}
[data-testid="stSidebar"] .block-container{padding-top:2rem}
[data-testid="stMetric"]{border:1px solid #e0e6ea;background:#fff;border-radius:12px;padding:18px 20px}
[data-testid="stMetricLabel"]{color:#647582;font-size:.82rem}
[data-testid="stMetricValue"]{color:#183b48}
h1{font-size:2.35rem!important;letter-spacing:-1px;color:#173846}
h2{font-size:1.5rem!important;color:#173846;margin-top:1rem}
h3{font-size:1.12rem!important;color:#284955}
.hero{border-radius:16px;background:linear-gradient(115deg,#153444,#235c60);padding:32px 36px;margin:8px 0 28px}
.hero h1{color:#fff!important;margin:6px 0 10px;line-height:1.15}
.hero p{color:#d9e8e9;max-width:750px;font-size:1rem;margin:0}
.kicker{font-size:.72rem;letter-spacing:2px;text-transform:uppercase;color:#8fb7b6}
.interpretation{border-left:3px solid #237d78;padding:9px 16px;margin:0 0 22px;background:#edf4f3;border-radius:0 8px 8px 0;color:#284a50;font-size:.92rem;line-height:1.6}
.stButton>button,.stDownloadButton>button{border-radius:8px}
</style>""", unsafe_allow_html=True)

SECTIONS = ["Inicio", "Exploración", "Limpieza y preparación", "Modelado", "Predicción", "Conclusiones"]


@st.cache_data
def get_bundle(signatures):
    return load_project_data()


@st.cache_data
def get_preparation(raw):
    return prepare_integrated(raw)


@st.cache_resource(show_spinner="Entrenando modelos y validando con cinco pliegues…")
def train_cached_experiment(raw, implementation_signature):
    return run_experiment(raw, integrated=True)


def get_experiment(raw):
    signature = experiment_signature()
    experiment = train_cached_experiment(raw, signature)
    if not experiment_complete(experiment):
        # No reutilizar objetos antiguos que solo contienen dos clasificadores.
        train_cached_experiment.clear(raw, signature)
        experiment = train_cached_experiment(raw, signature)
    if not experiment_complete(experiment):
        st.error("El entrenamiento no contiene todos los modelos. Revisa la configuración de modelado.")
        st.stop()
    return experiment


@st.cache_data(show_spinner="Analizando relevancia de variables solo con entrenamiento…")
def get_training_importance(raw, signature):
    from src.prediction import training_importance
    return training_importance(raw, get_experiment(raw))


@st.cache_resource(show_spinner="Validando modelos específicos para el formulario simplificado…")
def get_prediction_experiment(raw, signature):
    from src.prediction import run_reduced_experiment
    return run_reduced_experiment(raw, get_experiment(raw), get_training_importance(raw, signature))


def table(data):
    st.dataframe(data, hide_index=True, use_container_width=True)


def download(data, name, label="Descargar tabla"):
    st.download_button(label, data.to_csv(index=False).encode("utf-8-sig"),
                       file_name=name, mime="text/csv", key=f"download_{name}")


def explain(text):
    # Los textos calculados se muestran como Markdown, sin interpolarlos en HTML.
    st.markdown("**Lectura de los resultados**")
    st.caption(text)


def exhibit(item, key, show_table=False):
    st.subheader(item.title)
    item.chart.update_layout(height=340)
    st.plotly_chart(item.chart, use_container_width=True, key=key)
    explain(item.interpretation)
    if show_table and not item.table.empty:
        table(item.table.round(2))


def header(section):
    st.title(section)


def row_metrics(items):
    for col, (label, value) in zip(st.columns(len(items)), items):
        col.metric(label, value)



def quality_panel():
    st.caption("Calidad de los tres archivos integrados, antes de limpiar. Estos controles muestran la base completa y no cambian con los filtros exploratorios.")
    controls=quality_summary(raw)
    row_metrics([("Celdas ausentes",number(observed.isna().sum().sum(),0)),
                 ("Copias exactas",number(raw.duplicated().sum(),0)),
                 ("IDs repetidos · filas extra",number(raw.customerID.duplicated().sum(),0)),
                 ("Cargos negativos",number(observed.MonthlyCharges.lt(0).sum(),0))])
    exhibit(missing_exhibit(observed),"quality_missing")
    a,b=st.columns(2)
    with a:
        st.subheader("Controles de calidad")
        st.dataframe(controls, hide_index=True, use_container_width=True,
                     height=(len(controls)+1)*35+4)
    with b:
        st.subheader("Variantes y ausentes de contrato");table(frequency_table(observed,"Contract").round(2))
    st.caption("El límite RIC de esta auditoría es descriptivo. El tratamiento del modelo aprende un límite independiente dentro de cada entrenamiento y pliegue de validación.")
    with st.expander("Detalle por columna y coherencia entre servicios"):
        table(quality_report(raw));table(consistency_report(observed));download(quality_report(raw),"calidad_integrados.csv")
    with st.expander("Clientes con campos diferentes después de normalizar contratos"):
        table(preparation.conflicts)
        if not preparation.conflicts.empty:
            cid=st.selectbox("Revisar cliente",preparation.conflicts.Cliente.tolist(),key="conflict_customer")
            ids=observed.index[observed.customerID.eq(cid)]
            table(pd.concat([bundle.provenance.loc[ids],observed.loc[ids]],axis=1))
        raw_conflicts=int(controls.set_index("Control").loc["Clientes con campos diferentes en la entrada","Cantidad"])
        st.caption(f"Hay {raw_conflicts} clientes con diferencias en la entrada y {len(preparation.conflicts)} tras normalizar contratos. Sin fechas no se puede afirmar qué versión es más reciente. Se conserva la primera fila por orden de archivo y se registra la decisión; no se consulta el original para elegir.")


try:
    bundle = get_bundle(tuple((str(p), p.stat().st_mtime_ns, p.stat().st_size) for p in SOURCE_PATHS))
    raw = bundle.raw
    observed = clean_data(raw)
    preparation = get_preparation(raw)
except (OSError, ValueError) as error:
    st.error(f"No se pudieron leer o integrar los archivos: {error}")
    st.info("Se necesitan data/telco_csv_1.csv, telco_csv_2.csv y telco_csv_3.csv con el esquema Telco.")
    st.stop()

st.sidebar.markdown("### ◈ TELCO\nAnálisis del abandono")
st.sidebar.caption("TRABAJO FINAL · MÓDULO 1 · MAESTRÍA EN CIENCIA DE DATOS")
section = st.sidebar.radio("Secciones", SECTIONS, label_visibility="collapsed", key="navigation")
st.sidebar.divider()
st.sidebar.caption("Natalia Gutiérrez Soliz\n\nMaestría en Ciencia de Datos · UCB")
st.sidebar.caption(f"{number(len(raw),0)} registros integrados\n\n{number(raw.customerID.nunique(),0)} clientes únicos")

if section == "Inicio":
    st.markdown("""<div class="hero"><div class="kicker">Proyecto final de maestría</div>
    <h1>Análisis y predicción del abandono de clientes de telecomunicaciones</h1>
    <p>Telco Customer Churn</p></div>""", unsafe_allow_html=True)
    st.subheader("Objetivo general")
    st.write(OBJECTIVE)
    row_metrics([("Archivos CSV", str(len(bundle.inventory))), ("Registros integrados", number(len(raw),0)),
                 ("Clientes únicos", number(raw.customerID.nunique(),0)),
                 ("Abandono observado · por fila", number(observed.Churn.eq("Yes").mean()*100)+" %")])
    st.caption("La tasa inicial incluye filas repetidas. Los modelos utilizan una fila por cliente.")
    st.subheader("Archivos de datos")
    st.write("Se leen los tres CSV de data/ y se reúnen sus filas en una sola tabla, conservando las mismas 21 columnas.")
    table(bundle.inventory.drop(columns="SHA-256"))
    selected_file = st.selectbox("Ver datos de",["Integración completa"]+bundle.inventory.Archivo.tolist(),key="source_preview")
    selection = raw if selected_file=="Integración completa" else raw.loc[bundle.provenance.Archivo.eq(selected_file)]
    table(selection.head(30))
    st.caption("Se muestran las primeras 30 filas con los nombres y valores del archivo. Los CSV de entrada no se modifican.")
    download(selection,"datos_seleccionados.csv","Descargar CSV seleccionado")
    with st.expander("Detalles de la integración"):
        st.write("Los archivos se unen en orden 1 → 2 → 3. Se conserva el archivo y la fila de origen para revisar repeticiones y diferencias entre registros del mismo cliente.")
        table(bundle.inventory)
        download(bundle.provenance,"procedencia_integracion.csv")
    st.subheader("Diccionario de datos")
    dictionary = data_dictionary().style.format(escape="html").hide(axis="index")
    dictionary = dictionary.set_table_attributes('class="data-dictionary"')
    st.html("""<style>
    .data-dictionary {width:100%;border-collapse:collapse;table-layout:fixed;font-size:15px;line-height:1.65;color:#243747;background:#fff}
    .data-dictionary th,.data-dictionary td {padding:14px 16px;text-align:left;vertical-align:top;border:1px solid #dce5e8;white-space:normal!important;overflow-wrap:anywhere}
    .data-dictionary thead th {background:#e7eef1;color:#183b48;font-weight:700}
    .data-dictionary th:nth-child(1) {width:21%}
    .data-dictionary th:nth-child(2) {width:65%}
    .data-dictionary th:nth-child(3) {width:14%}
    .data-dictionary tbody td:first-child {background:#d5eae6;color:#153f3b;font-weight:600}
    .data-dictionary tbody tr:nth-child(even) td:not(:first-child) {background:#f6f9fa}
    </style>""" + dictionary.to_html())
    st.link_button("Fuente original en Kaggle","https://www.kaggle.com/datasets/blastchar/telco-customer-churn")

elif section == "Exploración":
    header(section)
    st.write("Explora las distribuciones, el abandono y la calidad de los datos. Los gráficos se recalculan sobre la versión y los filtros seleccionados.")
    version = st.selectbox("Versión de los datos",["Datos tal como se recibieron","Sin clientes repetidos · aún con valores faltantes","Datos limpios · valores faltantes completados"],key="exploration_version")
    if version=="Datos limpios · valores faltantes completados":
        dataset = prepared_view(get_experiment(raw),raw,MODEL_NAMES[0])
    elif version=="Sin clientes repetidos · aún con valores faltantes":
        dataset = preparation.frame
    else:
        dataset = observed
    if version=="Datos limpios · valores faltantes completados":
        st.caption("Se quitaron clientes repetidos, se corrigieron variantes de contrato y se completaron los valores faltantes con medianas o categorías frecuentes calculadas solo con los datos de entrenamiento. No son valores observados: son estimaciones para poder analizar y modelar.")
    elif version=="Sin clientes repetidos · aún con valores faltantes":
        st.caption("Hay una fila por cliente y se corrigieron variantes de contrato y cargos negativos. Los valores faltantes todavía no se completaron.")
    else:
        st.caption("Incluye las repeticiones, los valores faltantes y los errores presentes en los archivos recibidos.")
    with st.expander("Calidad de datos",expanded=False):
        quality_panel()
    with st.expander("Filtrar la exploración",expanded=False):
        a,b=st.columns(2)
        contracts=a.multiselect("Contrato",dataset.Contract.dropna().unique().tolist(),default=dataset.Contract.dropna().unique().tolist(),format_func=display_value,key="filter_contract")
        internet=b.multiselect("Internet",dataset.InternetService.dropna().unique().tolist(),default=dataset.InternetService.dropna().unique().tolist(),format_func=lambda v:"Sin internet" if v=="No" else display_value(v),key="filter_internet")
        include_missing=st.checkbox("Incluir contratos ausentes",value=True,key="filter_missing")
    mask=(dataset.Contract.isin(contracts) | (dataset.Contract.isna() & include_missing)) & dataset.InternetService.isin(internet)
    view=dataset.loc[mask]
    st.caption(f"{number(len(view),0)} de {number(len(dataset),0)} filas. Los filtros no afectan la preparación ni el entrenamiento.")
    if view.empty:
        st.info("No hay registros para estos filtros. Amplía la selección.")
    else:
        row_metrics([("Registros",number(len(view),0)),("Abandono",number(view.Churn.eq("Yes").mean()*100)+" %"),
                     ("Celdas ausentes",number(view[FEATURES].isna().sum().sum(),0))])
        tabs=st.tabs(["Resumen numérico","Distribuciones numéricas","Categorías","Abandono por contrato"])
        with tabs[0]:
            from src.interpretation import descriptive_statistics, spanish_statistics, descriptive_interpretation
            st.subheader("Estadísticas descriptivas de las variables numéricas")
            stats=descriptive_statistics(view)
            st.dataframe(spanish_statistics(stats), use_container_width=True)
            download(stats.reset_index(),"estadisticas_exploracion.csv")
            st.caption("Datos de la versión y filtros seleccionados; por defecto, los tres CSV integrados sin limpiar y sin filtros.")
            st.write(descriptive_interpretation(stats))
        with tabs[1]:
            for container,col in zip(st.columns(3),NUMERIC):
                with container:
                    exhibit(numeric_exhibit(view,col),f"hist_{col}")
        with tabs[2]:
            st.subheader("Frecuencias de todas las variables categóricas")
            col=st.selectbox("Variable",list(CATEGORIES)+["Churn"],format_func=LABELS.get,key="frequency_variable")
            table(frequency_table(view,col).round(2));download(frequency_table(view,col),"frecuencias_categoria.csv")
            st.caption("Las frecuencias incluyen valores ausentes, igual que value_counts(dropna=False) en el notebook.")
            categories=["gender","SeniorCitizen","Contract","InternetService","PaymentMethod","Churn"]
            for start in range(0,len(categories),2):
                for container,col in zip(st.columns(2),categories[start:start+2]):
                    with container:
                        exhibit(category_exhibit(view,col),f"category_{col}")
        with tabs[3]:
            exhibit(contract_exhibit(view),"contract_rates",show_table=True)
            download(contract_exhibit(view).table,"abandono_contrato.csv")
        with st.expander("KPIs complementarios del trabajo académico"):
            from src.exploration import abandonment_chart,abandonment_table,monthly_by_churn
            grouping=st.selectbox("Comparar abandono por",["InternetService","TramoAntiguedad","TramoCargo"],format_func=LABELS.get,key="extra_kpi")
            rates=abandonment_table(view,grouping)
            st.plotly_chart(abandonment_chart(view,grouping),use_container_width=True,key="extra_rates")
            top=rates.loc[rates["Abandono (%)"].idxmax()]
            explain(f"La tasa es abandonos / registros del grupo. «{top.iloc[0]}» alcanza el máximo: {number(top['Abandono (%)'])} %, sobre {int(top.Clientes)} registros. Los intervalos son descriptivos y no demuestran causalidad.")
            table(rates.round(2));table(monthly_by_churn(view).round(2))
        st.link_button("Notebook de exploración","https://colab.research.google.com/drive/1XvwNXWW5_AGOKTid3WT7Dfc16hh7h-vz")

elif section == "Limpieza y preparación":
    from inspect import getsource
    from src.cleaning import cleaning_steps, comparison_chart, field_checks, final_checks, remove_duplicates
    from src.integration import load_project_data, prepare_integrated
    from src.data_processing import make_preprocessor

    header(section)
    st.write("Sigue la unión de los archivos y comprueba con código, cálculos y gráficos cómo cambian los datos. Las fuentes no se modifican.")
    st.subheader("1. Unir los tres CSV")
    st.code('from pathlib import Path\nimport pandas as pd\n\narchivos = [Path("data") / f"telco_csv_{i}.csv" for i in (1, 2, 3)]\ntablas = [pd.read_csv(archivo, dtype=str, keep_default_na=False)\n          for archivo in archivos]\ndatos = pd.concat(tablas, ignore_index=True)\nprint(datos.shape)  # Filas y columnas de la tabla integrada', language="python")
    table(bundle.inventory.drop(columns="SHA-256"))
    st.info(" + ".join(number(n,0) for n in bundle.inventory.Filas)+f" = {number(len(raw),0)} filas y {len(raw.columns)} columnas. Se apilan filas porque los tres archivos tienen el mismo esquema.")
    with st.expander("Ver la función de integración utilizada por la aplicación"):
        st.code(getsource(load_project_data), language="python")

    st.subheader("2. Datos antes de limpiar")
    table(raw.head(20))
    row_metrics([("Registros",number(len(raw),0)),("Clientes únicos",number(raw.customerID.nunique(),0)),
                 ("Campos",str(len(raw.columns))),("Celdas faltantes",number(observed.isna().sum().sum(),0))])
    fields = field_checks(observed)
    fields["Tipo esperado"] = fields.Variable.map(lambda col:
        "Entero entre 0 y 72 meses" if col=="tenure" else
        "Número real no negativo" if col in ["MonthlyCharges","TotalCharges"] else
        "Identificador de cliente" if col=="customerID" else "Categoría del diccionario")
    fields["Tipo leído para análisis"] = fields.Variable.map(observed.dtypes.astype(str))
    with st.expander("Información y problemas de cada campo",expanded=True):
        table(fields)
    st.caption("Un número negativo puede ser válido para el tipo integer o float, pero no para un cargo cobrado o una antigüedad en este proyecto. El control depende del significado del campo, no solo de su tipo Python.")

    experiment=get_experiment(raw)
    steps=cleaning_steps(raw,experiment)
    prepared=steps.cleaned
    st.subheader("3. Detectar y quitar duplicados")
    st.code('copias_exactas = datos.duplicated().sum()\nfilas_extra_por_cliente = datos.duplicated(subset="customerID").sum()\n\n# Conservar la primera aparición según archivo y fila.\nsin_repetidos = datos.drop_duplicates(subset="customerID", keep="first").copy()\nprint(copias_exactas, filas_extra_por_cliente)\nprint(sin_repetidos.shape)',language="python")
    duplicate_before=[int(raw.duplicated().sum()),int(raw.customerID.duplicated().sum())]
    duplicate_after=[int(steps.deduplicated.duplicated().sum()),int(steps.deduplicated.customerID.duplicated().sum())]
    st.plotly_chart(comparison_chart(["Copias exactas","Filas adicionales del mismo cliente"],duplicate_before,duplicate_after),use_container_width=True,key="step_duplicates")
    st.write(f"Se detectan **{duplicate_before[0]} copias exactas**, incluidas en las **{duplicate_before[1]} filas adicionales por cliente**. No se suman ambos conteos. Al conservar una fila por cliente: {number(len(raw),0)} − {duplicate_before[1]} = **{number(len(steps.deduplicated),0)} registros**. Después quedan {duplicate_after[0]} copias exactas y {duplicate_after[1]} identificadores repetidos.")
    st.caption("Los campos como género o tipo de contrato pueden repetirse legítimamente. Aquí se eliminan filas repetidas o clientes repetidos, no valores comunes de cada columna. Los conflictos se registran y las etiquetas de abandono contradictorias bloquean el proceso.")
    with st.expander("Revisar diferencias entre filas del mismo cliente"):
        table(preparation.conflicts)

    st.subheader("4. Detectar y completar los valores nulos")
    st.write("Además de los vacíos existentes, los cargos negativos, contratos desconocidos y cargos extremos tratados se marcan como faltantes antes de completarlos. El siguiente apartado detalla esas correcciones. Los números faltantes se completan con la mediana y las categorías con la más frecuente, usando únicamente entrenamiento.")
    st.code('# Primero se separan clientes únicos en entrenamiento y prueba.\n# Cada pipeline se ajusta solo con entrenamiento.\npipeline.fit(X_train, y_train)\npreparacion = pipeline.named_steps["preparacion"]\n\nX_train_preparado = preparacion.transform(X_train)\nX_test_preparado = preparacion.transform(X_test)\n# La prueba utiliza las estadísticas aprendidas; nunca se hace fit con X_test.',language="python")
    nulls_before=steps.deduplicated.isna().sum()
    nulls_marked=steps.marked.isna().sum()
    nulls_after=prepared.isna().sum()
    affected=nulls_marked.index[(nulls_marked.gt(0)|nulls_before.gt(0))].tolist()
    st.plotly_chart(comparison_chart([LABELS.get(c,c) for c in affected],nulls_before[affected],nulls_after[affected],middle=nulls_marked[affected]),use_container_width=True,key="step_nulls")
    table(pd.DataFrame({"Variable":affected,"Tras quitar duplicados":nulls_before[affected].to_numpy(),"Después de marcar errores":nulls_marked[affected].to_numpy(),"Después de completar":nulls_after[affected].to_numpy()}))
    st.caption(f"Los nulos pasan de {int(nulls_before.sum())} tras quitar duplicados a {int(nulls_marked.sum())} al marcar los errores; finalmente quedan {int(nulls_after.sum())}. El aumento intermedio hace explícitos los valores que no deben usarse como observaciones válidas.")
    numeric=experiment.models[MODEL_NAMES[0]]["preparacion"].named_transformers_["numericas"]
    categorical=experiment.models[MODEL_NAMES[0]]["preparacion"].named_transformers_["categoricas"]
    table(pd.DataFrame({"Variable":NUMERIC,"Mediana aprendida en entrenamiento":numeric["imputacion"].statistics_}))
    contract_position=list(CATEGORIES).index("Contract")
    st.caption("Contrato usado para completar faltantes: "+display_value(categorical["imputacion"].statistics_[contract_position])+". La antigüedad completada es una estimación, no un valor observado.")
    with st.expander("Ver el código del preprocesamiento utilizado"):
        st.code(getsource(make_preprocessor),language="python")

    st.subheader("5. Corregir categorías mal escritas y valores incorrectos")
    st.code('import numpy as np\nfrom src.integration import CONTRACT_MAP\n\n# Variantes inequívocas → nombre de contrato válido.\ncontratos = sin_repetidos["Contract"].astype("string").str.strip().str.casefold()\nsin_repetidos["Contract"] = contratos.map(CONTRACT_MAP)\n# "Desconocido" queda ausente: no se adivina el contrato.\n\nfor columna in ["tenure", "MonthlyCharges", "TotalCharges"]:\n    valores = pd.to_numeric(sin_repetidos[columna], errors="coerce")\n    sin_repetidos[columna] = valores.mask(valores.lt(0))\n\n# El pipeline también marca los cargos mensuales extremos con\n# Q3 + 1,5 × RIC calculado SOLO con entrenamiento, y luego completa faltantes.',language="python")
    st.caption(f"El límite superior de cargo mensual aprendido en entrenamiento es {number(steps.upper_bound,2)} u. m. Cada pliegue de validación aprende el suyo. No se calcula un límite con toda la base para entrenar.")
    before_errors=field_checks(steps.deduplicated,steps.upper_bound)
    after_errors=field_checks(prepared,steps.upper_bound,completed=True)
    affected_errors=before_errors.loc[before_errors["Valores incorrectos"].gt(0),"Variable"].tolist()
    st.plotly_chart(comparison_chart([LABELS.get(c,c) for c in affected_errors],before_errors.set_index("Variable").loc[affected_errors,"Valores incorrectos"],after_errors.set_index("Variable").loc[affected_errors,"Valores incorrectos"]),use_container_width=True,key="step_invalid")
    table(before_errors[["Variable","Valores incorrectos"]].merge(after_errors[["Variable","Valores incorrectos"]],on="Variable",suffixes=(" antes"," después")))
    st.write("El gráfico cuenta categorías fuera del diccionario y números que incumplen las reglas del campo, incluidos los cargos superiores al límite aprendido. Se corrigen las variantes de escritura; los valores sin correspondencia segura se completan como faltantes. No se convierten cargos negativos a positivos ni se recuperan valores del original.")
    category_comparison=pd.concat([steps.deduplicated.Contract.value_counts(dropna=False).rename("Antes"),prepared.Contract.value_counts(dropna=False).rename("Después")],axis=1).fillna(0).astype(int).rename_axis("Contrato").reset_index()
    category_comparison["Contrato"]=category_comparison.Contrato.fillna("Sin dato")
    with st.expander("Ver cómo cambian las categorías de contrato"):
        table(category_comparison)
    for col in NUMERIC:
        with st.expander("Antes y después: "+LABELS[col]):
            exhibit(cleaning_exhibit(steps.initial,prepared,col),f"clean_{col}")
    with st.expander("Ver la función completa de limpieza y su auditoría"):
        st.code(getsource(prepare_integrated),language="python")
        st.code(getsource(cleaning_steps),language="python")

    st.subheader("6. Datos limpios y comprobación final")
    row_metrics([("Datos limpios",number(len(prepared),0)),("Campos",str(len(prepared.columns))),
                 ("Entrenamiento",number(len(experiment.train_index),0)),("Prueba",number(len(experiment.test_index),0))])
    table(prepared)
    download(prepared,"clientes_preparados_vista.csv","Descargar tabla de datos limpios")
    st.subheader("Control final por campo")
    final_fields=field_checks(prepared,steps.upper_bound,completed=True)
    table(final_fields)
    st.subheader("Control final de la tabla completa")
    checks=final_checks(prepared,steps.upper_bound)
    table(checks)
    if checks.Cantidad.eq(0).all():
        st.success("Verificado: 0 nulos, 0 filas duplicadas, 0 clientes repetidos y 0 valores incorrectos según los controles definidos.")
    else:
        st.warning("Quedan incidencias. Revisa los controles antes de usar esta tabla.")
    st.caption("Los ceros se calculan sobre la tabla final; no se fijan en el código. La vista está completada con estadísticas de entrenamiento, antes de codificar y escalar. Para entrenar se mantiene el pipeline original, no se reutiliza esta descarga como entrada.")

    st.subheader("Comparación de datos antes y después de la limpieza")
    from src.cleaning_comparison import cleaning_comparison
    comparisons, summary_table, predictor_count = cleaning_comparison(raw, steps, preparation, experiment)
    st.caption("Comparación calculada sobre los tres CSV completos. No aplica los filtros de Exploración ni reajusta el pipeline.")
    for index, item in enumerate(comparisons):
        st.markdown(f"**{item.title}**")
        st.plotly_chart(item.chart, use_container_width=True, key=f"cleaning_comparison_{index}")
        explain(item.interpretation)
    st.markdown("**Resumen final: entrada y vista de limpieza**")
    table(summary_table)
    st.caption(f"Los valores inválidos incluyen categorías fuera del diccionario, errores numéricos y cargos superiores al límite aprendido; los faltantes se cuentan por separado. Las copias exactas iniciales se auditan en los CSV leídos. La vista final mantiene {len(prepared.columns)} columnas originales, incluyendo customerID y Churn. La matriz de modelado contiene {predictor_count} variables predictoras tras codificar las categorías; excluye customerID y Churn. Los gráficos usan transformaciones ya ajustadas con entrenamiento, sin hacer fit sobre la prueba ni sobre la base completa.")

    st.subheader("Revisión final de valores atípicos")
    from src.cleaning import outlier_review
    outlier_figure, outlier_counts = outlier_review(steps.deduplicated, prepared)
    st.plotly_chart(outlier_figure, use_container_width=True, key="final_outlier_review")
    explain("Los diagramas de caja comparan los mismos clientes antes y después de limpiar, con una escala propia para cada variable. Los puntos fuera de los bigotes señalan valores más allá de 1,5 veces el rango intercuartílico. " +
            " · ".join(f"{row.Variable}: {row.Antes} antes y {row.Después} después" for row in outlier_counts.itertuples(index=False)) + ".")
    st.caption("Estos conteos son descriptivos y se calculan por separado en cada versión. No sustituyen el límite aprendido en entrenamiento ni modifican datos o modelos. Un valor atípico no implica necesariamente un error; puede permanecer después de la limpieza.")

elif section == "Modelado":
    from src.model_analysis import roc_plot, precision_recall_plot, probability_plot
    from src.prediction import TECHNIQUES
    header(section)
    experiment=get_experiment(raw)
    relevance=get_training_importance(raw,experiment_signature())
    st.caption(f"Experimento completo: {len(FEATURES)} predictores originales y {len(experiment.models[MODEL_NAMES[0]]['preparacion'].get_feature_names_out())} columnas transformadas. customerID y Churn no forman parte de la matriz predictora.")
    st.subheader("Plan de prueba y prevención de fuga de información")
    plan=[]
    for label,indices in [("Entrenamiento",experiment.train_index),("Prueba final",experiment.test_index)]:
        labels=preparation.frame.loc[indices,"Churn"]
        plan.append({"Partición":label,"Clientes":len(indices),"Porcentaje":100*len(indices)/len(preparation.frame),
                     "Abandonos":int(labels.eq("Yes").sum()),"Abandono (%)":100*labels.eq("Yes").mean()})
    table(pd.DataFrame(plan).round(2))
    explain("La división estratificada 80/20 conserva aproximadamente la proporción de abandono. Se consolidan los clientes antes de dividir; ningún ID cruza ambas particiones. Semilla 42. La prueba final se reserva para evaluación.")
    st.caption("En cada uno de los cinco pliegues, el pipeline aprende el límite mensual, medianas, modas, categorías y escalado únicamente con el entrenamiento del pliegue. GridSearchCV optimiza F1; el modelo ganador se reajusta con todo entrenamiento. La prueba solo se transforma. No se optimizan umbrales con ella.")
    st.caption(f"{number(len(experiment.train_index),0)} clientes de entrenamiento · {number(len(experiment.test_index),0)} de prueba · validación de 5 pliegues · umbral 0,50")
    model_tabs=st.tabs(MODEL_NAMES+["Comparación"])
    metric_labels={"Accuracy":"Exactitud","Precision":"Precisión","Recall":"Sensibilidad (recall)","F1-score":"F1","ROC-AUC":"ROC-AUC"}
    for index,name in enumerate(MODEL_NAMES):
        with model_tabs[index]:
            st.write(TECHNIQUES[name])
            score=experiment.metrics.set_index("Modelo").loc[name]
            row_metrics([(label,number(score[key],4)) for key,label in metric_labels.items()])
            if name=="Árbol de decisión":
                from src.tree_diagram import decision_tree_diagram, tree_download_html
                st.subheader("Diagrama del árbol de decisión")
                tree_model=experiment.models[name]
                tree_depth=tree_model['modelo'].get_depth()
                display_depth=st.slider("Profundidad visible del árbol",min_value=1,max_value=max(2,tree_depth),
                    value=min(3,max(2,tree_depth)),key="tree_display_depth")
                training_rows=preparation.frame.loc[experiment.train_index]
                diagram=decision_tree_diagram(tree_model,training_rows,
                    training_rows.Churn.map({'No':0,'Yes':1}),max_depth=display_depth)
                st.graphviz_chart(diagram,width="stretch",height=700)
                import streamlit.components.v1 as components
                components.html(tree_download_html(),height=85)
                st.caption("La imagen PNG conserva el diagrama mostrado y la profundidad seleccionada, sobre fondo blanco.")
                st.caption(f"Árbol completo de {tree_depth} niveles de profundidad, ajustado con {number(len(training_rows),0)} clientes de entrenamiento. El control muestra sus primeros niveles y no cambia el modelo; los nodos truncados indican que el árbol continúa.")
                explain("Cada rama responde Sí o No a la condición del nodo. Muestras indica clientes de entrenamiento; Clientes = [permanencia, abandono] muestra conteos observados. Naranja identifica permanencia y verde, abandono. Gini y clase estimada incorporan los pesos de clase usados por el modelo, por lo que la clase puede diferir de la mayoría de conteos sin ponderar. En categorías codificadas, ≤ 0,50 significa que esa categoría está ausente. Las reglas describen asociaciones predictivas, no causas.")
            a,b=st.columns(2)
            with a:
                st.subheader("Matriz de confusión")
                st.plotly_chart(confusion_chart(experiment.confusion[name]),use_container_width=True,key=f"matrix_{index}")
                tn,fp,fn,tp=experiment.confusion[name].ravel()
                st.caption(f"{tp} abandonos detectados · {fn} omitidos · {fp} falsas alertas · {tn} permanencias correctas")
            with b:
                st.subheader("Curva ROC")
                st.plotly_chart(roc_plot(experiment.y_test,experiment.probabilities[name]),use_container_width=True,key=f"roc_{index}")
                st.caption(f"ROC-AUC = {number(score['ROC-AUC'],4)}: capacidad para ordenar clientes de ambas clases a distintos umbrales. No mide calibración de probabilidades.")
            a,b=st.columns(2)
            with a:
                st.subheader("Precisión y sensibilidad")
                st.plotly_chart(precision_recall_plot(experiment.y_test,experiment.probabilities[name]),use_container_width=True,key=f"pr_{index}")
                st.caption(f"Con umbral 0,50, recall = {number(score['Recall'],4)} y precisión = {number(score['Precision'],4)}. El primero mide abandonos detectados; la segunda, la proporción de alertas correctas.")
            with b:
                st.subheader("Probabilidades por resultado real")
                st.plotly_chart(probability_plot(experiment.y_test,experiment.probabilities[name]),use_container_width=True,key=f"prob_{index}")
                st.caption("La superposición entre clases muestra incertidumbre de clasificación. Los histogramas son frecuencias y reflejan el desbalance; una probabilidad alta no garantiza abandono.")
            st.subheader("Relevancia de las variables originales")
            permutation=relevance.loc[relevance.Modelo.eq(name)].sort_values("Caída de F1",ascending=False)
            top_permutation=permutation.head(10).iloc[::-1].copy()
            fig=go.Figure(go.Bar(x=top_permutation['Caída de F1'],y=top_permutation.Variable.map(LABELS),orientation="h",
                error_x=dict(type="data",array=top_permutation.Desviación),marker_color=TEAL,name="Caída de F1"))
            fig.update_layout(height=380,showlegend=True)
            st.plotly_chart(styled(fig,"Caída de F1 al permutar","Variable original"),use_container_width=True,key=f"permutation_{index}")
            st.caption("Permutación compatible con los tres algoritmos: cinco repeticiones sobre una reserva interna del entrenamiento. Las barras de error indican desviación. Una caída mayor señala más dependencia predictiva; valores negativos pueden reflejar ruido. No son efectos causales ni resultados de la prueba final. Las configuraciones proceden de validación sobre entrenamiento; esta reserva no es una evaluación independiente.")
            details=importance_table(experiment.models[name])
            if name=="Naive Bayes":
                with st.expander("Medias por clase de las variables transformadas"):
                    table(details.round(4))
                    st.caption("GaussianNB estima medias separadas para permanencia y abandono sobre entradas transformadas. Sus diferencias describen distribuciones del entrenamiento; no son coeficientes causales ni medidas directamente comparables con la importancia del árbol.")
            else:
                label="Coeficientes de la regresión logística" if name=="Regresión logística" else "Importancia de las variables en el árbol"
                st.subheader(label)
                top=details.head(15).iloc[::-1]
                fig=go.Figure(go.Bar(x=top.Valor,y=top["Variable transformada"],orientation="h",marker_color=[TEAL if v>=0 else CORAL for v in top.Valor]))
                fig.update_layout(height=420)
                st.plotly_chart(styled(fig,"Coeficiente" if name=="Regresión logística" else "Importancia",None),use_container_width=True,key=f"importance_{index}")
                st.caption("Variables transformadas; asociaciones del modelo, sin interpretación causal.")
            with st.expander("Parámetros y resultados de validación"):
                st.write("Mejores hiperparámetros de la búsqueda")
                st.json(experiment.parameters[name])
                table(experiment.validation.loc[experiment.validation.Modelo.eq(name)].round(4))
                st.caption("Medias de cinco pliegues del entrenamiento. El criterio de búsqueda es F1 de Churn=Yes; se informa recall y ROC-AUC para valorar detección y discriminación.")
                st.write("Configuración efectiva del estimador")
                st.json(experiment.models[name]['modelo'].get_params())
    with model_tabs[-1]:
        st.subheader("Resultados en la misma prueba")
        comparison_metrics=experiment.metrics.loc[experiment.metrics.Modelo.isin(MODEL_NAMES)]
        table(comparison_metrics.rename(columns=metric_labels).round(4))
        explain("Los tres algoritmos se evalúan sobre los mismos clientes de prueba, con umbral 0,50 y Churn=Yes como clase positiva. Accuracy debe interpretarse junto con recall y F1 por el desbalance de clases. La selección del algoritmo se hace con validación, no buscando el mayor resultado de esta tabla.")
        download(experiment.metrics,"metricas_modelos_integrados.csv")
        fig=go.Figure()
        for metric,color in [("Recall",TEAL),("F1-score",CORAL),("ROC-AUC","#8295a3")]:
            fig.add_trace(go.Bar(x=comparison_metrics.Modelo,y=comparison_metrics[metric],name=metric_labels[metric],marker_color=color))
        fig.update_layout(barmode="group",yaxis_range=[0,1],height=360)
        st.plotly_chart(styled(fig,None,"Puntuación"),use_container_width=True,key="model_comparison")
        st.caption("Recall resume detección de abandonos; F1 equilibra precisión y recall; ROC-AUC compara discriminación sin depender de un único umbral. No existe un ganador universal para todos los costos de retención.")
        st.subheader("Curvas ROC comparadas")
        st.plotly_chart(roc_chart(experiment),use_container_width=True,key="comparison_roc")
        st.caption("Las curvas incluyen la referencia de clase mayoritaria. Una curva cercana a la diagonal aporta poca discriminación; una curva más alta indica mayor sensibilidad para una tasa dada de falsas alertas.")
        st.subheader("Validación cruzada")
        table(experiment.validation.round(4))
        st.success(f"Seleccionado por F1 en validación: {experiment.selected}")
        chosen=experiment.validation.set_index('Modelo').loc[experiment.selected]
        explain(f"Criterio explícito: mayor F1 medio de validación; desempate por recall y después ROC-AUC. {experiment.selected} obtiene F1={number(chosen['F1 medio (validación)'],4)}, recall={number(chosen['Recall medio (validación)'],4)} y ROC-AUC={number(chosen['ROC-AUC medio (validación)'],4)}. La prioridad académica es detectar abandonos equilibrando falsas alertas; no se cambia de ganador con la prueba.")
        with st.expander("Búsqueda de hiperparámetros"):
            table(experiment.candidates.round(4))
            st.caption("Cada fila es una configuración realmente evaluada en entrenamiento. No representa una búsqueda sobre los clientes de prueba.")

    st.subheader("Modelo específico para predicción simplificada")
    reduced=get_prediction_experiment(raw,experiment_signature())
    st.write("Ocho entradas: "+", ".join(LABELS[c] for c in reduced.features)+".")
    st.caption(f"Experimento reducido: {len(reduced.features)} entradas originales y {len(reduced.experiment.models[MODEL_NAMES[0]]['preparacion'].get_feature_names_out())} columnas transformadas. La limpieza conserva la base de clientes; cambia únicamente el conjunto de predictores de estos modelos específicos.")
    table(reduced.ranking.assign(Variable=lambda d:d.Variable.map(LABELS)).round(4))
    st.caption("Se mantienen antigüedad y cargo mensual como entradas numéricas base para reutilizar el filtro validado. Las otras seis se eligen por caída media de F1 al permutar, promediada entre los tres modelos solo en entrenamiento. La selección es previa a las métricas de prueba y no utiliza sus etiquetas. Esta selección no está anidada en la validación posterior; sus medias de validación pueden ser optimistas y la prueba permanece como comprobación separada.")
    reduced_metrics=reduced.experiment.metrics.loc[lambda d:d.Modelo.isin(MODEL_NAMES)]
    table(reduced_metrics.rename(columns=metric_labels).round(4))
    explain(f"Resultados de prueba de modelos nuevos con ocho entradas, sobre los mismos clientes y con la misma búsqueda y preparación que el experimento completo. Selección por validación: {reduced.experiment.selected}. Estas métricas corresponden al formulario; las métricas anteriores corresponden a 19 entradas.")
    validation_order=reduced.experiment.validation.sort_values('F1 medio (validación)',ascending=False)
    gap=validation_order['F1 medio (validación)'].iloc[0]-validation_order['F1 medio (validación)'].iloc[1]
    st.caption(f"La diferencia de F1 medio entre los dos primeros modelos reducidos es {number(gap,6)}. La selección es una regla reproducible; no demuestra una diferencia estadísticamente significativa. Conviene comprobar estabilidad con nuevas muestras.")
    download(reduced_metrics,"metricas_modelos_simplificados.csv")
    with st.expander("Validación y mejores parámetros de los modelos simplificados"):
        table(reduced.experiment.validation.round(4))
        st.json(reduced.experiment.parameters)
        st.caption("Resultados de cinco pliegues de entrenamiento. Cada pipeline conserva solo las columnas seleccionadas y aprende sus transformaciones dentro de cada pliegue.")
    st.subheader("Evidencia para evaluación y propuesta de implementación")
    selected_score=experiment.metrics.set_index('Modelo').loc[experiment.selected]
    baseline=experiment.metrics.set_index('Modelo').loc['Referencia mayoritaria']
    st.write(f"En esta prueba, el modelo completo seleccionado ({experiment.selected}) alcanza recall {number(selected_score.Recall,4)}, F1 {number(selected_score['F1-score'],4)} y ROC-AUC {number(selected_score['ROC-AUC'],4)}. Frente a la referencia mayoritaria (recall {number(baseline.Recall,4)}, F1 {number(baseline['F1-score'],4)}), aporta capacidad de identificar abandonos. Esto respalda el objetivo académico; no demuestra utilidad económica ni generalización a una empresa real.")
    st.write(f"Recomendación académica: {experiment.selected} para el análisis completo y {reduced.experiment.selected} para el formulario reducido, por el criterio de validación descrito. Antes de uso empresarial se requieren datos reales autorizados, validación temporal, revisión de conflictos, calibración y evaluación de costos y umbrales.")
    st.write("Implementado: comparación local reproducible y estimación individual sin guardar datos. Propuesta futura: incorporar puntuaciones a una lista revisada por el equipo de retención, comprobar satisfacción y evaluar campañas con un grupo de comparación. No hay integración con CRM, campañas automáticas ni despliegue empresarial.")
    st.write("Monitoreo propuesto: recall, F1, ROC-AUC, precisión, calibración, proporción de abandono y calidad de datos (faltantes, duplicados, categorías nuevas, cargos inválidos). Vigilar cambios en distribuciones y segmentos. Revisar/reentrenar ante deterioro sostenido frente a una referencia acordada, cambios de oferta o población, cambios de esquema o nuevas etiquetas; validar de nuevo antes de sustituir el modelo. No existen alertas ni reentrenamiento automático implementados.")

elif section == "Predicción":
    from src.prediction import (validate_reduced_customer, risk_level, RISK_THRESHOLDS,
                                profile_associations, retention_suggestions)
    header(section)
    reduced=get_prediction_experiment(raw,experiment_signature())
    experiment=reduced.experiment
    st.write(f"Estima el riesgo con ocho preguntas y modelos específicos evaluados con estas entradas. Recomendado por validación: {experiment.selected}. Los datos introducidos no se guardan.")
    selected=st.selectbox("Modelo simplificado",MODEL_NAMES,index=MODEL_NAMES.index(experiment.selected),key="predict_model")
    preprocessor=experiment.models[selected]['preparacion']
    medians=dict(zip([c for c in NUMERIC if c in reduced.features],preprocessor.named_transformers_['numericas']['imputacion'].statistics_))
    modes=dict(zip([c for c in CATEGORIES if c in reduced.features],preprocessor.named_transformers_['categoricas']['imputacion'].statistics_))
    customer={}
    # Estos controles actualizan opciones dependientes antes de enviar el formulario.
    for col in ['InternetService','PhoneService']:
        if col in reduced.features:
            options=CATEGORIES[col]
            customer[col]=st.selectbox(LABELS[col],options,index=options.index(modes[col]),
                format_func=lambda v, field=col: "Sin internet" if v=="No" and field=="InternetService" else display_value(v),
                key='predict_internet' if col=='InternetService' else 'predict_phone')
    with st.form("prediction"):
        cols=st.columns(2)
        for i,col in enumerate(c for c in reduced.features if c not in customer):
            if col in NUMERIC:
                if col=='tenure':
                    customer[col]=cols[i%2].number_input("Antigüedad (meses)",min_value=0,max_value=72,value=int(medians[col]),step=1,key='predict_tenure')
                else:
                    customer[col]=cols[i%2].number_input(LABELS[col]+" (u. m.)",min_value=0.0,value=float(medians[col]),step=1.0,key=f'predict_{col}')
            else:
                options=CATEGORIES[col]
                if col=='MultipleLines' and 'PhoneService' in customer:
                    options=['No phone service'] if customer['PhoneService']=='No' else ['No','Yes']
                if col in ['OnlineSecurity','TechSupport','OnlineBackup','DeviceProtection','StreamingTV','StreamingMovies'] and 'InternetService' in customer:
                    options=['No internet service'] if customer['InternetService']=='No' else ['No','Yes']
                default=modes[col] if modes[col] in options else options[0]
                customer[col]=cols[i%2].selectbox(LABELS[col],options,index=options.index(default),format_func=display_value,key=f'predict_{col}')
        submitted=st.form_submit_button("Estimar abandono",type="primary")
    st.caption(f"Riesgo bajo: menos de {number(RISK_THRESHOLDS[0]*100,0)} %; medio: desde {number(RISK_THRESHOLDS[0]*100,0)} % hasta menos de {number(RISK_THRESHOLDS[1]*100,0)} %; alto: desde {number(RISK_THRESHOLDS[1]*100,0)} %. Son bandas ilustrativas, no umbrales optimizados ni decisiones comerciales validadas. Los valores predeterminados proceden de entrenamiento.")
    if submitted:
        try:
            probability=float(experiment.models[selected].predict_proba(validate_reduced_customer(customer,reduced.features))[0,1])
            st.metric("Probabilidad estimada de abandono",number(probability*100)+" %")
            st.progress(probability)
            st.write(f"**Riesgo {risk_level(probability).lower()}** · Clasificación binaria con umbral 0,50: "+("Abandono" if probability>=.5 else "Permanencia"))
            threshold=experiment.models[selected]["preparacion"].named_transformers_["numericas"]["atipicos"].upper_bound_
            if customer["MonthlyCharges"]>threshold:
                st.info(f"El cargo mensual supera el límite aprendido ({number(threshold,2)}); se trata como ausente y se completa con la mediana de entrenamiento. No se interpreta ese cargo extremo como una observación válida.")
            st.subheader("Factores asociados en grupos comparables")
            associations=profile_associations(raw,experiment,customer).head(4)
            associations['Variable']=associations.Variable.map(LABELS)
            associations['Grupo comparable']=associations['Grupo comparable'].map(lambda v:display_value(v))
            table(associations.round(2))
            explain("La tabla compara características del cliente con grupos observados en entrenamiento. Una diferencia positiva señala mayor abandono observado frente al conjunto de entrenamiento. Son asociaciones descriptivas, no contribuciones individuales del modelo ni causas del resultado; los grupos pueden ser pequeños o estar relacionados entre sí.")
            st.subheader("Orientaciones generales de retención")
            for advice in retention_suggestions(customer):
                st.markdown('- '+advice)
            st.caption("Estimación estadística académica, no una certeza. Las probabilidades no tienen calibración operativa. Las recomendaciones son propuestas generales sin efectividad ni ahorros demostrados.")
        except ValueError as error:
            st.error(str(error))

else:
    header(section)
    experiment=get_experiment(raw)
    prepared=prepared_view(experiment,raw,MODEL_NAMES[0])
    st.subheader("Lo que muestran los datos")
    st.write(f"Se integraron {number(len(raw),0)} filas de tres archivos y se conservaron {number(len(prepared),0)} clientes únicos. La preparación elimina repeticiones, normaliza contratos y trata ausentes y cargos inválidos sin usar información de prueba.")
    rates=contract_exhibit(prepared)
    explain(rates.interpretation.replace("En la vista sucia las variantes siguen separadas; ",""))
    st.subheader("Lo que muestran los modelos")
    for insight in model_conclusions(experiment):
        st.markdown(f"- {insight}")
    st.subheader("Implicaciones y límites")
    st.write("Los segmentos con mayor abandono observado orientan hipótesis de retención que requieren validación. Las asociaciones no demuestran causas. Sin costos de campañas no se estima retorno de inversión; sin fechas no se comprueba generalización temporal. La elección de la primera fila en conflictos no identifica una verdad histórica: es una regla reproducible con registro de auditoría.")
    st.write("Próximos pasos académicos: revisar los conflictos con la fuente, validar sensibilidad a otros tratamientos de extremos, calibrar probabilidades y evaluar estabilidad en otras particiones sin reutilizar la prueba para ajustar parámetros.")
    st.caption("Esta sección sintetiza las etapas 1–4. No constituye implementación de las etapas empresariales 5 y 6 de CRISP-DM.")

st.divider()
st.caption("Proyecto académico · Natalia Gutiérrez Soliz · Datos locales")
