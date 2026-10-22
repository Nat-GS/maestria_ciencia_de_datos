"""Informe del flujo integrado: python -m src.report --output resultados."""
import argparse
import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

from src.data_processing import NUMERIC, ROOT, clean_data, quality_report
from src.exploration import abandonment_table, monthly_by_churn
from src.integration import load_project_data, prepare_integrated, quality_summary
from src.interpretation import frequency_table, numeric_statistics, contract_exhibit
from src.modeling import MODEL_NAMES, prepared_view, run_experiment
from src.cleaning import field_checks, final_checks


def export_report(output: Path):
    output.mkdir(parents=True, exist_ok=True)
    bundle=load_project_data()
    raw=bundle.raw
    observed=clean_data(raw)
    preparation=prepare_integrated(raw)
    experiment=run_experiment(raw,integrated=True)
    prepared=prepared_view(experiment,raw,MODEL_NAMES[0])
    tables={"metricas_prueba":experiment.metrics,"validacion":experiment.validation,
            "busqueda":experiment.candidates,"fuentes":bundle.inventory,
            "procedencia":bundle.provenance,"calidad_columnas":quality_report(raw),
            "calidad_resumen":quality_summary(raw),"acciones_preparacion":preparation.actions,
            "conflictos":preparation.conflicts,"vista_preparada":prepared,
            "controles_finales":final_checks(prepared.drop(columns="Partición"),
                experiment.models[MODEL_NAMES[0]]["preparacion"].named_transformers_["numericas"]["atipicos"].upper_bound_),
            "controles_finales_por_campo":field_checks(prepared.drop(columns="Partición"),
                experiment.models[MODEL_NAMES[0]]["preparacion"].named_transformers_["numericas"]["atipicos"].upper_bound_,completed=True),
            "estadisticas_integrados":numeric_statistics(observed),
            "estadisticas_preparados":numeric_statistics(prepared),
            "contratos_integrados":contract_exhibit(observed).table,
            "cargos_por_abandono":monthly_by_churn(prepared)}
    for name,frame in [("integrados",observed),("consolidados",preparation.frame),("preparados",prepared)]:
        tables[f"ausentes_{name}"]=frame[raw.columns].isna().sum().rename_axis("Variable").reset_index(name="Ausentes")
    for group in ["Contract","InternetService","TramoAntiguedad","TramoCargo"]:
        tables[f"abandono_{group}"]=abandonment_table(prepared,group)
    for group in raw.columns:
        if group not in NUMERIC+["customerID"]:
            tables[f"frecuencias_{group}"]=frequency_table(observed,group)
    for name,data in tables.items():
        data.to_csv(output/f"{name}.csv",index=False,encoding="utf-8-sig")
    numeric=experiment.models[MODEL_NAMES[0]]["preparacion"].named_transformers_["numericas"]
    metadata={"fuentes":bundle.inventory.to_dict(orient="records"),
              "filas_integradas":len(raw),"clientes_preparados":len(prepared),
              "modelo_seleccionado_validacion":experiment.selected,
              "parametros":experiment.parameters,
              "indices_entrenamiento":experiment.train_index,"indices_prueba":experiment.test_index,
              "clientes_entrenamiento":experiment.train_ids,"clientes_prueba":experiment.test_ids,
              "matrices_confusion":{k:v.tolist() for k,v in experiment.confusion.items()},
              "limite_cargo_mensual_entrenamiento":numeric["atipicos"].upper_bound_,
              "medianas_entrenamiento":dict(zip(NUMERIC,numeric["imputacion"].statistics_.tolist())),
              "columnas_codificadas":len(experiment.models[MODEL_NAMES[0]]["preparacion"].get_feature_names_out()),
              "semilla":42,"python":platform.python_version(),
              "versiones":{name:version(name) for name in ["pandas","numpy","scikit-learn","plotly","streamlit"]},
              "sha256_notebook_exploracion":hashlib.sha256((ROOT/"docs/referencias/Exploracion.ipynb").read_bytes()).hexdigest()}
    (output/"experimento.json").write_text(json.dumps(metadata,ensure_ascii=False,indent=2),encoding="utf-8")
    print(experiment.metrics.round(4).to_string(index=False))
    print(f"Seleccionado en validación: {experiment.selected}")
    print(f"Informe exportado en {output.resolve()}")


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,default=Path("resultados"))
    export_report(parser.parse_args().output)
