"""Compatibilidad del generador histórico, sin depender de archivos retirados."""
import json
import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from src.integration import SOURCE_PATHS
generator=pytest.importorskip("src.generate_dirty_data", reason="El generador histórico fue retirado; no interviene en el flujo de tres CSV.")
NOTEBOOK_PATH=generator.NOTEBOOK_PATH
generate_dirty=generator.generate_dirty
generate_file=generator.generate_file
alteration_report=generator.alteration_report


def test_generator_still_matches_original_notebook_on_available_input():
    original=pd.read_csv(SOURCE_PATHS[0])
    before=original.copy(deep=True)
    notebook=json.loads(NOTEBOOK_PATH.read_text())
    source="".join(notebook["cells"][2]["source"]).split("# 3. Mostrar los resultados")[0]
    ns={"np":np,"pd":pd,"df":original}
    exec(compile(source,str(NOTEBOOK_PATH),"exec"),ns)
    dirty,manifest=generate_dirty(original)
    assert_frame_equal(dirty,ns["df_sucio"])
    assert_frame_equal(before,original)
    assert len(dirty)==len(original)+int(len(original)*.02)
    assert len(manifest["indices_duplicados"])==int(len(original)*.02)
    assert alteration_report(original,dirty)["Valores observados alterados"].sum()==0


def test_generator_protects_sources_and_is_repeatable(tmp_path):
    for p in SOURCE_PATHS:
        with pytest.raises(ValueError): generate_file(SOURCE_PATHS[0],p)
    a,b=tmp_path/"a.csv",tmp_path/"b.csv"
    generate_file(SOURCE_PATHS[0],a);generate_file(SOURCE_PATHS[0],b)
    assert a.read_bytes()==b.read_bytes()
