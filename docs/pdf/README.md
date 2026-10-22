# Exportación de Inicio y Exploración

Los PDF contienen capturas del navegador de los componentes originales de Streamlit;
no se reconstruyen tablas o gráficos con otra biblioteca. Cada tabla se conserva
como un componente independiente. Las tablas anchas/largas se capturan en partes
mediante sus desplazamientos reales para incluir columnas y filas ocultas.

Exploración: «Datos tal como se recibieron», sin filtros, todos los contratos y
servicios de internet e inclusión de contratos ausentes. Se incluyen las cuatro
subpestañas, las tablas de frecuencia de todas las variables categóricas, calidad
de datos y las tres opciones de KPIs complementarios. El ejemplo de cliente con
conflictos conserva la primera opción del selector; no se enumeran todos los
estados posibles de ese selector ni todas las combinaciones de filtros.

Inicio: objetivo general, indicadores, inventario, integración, diccionario y las
vistas previas de 30 filas de la integración y cada uno de los tres archivos.
Los PDF son fotografías de estados de la aplicación; no sustituyen los CSV completos.

El manifiesto registra fecha, hashes de los datos, versión de app, estado y capturas.
Los originales y el documento académico DOCX no se modifican.

Reproducción local Linux (requiere Chromium y DejaVu Sans):

```bash
.venv/bin/python -m pip install -r requirements-export.txt
.venv/bin/python -m playwright install chromium
.venv/bin/python scripts/export_pages_pdf.py
```

El script inicia un servidor temporal en localhost:8531 y lo detiene al finalizar.
Los PDF están compuestos por capturas de pantalla; el cuerpo capturado no es texto
seleccionable. Los rótulos y pies de página se añaden al PDF para identificar cada vista.
