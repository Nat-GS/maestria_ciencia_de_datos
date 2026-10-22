# Capturas del dashboard incorporadas en el documento

Las imágenes PNG son capturas reales de los componentes de Streamlit, obtenidas
con Chromium. No son recreaciones con Matplotlib ni tablas con otra agrupación.

Estado de exploración: **Datos tal como se recibieron**, sin filtros, 7.303 filas.
El panel de calidad representa la misma integración completa. Las tablas de
frecuencias corresponden al selector de cada variable en Categorías. Los anchos
de ventana varían para aprovechar el diseño responsivo y facilitar la lectura.
No se alteraron estilos del navegador para producir las capturas.

`manifest.json` registra el inventario y hashes de los tres CSV, la vista de origen,
el hash de cada captura y el documento actualizado. Se comprobó que cada imagen
incrustada en el Word coincide byte a byte con su captura. Los CSV de esta carpeta
contienen las mismas tablas y redondeo del dashboard, para comprobar los valores.

Si cambian los datos, los filtros o el diseño de la página, deben renovarse las
capturas y el Word juntos. No reemplazar estos gráficos por otros con distinto
formato ni combinar tablas que la aplicación muestra por separado.
