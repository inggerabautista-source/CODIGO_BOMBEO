# Memoria técnica de línea de bombeo (criterios MAPAS-CONAGUA)

App en Streamlit que calcula y documenta una línea de conducción por bombeo, desde la fuente
(pozo profundo o cárcamo) hasta el tanque, cisterna o tanque elevado.

## Funciones
- Caudales de diseño: población y dotación → Qmed, Qmd, Qmh con CVd y CVh editables (Qmd: captación, conducción y regulación; Qmh: alimentación a red y red)
- Croquis de la altura de descarga, del pozo/cárcamo y de tanques/cisterna con cotas rotuladas
- Perfil de tubería opcional (mismo formato que el terreno)
- Esquema general de niveles y tabla de niveles de todos los elementos (cárcamo/pozo, bomba, línea, tanque, cisterna)
- Perfil por CSV/Excel (cadenamiento y cota; acepta formato `0+500`) con vista previa gráfica
- Fricción (Darcy-Weisbach / Hazen-Williams) y pérdidas locales con K editables
- Detección de puntos altos/bajos y sugerencia de válvulas de aire y desagüe
- Diámetro iterativo con alertas de velocidad; diámetros de accesorios
- Carga dinámica total, potencia de bomba y motor comercial, clase de tubería, golpe de ariete estimado
- Pozo (NE/ND, profundidad de bomba) o cárcamo (volumen, sumergencia, NAMINO/NAMO/NAME)
- Tanque de regulación (coeficiente o curva masa), reserva y dimensiones
- Exporta Excel, CSV y memoria en Word

## Ejecutar local
```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Desplegar en Streamlit Community Cloud
1. Sube este repositorio a GitHub.
2. En https://share.streamlit.io elige "New app", selecciona el repo y la rama.
3. Main file path: `streamlit_app.py`. Deploy.

## Aviso
Los valores precargados (rugosidades, K, tiempos de ciclo, coeficiente de regulación, patrón de
demanda) son de referencia y editables. Verifícalos contra la edición vigente de los manuales
MAPAS antes de emitir una memoria oficial. El golpe de ariete es solo una estimación; se requiere
análisis de transitorios. Los resultados deben ser revisados por el proyectista responsable.
