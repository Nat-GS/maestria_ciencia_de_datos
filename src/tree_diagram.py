"""Diagrama del árbol real con reglas y conteos observados de entrenamiento."""
import json
from pathlib import Path

import numpy as np

from src.data_processing import CATEGORIES
from src.exploration import LABELS, display_value
from src.interpretation import number


def tree_download_html():
    """Botón local que rasteriza el SVG realmente mostrado en Streamlit."""
    return Path(__file__).with_name('tree_download.html').read_text(encoding='utf-8')


def _feature_label(name):
    if name in LABELS:
        return LABELS[name]
    for column in CATEGORIES:
        if name.startswith(column + "_"):
            return f"{LABELS[column]}: {display_value(name[len(column)+1:])}"
    return name


def decision_tree_diagram(model, training, target, max_depth=3):
    """Genera DOT sin ajustar ni cambiar el pipeline.

    Los conteos de clases proceden del recorrido de clientes de entrenamiento,
    no de tree.value, que en sklearn puede contener proporciones ponderadas.
    La impureza, clase estimada y colores sí corresponden al estimador ajustado.
    Los nodos truncados se identifican explícitamente.
    """
    estimator = model['modelo']
    tree = estimator.tree_
    names = model['preparacion'].get_feature_names_out()
    transformed = model['preparacion'].transform(training)
    target = np.asarray(target)
    if len(target) != len(training):
        raise ValueError("Las etiquetas deben corresponder a los clientes de entrenamiento.")
    paths = estimator.decision_path(transformed)
    samples = np.asarray(paths.sum(axis=0)).ravel().astype(int)
    if not np.array_equal(samples, tree.n_node_samples):
        raise ValueError("El diagrama requiere las mismas filas utilizadas para entrenar el árbol.")
    counts = np.column_stack([np.asarray(paths[target == cls].sum(axis=0)).ravel()
                              for cls in estimator.classes_]).astype(int)
    classes = {0: 'Permanencia', 1: 'Abandono'}
    lines = ['digraph Arbol {', 'graph [rankdir=TB, bgcolor="transparent", nodesep=0.25, ranksep=0.7];',
             'node [shape=box, style="rounded,filled", fontname="Arial", fontsize=13, color="#647582", margin="0.14,0.10"];',
             'edge [fontname="Arial", fontsize=12, color="#647582"];']

    def visit(node, depth):
        leaf = tree.children_left[node] == tree.children_right[node]
        truncated = not leaf and depth >= max_depth
        weights = tree.value[node][0]
        winner = int(np.argmax(weights))
        predicted = classes[int(estimator.classes_[winner])]
        strength = float(abs(weights[0]-weights[1])/weights.sum())
        base = (35, 125, 120) if predicted == 'Abandono' else (201, 120, 85)
        blend = .15 + .6*strength
        color = '#' + ''.join(f'{round(255*(1-blend)+channel*blend):02x}' for channel in base)
        label = []
        if not leaf:
            label.append(f"{_feature_label(names[tree.feature[node]])} ≤ {number(tree.threshold[node],2)}")
        label.extend([f"Gini = {number(tree.impurity[node],3)}", f"Muestras = {number(samples[node],0)}",
                      f"Clientes = [{number(counts[node,0],0)}, {number(counts[node,1],0)}]",
                      f"Clase estimada = {predicted}"])
        if truncated:
            label.append("Continúa en niveles inferiores…")
        lines.append(f'{node} [label={json.dumps(chr(10).join(label),ensure_ascii=False)}, fillcolor="{color}"];')
        if not leaf and not truncated:
            for child, branch in [(tree.children_left[node], 'Sí'), (tree.children_right[node], 'No')]:
                lines.append(f'{node} -> {child} [label="{branch}"];')
                visit(child, depth+1)
    visit(0,0)
    lines.append('}')
    return '\n'.join(lines)
