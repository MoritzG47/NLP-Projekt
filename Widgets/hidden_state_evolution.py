import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
import torch

class HiddenStateEvolutionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 7)) 
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_hidden_states(self, tokens, hidden_states):
        token_norms_per_layer = []

        for h in hidden_states:
            token_norms = torch.norm(h[0], dim=-1)
            token_norms_per_layer.append(token_norms.tolist())
        token_norms_per_layer = list(zip(*token_norms_per_layer))

        num_layers = len(hidden_states)
        seq_len = len(tokens)        
        
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        for i in range(seq_len):
            ax.plot(range(num_layers), token_norms_per_layer[i], marker="o", label=tokens[i])

        ax.grid(True)
        ax.set_xticks(range(num_layers))
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.set_title("Hidden-State Norm Evolution Per Token")
        ax.set_xlabel("Layer")
        ax.set_ylabel("L2 Norm of Hidden State")

        self.figure.tight_layout()
        self.canvas.draw()