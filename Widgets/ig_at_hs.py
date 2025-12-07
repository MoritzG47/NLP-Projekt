from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from captum.attr import IntegratedGradients
import torch

class IGatHSWidget(QWidget): # Integrated Gradients at Hidden States
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 7)) 
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_ig_barplot(self, tokens, hidden_states, layer_index):
        def forward_func(x):
            out = x.sum(dim=-1).abs().sum()
            return out.unsqueeze(0)

        ig = IntegratedGradients(forward_func)

        input_hidden = hidden_states[layer_index].detach().requires_grad_(True)

        attributions, delta = ig.attribute(
            input_hidden,
            baselines=torch.zeros_like(input_hidden),
            return_convergence_delta=True
        )

        token_attr = attributions.sum(dim=-1).squeeze().tolist()

        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.bar(range(len(tokens)), token_attr)
        ax.set_xticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45)
        ax.set_ylabel("Integrated Gradients Attribution (sum over hidden units)")
        ax.set_title(f"Integrated Gradients on Hidden States (Layer {layer_index})")

        self.figure.tight_layout()
        self.canvas.draw()