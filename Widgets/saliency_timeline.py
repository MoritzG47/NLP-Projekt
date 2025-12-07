import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class SaliencyTimelineWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 7)) 
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_saliency_heatmap(self, tokens, saliency_matrix):
        seq_len, hidden_size = saliency_matrix.shape

        self.figure.clear()
        ax = self.figure.add_subplot(111)

        im = ax.imshow(saliency_matrix, aspect='auto', cmap='viridis')
        self.figure.colorbar(im, ax=ax, label="Saliency")

        ax.set_xticks(np.linspace(0, hidden_size - 1, 8, dtype=int))
        ax.set_xticklabels([str(d) for d in np.linspace(0, hidden_size - 1, 8, dtype=int)])
        ax.set_yticks(np.arange(seq_len))
        ax.set_yticklabels(tokens)

        ax.set_xlabel("Embedding Dimension")
        ax.set_ylabel("Tokens")
        ax.set_title("Saliency Heatmap Timeline")

        self.figure.tight_layout()
        self.canvas.draw()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    
    import torch
    tokens = ["This", "movie", "was", "absolutely", "fantastic", "!"]
    saliency = torch.rand(len(tokens), 768)  # dummy saliency

    widget = SaliencyTimelineWidget()
    widget.plot_saliency_heatmap(tokens, saliency)
    widget.show()


    sys.exit(app.exec_())