import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

class SaliencyProjectionWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 7))
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_projection(self, tokens, saliency_matrix):
        # Normalize
        eps = 1e-10
        norms = np.linalg.norm(saliency_matrix, axis=1, keepdims=True) + eps
        saliency_matrix = saliency_matrix / norms

        # Dimensionality reduction
        reducer = PCA(n_components=2)
        proj = reducer.fit_transform(saliency_matrix)

        x, y = proj[:, 0], proj[:, 1]
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.scatter(x, y, c=np.arange(len(tokens)), cmap="viridis", s=100)

        for i, tok in enumerate(tokens):
            ax.text(x[i]+0.01, y[i]+0.01, tok, fontsize=9)

        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title("Saliency PCA Projection")
        self.figure.tight_layout()
        self.canvas.draw()

if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication
    import torch

    app = QApplication(sys.argv)

    tokens = ["This", "movie", "was", "absolutely", "fantastic", "!"]
    saliency = torch.rand(len(tokens), 768)

    widget = SaliencyProjectionWidget()
    widget.plot_projection(tokens, saliency)
    widget.show()

    sys.exit(app.exec_())