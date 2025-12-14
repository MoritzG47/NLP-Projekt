import sys
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class TokenInfluenceWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(4, 2)) 
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_influence(self, tokens, scores):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        im = ax.imshow(scores[:, None], cmap='viridis', aspect='auto')
        self.figure.colorbar(im, ax=ax)
        ax.set_yticks(range(len(tokens)))
        ax.set_yticklabels(tokens)
        ax.set_xticks([])
        ax.set_title("Token Influence Heatmap (Gradient x Input)")

        self.figure.subplots_adjust(left=0.4, right=0.6)
        self.canvas.draw()


if __name__ == "__main__":
    tokens = ["[CLS]", "the", "rabbit", "quickly", "hopped", "[SEP]", "the", "turtle", "slowly", "crawled", "[SEP]"]
    scores = np.random.rand(len(tokens))
    app = QApplication(sys.argv)
    w = TokenInfluenceWidget()
    w.plot_influence(tokens, scores)
    w.show()
    sys.exit(app.exec_())
