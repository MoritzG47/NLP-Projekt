import sys
import numpy as np
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class AttentionHeatmapWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(7, 7)) 
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_heatmap(self, tokens, attentions):
        self.figure.clear()
        ax = self.figure.add_subplot(111)

        im = ax.imshow(attentions, cmap='viridis', aspect='auto')
        self.figure.colorbar(im, ax=ax)
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=90)
        ax.set_yticklabels(tokens)
        ax.set_title("Attention Heatmap")
        self.canvas.draw()

if __name__ == "__main__":
    tokens = ["[CLS]", "the", "rabbit", "quickly", "hopped", "[SEP]", "the", "turtle", "slowly", "crawled", "[SEP]"]
    attn_matrix = np.random.rand(len(tokens), len(tokens))
    app = QApplication(sys.argv)
    w = AttentionHeatmapWidget()
    w.plot_heatmap(tokens, attn_matrix)
    w.show()
    sys.exit(app.exec_())