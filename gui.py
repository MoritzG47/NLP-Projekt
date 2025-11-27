import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QTabWidget, QMainWindow
from pyqtgraph import PlotWidget
import numpy as np

class GraphsInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graphs")

        self.tab_widgets = QTabWidget()
        self.setCentralWidget(self.tab_widgets)

    def add_tab(self, plot: PlotWidget, title: str):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(plot)
        self.tab_widgets.addTab(widget, title)

app = QApplication(sys.argv)
window = GraphsInterface()

if __name__ == "__main__":
    plot = PlotWidget()
    x = np.linspace(0, 10, 500)
    y = np.sin(x)
    plot.plot(x, y)
    window.add_tab(plot, "Sine Wave")

    plot = PlotWidget()
    x = np.random.normal(size=100)
    y = np.random.normal(size=100)
    plot.plot(x, y, pen=None, symbol='o')
    window.add_tab(plot, "Scatter Plot")

window.show()
app.exec()
