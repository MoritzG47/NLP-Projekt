import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QTabWidget, QLineEdit, QApplication, 
    QPushButton, QHBoxLayout
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt
from pyqtgraph import PlotWidget
import numpy as np

"""
To Do:
Add Loading Indicator

"""

class GraphsInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graphs")
        self.center_window()
        self.setFont(QFont("Consolas"))

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        self.main_layout = QVBoxLayout(main_widget)
        self.init_text_input()

        self.tab_widgets = QTabWidget()
        self.main_layout.addWidget(self.tab_widgets)

    def center_window(self):
        """Written by ChatGPT to center the window on the screen."""
        self.screen_dimensions = QApplication.primaryScreen().availableGeometry()
        width, height = int(self.screen_dimensions.width()*0.7), int(self.screen_dimensions.height()*0.7)
        self.resize(width, height)
        frame_geometry = self.frameGeometry()
        center_point = self.screen_dimensions.center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

    def add_tab(self, plot: PlotWidget, title: str):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(plot)
        self.tab_widgets.addTab(widget, title)

    def init_text_input(self):
        input_row = QHBoxLayout()
        input_row.setAlignment(Qt.AlignmentFlag.AlignLeft)
        input_row.setSpacing(2)

        textbox = QLineEdit()
        font = self.font()
        font.setPointSize(12)
        textbox.setFont(font)
        textbox.setPlaceholderText("Text Input for the Model")
        textbox.setFixedHeight(25)
        textbox.setFixedWidth(int(self.width() * 0.5))
        textbox.returnPressed.connect(lambda: self.process_text(textbox.text()))
        input_row.addWidget(textbox)

        enter_button = QPushButton("Enter")
        enter_button.setFont(font)
        enter_button.setFixedHeight(25)
        enter_button.setFixedWidth(100)
        enter_button.clicked.connect(lambda: self.process_text(textbox.text()))
        input_row.addWidget(enter_button)

        random_button = QPushButton("Random")
        random_button.setFont(font)
        random_button.setFixedHeight(25)
        random_button.setFixedWidth(100)
        random_button.clicked.connect(lambda: self.process_text(self.random_text()))
        input_row.addWidget(random_button)

        self.main_layout.addLayout(input_row)

    def process_text(self, text: str):
        print(f"Processing text: {text}")

    def random_text(self):
        pass

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = GraphsInterface()

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

