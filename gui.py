import sys
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QTabWidget, QLineEdit, QApplication, 
    QPushButton, QHBoxLayout, QLabel, QTextEdit
)
from PyQt6.QtGui import QFont
from PyQt6.QtCore import Qt, QRectF
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import numpy as np
import matplotlib.pyplot as plt
from typing import overload, Union
import pandas as pd
import extraction
import attention_lines


"""
To Do:
Add Loading Indicator
the rabbit quickly hopped. the turtle slowly crawled
"""

class GraphsInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graphs")
        self.center_window()
        self.setFont(QFont("Consolas"))
        self.init_variables()

        # Load dataset for random text generation https://huggingface.co/datasets/nyu-mll/multi_nli
        # Edited to be smaller for faster loading
        self.df = pd.read_parquet(r"datasets\multi_nli\filtered.parquet")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)

        self.main_layout = QVBoxLayout(main_widget)
        self.init_text_input()

        self.tab_widgets = QTabWidget()
        self.main_layout.addWidget(self.tab_widgets)

    def init_variables(self):
        self.hidden_states = None
        self.attentions = None
        self.tokens = None
        self.layer = 0
        self.head = 0

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

        self.textbox = QLineEdit()
        font = self.font()
        font.setPointSize(12)
        self.textbox.setFont(font)
        self.textbox.setPlaceholderText("Text Input for the Model")
        self.textbox.setFixedHeight(25)
        self.textbox.setFixedWidth(int(self.width() * 0.5))
        self.textbox.returnPressed.connect(lambda: self.process_text(self.textbox.text()))
        input_row.addWidget(self.textbox)

        enter_button = QPushButton("Enter")
        enter_button.setFont(font)
        enter_button.setFixedHeight(25)
        enter_button.setFixedWidth(100)
        enter_button.clicked.connect(lambda: self.process_text(self.textbox.text()))
        input_row.addWidget(enter_button)

        random_button = QPushButton("Random")
        random_button.setFont(font)
        random_button.setFixedHeight(25)
        random_button.setFixedWidth(100)
        random_button.clicked.connect(lambda: self.process_text(self.random_text()))
        input_row.addWidget(random_button)

        self.main_layout.addLayout(input_row)

    def process_text(self, text: str):
        self.textbox.setText(text)
        self.textbox.setCursorPosition(0)
        print(f"Processing text: {text}")
        self.hidden_states, self.attentions, self.tokens = extraction.extract_all(text)
        self.update_widgets()
        print("Processing complete.")   

    def update_widgets(self):
        self.update_heatmap(layer=self.layer, head=self.head)
        self.update_attention_lines(self.attentions[self.layer][0][self.head].detach().numpy(), self.tokens)

    def random_text(self):
        text = self.df.sample(n=1)["premise"].values[0]
        return text

    def get_attention_heatmap(self, attn, tokens) -> PlotWidget:
        """Heatmap base structure by ChatGPT """
        plot = PlotWidget()

        img = np.array(attn)
        img = np.flipud(img)

        img_item = pg.ImageItem(img)
        plot.addItem(img_item)

        # Virdis colormap: Purple (low) to Yellow (high)
        lut = pg.colormap.get("viridis").getLookupTable(0.0, 1.0, 256)
        img_item.setLookupTable(lut)
        img_item.setRect(QRectF(0, 0, img.shape[1], img.shape[0]))

        for i, tok in enumerate(tokens):
            plot.getAxis("bottom").setTicks([[(i, tok) for i, tok in enumerate(tokens)]])
        for i, tok in enumerate(tokens):
            plot.getAxis("left").setTicks([[(i, tokens[::-1][i]) for i in range(len(tokens))]])

        return plot

    def show_attention_heatmap(self, attn, tokens, num_layers: int, num_heads: int):
        self.plot = self.get_attention_heatmap(attn, tokens)

        widget = QWidget()
        self.heatmap_layout = QHBoxLayout(widget)

        self.heatmap_layout.addWidget(self.plot)

        side_layout = QVBoxLayout()
        self.heatmap_label = QLabel(f"Attention Heatmap - Layer {self.layer} Head {self.head}")
        self.heatmap_legend = QLabel("Color Legend: Low (Purple) to High (Yellow)")
        side_layout.addWidget(self.heatmap_label)
        side_layout.addWidget(self.heatmap_legend)

        sidebutton_layout = QHBoxLayout()

        def add_buttons(name: str, count: int):
            button_layout = QVBoxLayout()
            button_layout.setSpacing(10)
            label = QLabel(name)
            button_layout.addWidget(label)
            for i in range(count):
                button = QPushButton(f"{i}")
                button.clicked.connect(lambda _, x=i: self.update_heatmap(layer=x if name=="Layers" else None, head=x if name=="Heads" else None))
                button_layout.addWidget(button)
            sidebutton_layout.addLayout(button_layout)

        add_buttons("Layers", num_layers)
        add_buttons("Heads", num_heads)
        side_layout.addStretch()
        side_layout.addLayout(sidebutton_layout)
        self.heatmap_layout.addLayout(side_layout)
    
        self.tab_widgets.addTab(widget, f"Attention Heatmap")

    def update_heatmap(self, layer: int=None, head: int=None):
        old_widget = self.plot
        self.layer = layer if layer is not None else self.layer
        self.head = head if head is not None else self.head
        self.heatmap_label.setText(f"Attention Heatmap - Layer {self.layer} Head {self.head}")
        attn = self.attentions[self.layer][0][self.head].detach().numpy()
        tokens = self.tokens
        self.plot = self.get_attention_heatmap(attn, tokens)

        self.heatmap_layout.replaceWidget(old_widget, self.plot)

        old_widget.setParent(None)
        old_widget.deleteLater()

    def show_attention_lines(self, attn, tokens):
        self.attn_line_widget = attention_lines.AttentionLinesWidget(tokens, attn)
        self.tab_widgets.addTab(self.attn_line_widget, "Attention Lines")

    def update_attention_lines(self, attn, tokens):
        self.attn_line_widget.set_attention(tokens, attn)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = GraphsInterface()

    attn = np.random.rand(10, 10)
    tokens = ["[CLS]","Hello","world",",","this","is","a","test",".","[SEP]"]
    window.show_attention_heatmap(attn, tokens, 12, 12)
    window.show_attention_lines(attn, tokens)

    window.show()
    app.exec()