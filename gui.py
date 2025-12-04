import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QTabWidget, QLineEdit, QApplication, 
    QPushButton, QHBoxLayout, QLabel, QTextEdit,
    QComboBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt, QRectF
from pyqtgraph import PlotWidget
import pyqtgraph as pg
import numpy as np
import pandas as pd
import extraction
import attention_lines
from language_model import LanguageModel
from token_influence import token_influence_widget
"""
To Do:

"""

class GraphsInterface(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Graphs")
        self.center_window()
        self.setFont(QFont("Consolas"))
        self.init_stylesheet()
        self.init_variables()

        # Load dataset for random text generation https://huggingface.co/datasets/nyu-mll/multi_nli
        # Edited to be smaller for faster loading
        self.df = pd.read_parquet(r"datasets\filtered.parquet")

        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        self.main_layout = QVBoxLayout(main_widget)
        self.init_text_input()
        self.tab_widgets = QTabWidget()
        self.main_layout.addWidget(self.tab_widgets)

        self.init_model()

    def init_stylesheet(self):
        bg_color = "#D2D2D2"
        border_color = "#05257A"
        border_size = 2
        self.setStyleSheet(f"""
            QMainWindow {{
                background-color: {bg_color};
            }}
            QPushButton {{
                background-color: #FFFFFF;
                text-align: center;
                text-decoration: none;
                font-size: 24px;
                border: {border_size}px solid {border_color};
                height: 40px;
                border-radius: 4px;
                font-family: Consolas;
            }}
            QPushButton:hover {{
                background-color: #E0E0E0;
            }}
            QPushButton:pressed {{
                background-color: #A0A0A0;
            }}
            QLineEdit {{
                background-color: #FFFFFF;
                border: {border_size}px solid {border_color};
                border-radius: 4px;
                font-size: 22px;
                height: 40px;
                font-family: Consolas;
            }}
            QComboBox {{
                background-color: #FFFFFF;
                border: {border_size}px solid {border_color};
                border-radius: 4px;
                padding: 5px;
                font-size: 24px;
                font-family: Consolas;
            }}
            QTabWidget::pane {{
                border: {border_size}px solid {border_color};
                border-radius: 4px;
                padding: 5px;
            }}
            QTabBar::tab {{
                font-family: Consolas;
                background: {bg_color};
                border: {border_size}px solid {border_color};
                border-radius: 4px;
                padding: 7px;
                font-size: 18px;
                width: 200px;
            }}
            QTabBar::tab:selected {{
                background: #FFFFFF;
            }}
            QTabBar::tab:hover {{
                background: #E0E0E0;
            }}
            QLabel {{
                font-family: Consolas;
                font-size: 24px;
            }}
            """)

    def init_model(self, text: str = "The rabbit quickly hopped. The turtle slowly crawled"):
        self.textbox.setText(text)
        self.textbox.setCursorPosition(0)
        self.hidden_states, self.attentions, self.tokens, self.scores = extraction.extract_all(text, self.lang_model)
        attn = self.attentions[self.layer][0][self.head].detach().numpy()
        self.num_layers = len(self.attentions)
        self.num_heads = self.attentions[0][0].__len__()
        tokens = self.tokens
        scores = self.scores
        self.show_attention_heatmap(attn, tokens)
        self.show_attention_lines(attn, tokens)
        self.show_token_influence(tokens, scores)

    def init_variables(self):
        self.hidden_states = None
        self.attentions = None
        self.tokens = None
        self.scores = None
        self.layer = 0
        self.head = 0
        self.num_layers = 12
        self.num_heads = 12
        self.lang_model = LanguageModel()
        
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
        self.textbox.setFixedWidth(int(self.width() * 0.5))
        self.textbox.returnPressed.connect(lambda: self.process_text(self.textbox.text()))
        input_row.addWidget(self.textbox)

        enter_button = QPushButton("Enter")
        enter_button.setFont(font)
        enter_button.clicked.connect(lambda: self.process_text(self.textbox.text()))
        input_row.addWidget(enter_button)

        random_button = QPushButton("Random")
        random_button.setFont(font)
        random_button.clicked.connect(lambda: self.process_text(self.random_text()))
        input_row.addWidget(random_button)

        select_model_dropdown = QComboBox()
        select_model_dropdown.setFont(font)
        select_model_dropdown.addItems(["distilbert-base-uncased", "roberta-base", "bert-base-uncased"])
        select_model_dropdown.currentIndexChanged.connect(lambda: self.update_model(select_model_dropdown.currentText()))
        input_row.addWidget(select_model_dropdown)

        self.main_layout.addLayout(input_row)

    def update_model(self, model_name: str):
        current_tab = self.tab_widgets.currentIndex()
        self.lang_model.change_model(model_name)
        self.process_text(self.textbox.text())
        self.tab_widgets.clear()
        self.init_model(self.textbox.text())
        self.tab_widgets.setCurrentIndex(current_tab)

    def process_text(self, text: str):
        self.textbox.setText(text)
        self.textbox.setCursorPosition(0)
        print(f"Processing text: {text}")
        self.hidden_states, self.attentions, self.tokens, self.scores = extraction.extract_all(text, self.lang_model)
        self.update_widgets()
        print("Processing complete.")   

    def update_widgets(self, layer: int=None, head: int=None):
        self.layer = layer if layer is not None else self.layer
        self.head = head if head is not None else self.head
        self.update_heatmap()
        self.update_attention_lines()
        self.update_token_influence()

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

    def show_attention_heatmap(self, attn, tokens):
        self.plot = self.get_attention_heatmap(attn, tokens)

        widget = QWidget()
        self.heatmap_layout = QHBoxLayout(widget)
        self.heatmap_layout.addWidget(self.plot, 2)
        self.heatmap_layout = self.attention_layout(self.heatmap_layout, "heatmap")

        self.tab_widgets.addTab(widget, f"Attention Heatmap")

    def update_heatmap(self):
        old_widget = self.plot
        self.heatmap_label.setText(f"Attention Heatmap - Layer {self.layer} Head {self.head}")
        attn = self.attentions[self.layer][0][self.head].detach().numpy()
        tokens = self.tokens
        self.plot = self.get_attention_heatmap(attn, tokens)

        self.heatmap_layout.replaceWidget(old_widget, self.plot)

        old_widget.setParent(None)
        old_widget.deleteLater()

    def show_attention_lines(self, attn, tokens):
        self.attn_line_widget = attention_lines.AttentionLinesWidget(tokens, attn)
        self.attn_line_mainwidget = QWidget()
        self.lines_layout = QHBoxLayout(self.attn_line_mainwidget)
        self.lines_layout.addWidget(self.attn_line_widget, 2)
        _ = self.attention_layout(self.lines_layout, "lines")
        self.tab_widgets.addTab(self.attn_line_mainwidget, "Attention Lines")

    def update_attention_lines(self):
        old_widget = self.attn_line_widget
        self.attentionline_label.setText(f"Attention Lines - Layer {self.layer} Head {self.head}")
        attn = self.attentions[self.layer][0][self.head].detach().numpy()
        tokens = self.tokens
        self.attn_line_widget.set_attention(tokens, attn)

    def attention_layout(self, layout: QHBoxLayout, type: str) -> QHBoxLayout:
        side_layout = QVBoxLayout()
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        if type == "heatmap":
            self.heatmap_label = QLabel(f"Attention Heatmap - Layer {self.layer} Head {self.head}")
            heatmap_legend = QLabel("Color Legend: Low (Purple) to High (Yellow)")
            side_layout.addWidget(self.heatmap_label)
            side_layout.addWidget(heatmap_legend)
        elif type == "lines":
            self.attentionline_label = QLabel(f"Attention Lines - Layer {self.layer} Head {self.head}")
            attentionline_legend = QLabel("Hover over tokens to see attention lines.\nLine opacity indicates attention weight.")
            side_layout.addWidget(self.attentionline_label)
            side_layout.addWidget(attentionline_legend)
        sidebutton_layout = QHBoxLayout()
        sidebutton_layout.setAlignment(Qt.AlignmentFlag.AlignTop)

        def add_buttons(name: str, count: int):
            button_layout = QVBoxLayout()
            button_layout.setSpacing(10)
            button_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
            label = QLabel(name)
            button_layout.addWidget(label)
            for i in range(count):
                button = QPushButton(f"{i}")
                button.clicked.connect(lambda _, x=i: self.update_widgets(layer=x if name=="Layers" else None, head=x if name=="Heads" else None))
                button_layout.addWidget(button)
            sidebutton_layout.addLayout(button_layout)

        add_buttons("Layers", self.num_layers)
        add_buttons("Heads", self.num_heads)
        side_layout.addStretch()
        side_layout.addLayout(sidebutton_layout)
        layout.addLayout(side_layout, stretch=1)
        return layout

    def show_token_influence(self, tokens, scores):
        self.token_influence_widget = token_influence_widget()
        self.token_influence_widget.plot_influence(tokens, scores)
        self.tab_widgets.addTab(self.token_influence_widget, "Token Influence")

    def update_token_influence(self):
        self.token_influence_widget.plot_influence(self.tokens, self.scores)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = GraphsInterface()

    window.show()
    app.exec()