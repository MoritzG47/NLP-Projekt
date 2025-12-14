print("Starting GUI...")
print("It might take up to a minute to start up...")
import sys
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout,
    QTabWidget, QLineEdit, QApplication, 
    QPushButton, QHBoxLayout, QLabel,
    QComboBox
)
from PyQt5.QtGui import QFont
from PyQt5.QtCore import Qt
import pandas as pd
import extraction
from language_model import LanguageModel
from Widgets.attention_lines import AttentionLinesWidget
from Widgets.token_influence import TokenInfluenceWidget
from Widgets.attention_heatmap import AttentionHeatmapWidget
from Widgets.saliency_timeline import SaliencyTimelineWidget
from Widgets.saliency_projection import SaliencyProjectionWidget
from Widgets.hidden_state_evolution import HiddenStateEvolutionWidget
from Widgets.ig_at_hs import IGatHSWidget
from Widgets.attention_rollout import RolloutWidget

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
        self.hidden_states, self.attentions, self.tokens, self.saliency = extraction.extract_all(text, self.lang_model)
        self.num_layers = len(self.attentions)
        self.num_heads = len(self.attentions[0][0])
        self.num_hs_layers = len(self.hidden_states)
        self.heads = [i for i in range(self.num_heads)]
        tokens = self.tokens
        saliency = self.saliency
        self.heads_buttons = []
        self.layer_buttons = []
        self.heads = [i for i in range(self.num_heads)]
        self.layer = 0
        self.hs_layer = 0
        self.show_attention_heatmap(self.attentions, tokens)
        self.show_attention_lines(self.attentions, tokens)
        self.show_token_influence(tokens, saliency)
        #self.show_saliency_timeline(tokens, saliency)
        self.show_saliency_projection(tokens, saliency)
        #self.show_hidden_state_evolution(tokens, self.hidden_states)
        self.show_ig_at_hs(tokens, self.hidden_states, self.hs_layer)
        #self.show_attention_rollout(self.attentions, tokens)

    def init_variables(self):
        self.hidden_states = None
        self.attentions = None
        self.tokens = None
        self.saliency = None
        self.layer = 0
        self.hs_layer = 0
        self.heads = []
        self.num_layers = 12
        self.num_heads = 12
        self.num_hs_layers = 13
        self.heads_buttons: list[list[QPushButton]] = []
        self.layer_buttons: list[list[QPushButton]] = []
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

    def add_tab(self, plot: QWidget, title: str):
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
        self.hidden_states, self.attentions, self.tokens, self.saliency = extraction.extract_all(text, self.lang_model)
        self.update_widgets()
        print("Processing complete.")   

    def update_widgets(self):
        self.update_heatmap()
        self.update_attention_lines()
        self.update_token_influence()
        #self.update_saliency_timeline()
        self.update_saliency_projection()
        #self.update_hidden_state_evolution()
        self.update_ig_at_hs(self.hs_layer)
        #self.update_attention_rollout()

    def getButtonStyle(self, active: bool):
        if active:
            return """
                    QPushButton {
                        background-color: #C9B93D;
                    }
                    QPushButton:hover {
                        background-color: #988A22;
                    }
                    QPushButton:pressed {
                        background-color: #84781E;
                    }
                    """
        else:
            return """
                    QPushButton {
                        background-color: #FFFFFF;
                    }
                    QPushButton:hover {
                        background-color: #E0E0E0;
                    }
                    QPushButton:pressed {
                        background-color: #A0A0A0;
                    }
                    """

    def update_attn_widgets(self, layer: int=None, head: int=None):
        if layer is not None:
            self.layer = layer
            for tab in self.layer_buttons:
                for i, tab_button in enumerate(tab):
                    if i == layer:
                        tab_button.setStyleSheet(self.getButtonStyle(True))

                    else:
                        tab_button.setStyleSheet(self.getButtonStyle(False))
        if head is not None:
            if head in self.heads:
                if len(self.heads) > 1:
                    self.heads.remove(head)
                    for tab in self.heads_buttons:
                        tab[head].setStyleSheet(self.getButtonStyle(False))
            else:
                self.heads.append(head)
                for tab in self.heads_buttons:
                    tab[head].setStyleSheet(self.getButtonStyle(True))
        self.update_heatmap()
        self.update_attention_lines()

    def random_text(self):
        text = self.df.sample(n=1)["premise"].values[0]
        return text

    def show_attention_heatmap(self, attn, tokens):
        attn_heads = attn[self.layer][0]
        attn = attn_heads[self.heads].mean(axis=0).detach().numpy()
        self.plot = AttentionHeatmapWidget()
        self.plot.plot_heatmap(tokens, attn)

        widget = QWidget()
        self.heatmap_layout = QHBoxLayout(widget)
        self.heatmap_layout.addWidget(self.plot, 2)
        self.heatmap_layout = self.attention_layout(self.heatmap_layout, "heatmap")

        self.tab_widgets.addTab(widget, f"Attention Heatmap")

    def update_heatmap(self):
        self.heatmap_label.setText(f"Attention Heatmap - Layer {self.layer}")
        attn_heads = self.attentions[self.layer][0]
        attn = attn_heads[self.heads].mean(axis=0).detach().numpy()
        self.plot.plot_heatmap(self.tokens, attn)

    def show_attention_lines(self, attn, tokens):
        attn_heads = attn[self.layer][0]
        attn = attn_heads[self.heads].mean(axis=0).detach().numpy()
        self.attn_line_widget = AttentionLinesWidget(tokens, attn)
        self.attn_line_mainwidget = QWidget()
        self.lines_layout = QHBoxLayout(self.attn_line_mainwidget)
        self.lines_layout.addWidget(self.attn_line_widget, 2)
        _ = self.attention_layout(self.lines_layout, "lines")
        self.tab_widgets.addTab(self.attn_line_mainwidget, "Attention Lines")

    def update_attention_lines(self):
        old_widget = self.attn_line_widget
        self.attentionline_label.setText(f"Attention Lines - Layer {self.layer}")
        attn_heads = self.attentions[self.layer][0]
        attn = attn_heads[self.heads].mean(axis=0).detach().numpy()
        tokens = self.tokens
        self.attn_line_widget.set_attention(tokens, attn)

    def attention_layout(self, layout: QHBoxLayout, type: str) -> QHBoxLayout:
        side_layout = QVBoxLayout()
        side_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        if type == "heatmap":
            self.heatmap_label = QLabel(f"Attention Heatmap - Layer {self.layer}")
            side_layout.addWidget(self.heatmap_label)
        elif type == "lines":
            self.attentionline_label = QLabel(f"Attention Lines - Layer {self.layer}")
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
            buttonlist = []
            for i in range(count):
                button = QPushButton(f"{i}")
                if name == "Heads" and i in self.heads:
                    button.setStyleSheet(self.getButtonStyle(True))
                elif name == "Layers" and i == self.layer:
                    button.setStyleSheet(self.getButtonStyle(True))
                button.clicked.connect(lambda _, x=i: self.update_attn_widgets(layer=x if name=="Layers" else None, head=x if name=="Heads" else None))
                button_layout.addWidget(button)
                buttonlist.append(button)
            sidebutton_layout.addLayout(button_layout)
            return buttonlist

        self.layer_buttons.append(add_buttons("Layers", self.num_layers))
        self.heads_buttons.append(add_buttons("Heads", self.num_heads))
        side_layout.addLayout(sidebutton_layout)
        layout.addLayout(side_layout, stretch=1)
        return layout

    def show_token_influence(self, tokens, saliency):
        saliency = saliency.sum(dim=-1).squeeze().abs()
        scores = saliency.detach().numpy()
        self.token_influence_widget = TokenInfluenceWidget()
        self.token_influence_widget.plot_influence(tokens, scores)
        self.tab_widgets.addTab(self.token_influence_widget, "Token Influence")

    def update_token_influence(self):
        saliency = self.saliency.sum(dim=-1).squeeze().abs()
        scores = saliency.detach().numpy()
        self.token_influence_widget.plot_influence(self.tokens, scores)

    def show_saliency_timeline(self, tokens, saliency):
        self.saliency_timeline_widget = SaliencyTimelineWidget()
        saliency_matrix = saliency.squeeze(0).abs()
        saliency_matrix = saliency_matrix.detach().numpy()
        self.saliency_timeline_widget.plot_saliency_heatmap(tokens, saliency_matrix)
        self.tab_widgets.addTab(self.saliency_timeline_widget, "Saliency Timeline")

    def update_saliency_timeline(self):
        saliency_matrix = self.saliency.squeeze(0).abs()
        saliency_matrix = saliency_matrix.detach().numpy()
        self.saliency_timeline_widget.plot_saliency_heatmap(self.tokens, saliency_matrix)

    def show_saliency_projection(self, tokens, saliency):
        self.saliency_projection_widget = SaliencyProjectionWidget()
        saliency_matrix = saliency.squeeze(0).abs()
        saliency_matrix = saliency_matrix.detach().numpy()
        self.saliency_projection_widget.plot_projection(tokens, saliency_matrix)
        self.tab_widgets.addTab(self.saliency_projection_widget, "Saliency Projection")

    def update_saliency_projection(self):
        saliency_matrix = self.saliency.squeeze(0).abs()
        saliency_matrix = saliency_matrix.detach().numpy()
        self.saliency_projection_widget.plot_projection(self.tokens, saliency_matrix)

    def show_hidden_state_evolution(self, tokens, hidden_states):
        self.hidden_state_evolution_widget = HiddenStateEvolutionWidget()
        self.hidden_state_evolution_widget.plot_hidden_states(tokens, hidden_states)
        self.tab_widgets.addTab(self.hidden_state_evolution_widget, "Hidden State Evo.")

    def update_hidden_state_evolution(self):
        self.hidden_state_evolution_widget.plot_hidden_states(self.tokens, self.hidden_states)

    def show_ig_at_hs(self, tokens, hidden_states, layer_index):
        self.ig_at_hs_widget = QWidget()
        layout = QVBoxLayout(self.ig_at_hs_widget)
        self.ig_at_hs_plot = IGatHSWidget()
        self.ig_at_hs_plot.plot_ig_barplot(tokens, hidden_states, layer_index)

        layer_combo = QComboBox()
        layer_combo.setFont(self.font())
        layer_combo.addItems([f"Hidden Layer: {i}" for i in range(self.num_hs_layers)])
        layer_combo.setCurrentIndex(layer_index)
        layer_combo.currentIndexChanged.connect(lambda index: self.update_ig_at_hs(index))
        layout.addWidget(layer_combo)
        layout.addWidget(self.ig_at_hs_plot)
        self.tab_widgets.addTab(self.ig_at_hs_widget, "IG at Hidden States")

    def update_ig_at_hs(self, layer_index):
        self.ig_at_hs_plot.plot_ig_barplot(self.tokens, self.hidden_states, layer_index)

    def show_attention_rollout(self, attentions, tokens):
        self.rollout_widget = RolloutWidget()
        rollouts = self.rollout_widget.calc_attention_rollout_per_layer(attentions)
        self.rollout_widget.plot_rollout_animation(rollouts, tokens)
        self.tab_widgets.addTab(self.rollout_widget, "Attention Rollout")

    def update_attention_rollout(self):
        rollouts = self.rollout_widget.calc_attention_rollout_per_layer(self.attentions)
        self.rollout_widget.plot_rollout_animation(rollouts, self.tokens)

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = GraphsInterface()

    window.show()
    app.exec()