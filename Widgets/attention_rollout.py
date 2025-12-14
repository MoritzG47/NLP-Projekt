import sys
from matplotlib import colors
import torch
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout
from PyQt5.QtCore import QTimer
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure

class RolloutWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.figure = Figure(figsize=(5, 5)) 
        self.canvas = FigureCanvas(self.figure)
        self.toolbar = NavigationToolbar(self.canvas, self)
        self.current_layer = 0

        layout = QVBoxLayout()
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas)
        self.setLayout(layout)

    def plot_rollout_animation(self, rollouts, tokens):
        num_layers = len(rollouts)
        self.figure.clear()
        self.ax = self.figure.add_subplot(111)
        self.im = self.ax.imshow(rollouts[0], cmap="viridis")
        self.figure.colorbar(self.im, ax=self.ax, label="Attention Strength")
        
        self.ax.set_title(f"Rollout Layer 1")
        self.ax.set_xticks(range(len(tokens)))
        self.ax.set_yticks(range(len(tokens)))
        self.ax.set_xticklabels(tokens, rotation=45)
        self.ax.set_yticklabels(tokens)

        self.canvas.draw()

        self.timer = QTimer()
        self.timer.timeout.connect(lambda: self.update_frame(num_layers, rollouts))
        interval = 700
        self.timer.start(interval)

    def calc_attention_rollout_per_layer(self, attentions, alpha=0.9):
        attn_layers = [a.mean(dim=1) for a in attentions]
        attn_layers = [a[0] for a in attn_layers]

        # Residual connection + normalize
        attn_layers = [
            alpha * a + (1 - alpha) * torch.eye(a.size(-1))
            for a in attn_layers
        ]
        attn_layers = [a / a.sum(dim=-1, keepdim=True) for a in attn_layers]

        rollout_matrices = []
        current_rollout = attn_layers[0]
        rollout_matrices.append(current_rollout.detach().numpy())

        for i in range(1, len(attn_layers)):
            current_rollout = attn_layers[i] @ current_rollout
            rollout_matrices.append(current_rollout.detach().numpy())

        return rollout_matrices

    def update_frame(self, num_layers, rollouts):
        frame = rollouts[self.current_layer]
        self.im.set_data(frame)
    
        # Update color normalization dynamically
        self.im.set_norm(colors.Normalize(
            vmin=frame.min(),
            vmax=frame.max()
        ))

        self.ax.set_title(f"Rollout Layer {self.current_layer + 1}")
        self.canvas.draw()
        self.current_layer = (self.current_layer + 1) % num_layers
