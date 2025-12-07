from PyQt5.QtWidgets import QWidget, QVBoxLayout
from PyQt5.QtGui import QPainter, QColor, QPen, QFont
from PyQt5.QtCore import Qt, QPointF
from PyQt5.QtWidgets import QWidget
from PyQt5.QtGui import QPainter, QFont, QPen, QColor, QFontMetrics, QBrush
from PyQt5.QtCore import Qt, QPointF, QRectF

class AttentionLinesWidget(QWidget):
    def __init__(self, tokens, attention_matrix, parent=None):
        super().__init__(parent)
        self.tokens = tokens
        token_length = len(tokens) if tokens is not None else 8              
        self.attn = attention_matrix         
        self.left_margin = 120
        self.right_margin = 200
        self.line_height = 50
        self.token_font = QFont("Consolas", 10)
        
        self.hovered_index = -1
        self.setMouseTracking(True)

        self.left_token_rects = []
        self.right_token_rects = []
        
        self.setMinimumWidth(500)
        self.setMinimumHeight(token_length * self.line_height + 40)

    def set_attention(self, tokens, attention_matrix):
        """Allows dynamic updates."""
        self.tokens = tokens
        self.attn = attention_matrix
        self.update_token_rects()
        self.update()

    def update_token_rects(self):
        """Update token bounding rectangles for hit testing."""
        seq_len = len(self.tokens)
        width = self.width()
        
        self.left_token_rects = []
        self.right_token_rects = []
        
        fm = QFontMetrics(self.token_font)
        
        for i in range(seq_len):
            y_pos = 20 + i * self.line_height
            text_width = fm.horizontalAdvance(self.tokens[i])
            left_rect = QRectF(10, y_pos, text_width, self.line_height)
            self.left_token_rects.append(left_rect)
            
            right_x = width - self.right_margin + 10
            right_rect = QRectF(right_x, y_pos, text_width, self.line_height)
            self.right_token_rects.append(right_rect)

    def resizeEvent(self, event):
        """Update token rectangles when widget is resized."""
        super().resizeEvent(event)
        self.update_token_rects()

    def mouseMoveEvent(self, event):
        """Handle mouse hover events."""
        pos = event.pos()
        pos_f = QPointF(pos)
        old_hover = self.hovered_index
        self.hovered_index = -1
        
        for i, rect in enumerate(self.left_token_rects):
            if rect.contains(pos_f):
                self.hovered_index = i
                break

        if self.hovered_index == -1:
            for i, rect in enumerate(self.right_token_rects):
                if rect.contains(pos_f):
                    self.hovered_index = i
                    break
        
        if self.hovered_index != old_hover:
            self.update()
        if self.hovered_index != -1:
            self.setCursor(Qt.CursorShape.PointingHandCursor)
        else:
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def leaveEvent(self, event):
        """Clear hover when mouse leaves widget."""
        if self.hovered_index != -1:
            self.hovered_index = -1
            self.update()
            self.setCursor(Qt.CursorShape.ArrowCursor)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setFont(self.token_font)
        seq_len = len(self.tokens)

        # --- Compute vertical positions ---
        y_positions = [
            20 + i * self.line_height
            for i in range(seq_len)
        ]

        if not self.left_token_rects or not self.right_token_rects:
            self.update_token_rects()

        # --- Draw tokens on left ---
        for i, tok in enumerate(self.tokens):
            if i == self.hovered_index:
                painter.save()
                painter.setPen(QPen(QColor(255, 100, 100)))
                painter.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
                painter.drawText(10, y_positions[i] + 5, tok)
                painter.restore()
            else:
                painter.drawText(10, y_positions[i] + 5, tok)

        # --- Draw tokens on right ---
        width = self.width()
        for i, tok in enumerate(self.tokens):
            if i == self.hovered_index:
                painter.save()
                painter.setPen(QPen(QColor(255, 100, 100)))
                painter.setFont(QFont("Consolas", 11, QFont.Weight.Bold))
                painter.drawText(width - self.right_margin + 10, y_positions[i] + 5, tok)
                painter.restore()
            else:
                painter.drawText(width - self.right_margin + 10, y_positions[i] + 5, tok)

        # --- Draw attention lines ---
        max_attn = self.attn.max() if self.attn.max() > 0 else 1.0
        left_x = self.left_margin
        right_x = width - self.right_margin

        if self.hovered_index != -1:
            i = self.hovered_index
            for j in range(seq_len):
                weight = float(self.attn[i, j])
                if weight <= 0:
                    continue

                alpha = min(255, int(255 * (weight / max_attn)))
                pen = QPen(QColor(70, 130, 255, alpha))
                pen.setWidth(3)
                painter.setPen(pen)

                start = QPointF(left_x, y_positions[i])
                end = QPointF(right_x, y_positions[j])
                painter.drawLine(start, end)
                
                if j != i:
                    weight = float(self.attn[j, i])
                    if weight > 0:
                        alpha = min(255, int(255 * (weight / max_attn)))
                        pen = QPen(QColor(255, 130, 70, alpha))
                        pen.setWidth(3)
                        painter.setPen(pen)
                        
                        start = QPointF(left_x, y_positions[j])
                        end = QPointF(right_x, y_positions[i])
                        painter.drawLine(start, end)
        else:
            for i in range(seq_len):
                for j in range(seq_len):
                    weight = float(self.attn[i, j])
                    if weight <= 0:
                        continue
                    alpha = min(100, int(100 * (weight / max_attn)))

                    pen = QPen(QColor(70, 130, 255, alpha))
                    pen.setWidth(2)
                    painter.setPen(pen)

                    start = QPointF(left_x, y_positions[i])
                    end = QPointF(right_x, y_positions[j])
                    painter.drawLine(start, end)
        


if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication, QMainWindow
    import numpy as np
    import sys

    class MainWindow(QMainWindow):
        def __init__(self):
            super().__init__()

            tokens = ["[CLS]", "the", "rabbit", "quickly", "hopped", "[SEP]"]
            attention = np.random.rand(len(tokens), len(tokens))

            self.widget = AttentionLinesWidget(tokens, attention)
            self.setCentralWidget(self.widget)
            self.resize(700, 500)

    app = QApplication(sys.argv)
    w = MainWindow()
    w.show()
    sys.exit(app.exec())
