import os
import sys
import tkinter as tk
from tkinter import messagebox

import numpy as np
import tensorflow as tf


GRID_SIZE = 28
CELL_SIZE = 14
BRUSH_RADIUS = 0
MNIST_DIGIT_SIZE = 20
MODEL_FILE = "model.keras"
BG_COLOR = "#000000"
GRID_COLOR = "#181818"
TEXT_COLOR = "#f5f5f5"
BUTTON_BG = "#151515"
BUTTON_ACTIVE = "#2a2a2a"


class DigitDrawer:
    def __init__(self, root, model_path):
        self.root = root
        self.root.title("")
        self.root.configure(bg=BG_COLOR)
        self.model = tf.keras.models.load_model(model_path)
        self.pixels = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
        self.last_cell = None

        canvas_size = GRID_SIZE * CELL_SIZE
        self.canvas = tk.Canvas(
            root,
            width=canvas_size,
            height=canvas_size,
            bg=BG_COLOR,
            highlightthickness=0,
        )
        self.canvas.pack(padx=18, pady=(18, 12))

        self.result = tk.StringVar(value="?")
        result_box = tk.Label(
            root,
            textvariable=self.result,
            width=4,
            height=1,
            bg=BG_COLOR,
            fg=TEXT_COLOR,
            relief="flat",
            font=("Segoe UI", 42, "bold"),
        )
        result_box.pack(padx=18, pady=(0, 10))

        buttons = tk.Frame(root, bg=BG_COLOR)
        buttons.pack(pady=(0, 18))

        predict_button = self.create_button(buttons, "Predict", self.predict)
        predict_button.pack(side=tk.LEFT, padx=6)

        clear_button = self.create_button(buttons, "Clear", self.clear)
        clear_button.pack(side=tk.LEFT, padx=6)

        self.canvas.bind("<B1-Motion>", self.draw)
        self.canvas.bind("<Button-1>", self.draw)
        self.canvas.bind("<ButtonRelease-1>", self.stop_drawing)

        self.draw_grid()

    def create_button(self, parent, text, command):
        return tk.Button(
            parent,
            text=text,
            command=command,
            bg=BUTTON_BG,
            fg=TEXT_COLOR,
            activebackground=BUTTON_ACTIVE,
            activeforeground=TEXT_COLOR,
            relief="flat",
            bd=0,
            padx=18,
            pady=8,
            font=("Segoe UI", 10, "bold"),
        )

    def draw_grid(self):
        canvas_size = GRID_SIZE * CELL_SIZE
        for i in range(GRID_SIZE + 1):
            pos = i * CELL_SIZE
            self.canvas.create_line(pos, 0, pos, canvas_size, fill=GRID_COLOR)
            self.canvas.create_line(0, pos, canvas_size, pos, fill=GRID_COLOR)

    def draw(self, event):
        col = event.x // CELL_SIZE
        row = event.y // CELL_SIZE

        if self.last_cell is None:
            self.paint_cell(row, col)
        else:
            last_row, last_col = self.last_cell
            steps = max(abs(row - last_row), abs(col - last_col), 1)
            for i in range(steps + 1):
                y = round(last_row + (row - last_row) * i / steps)
                x = round(last_col + (col - last_col) * i / steps)
                self.paint_cell(y, x)

        self.last_cell = (row, col)

    def stop_drawing(self, _event):
        self.last_cell = None

    def paint_cell(self, row, col):
        for y in range(row - BRUSH_RADIUS, row + BRUSH_RADIUS + 1):
            for x in range(col - BRUSH_RADIUS, col + BRUSH_RADIUS + 1):
                if 0 <= x < GRID_SIZE and 0 <= y < GRID_SIZE:
                    self.pixels[y, x] = 1.0
                    shade = int(self.pixels[y, x] * 255)
                    color = f"#{shade:02x}{shade:02x}{shade:02x}"
                    self.canvas.create_rectangle(
                        x * CELL_SIZE + 1,
                        y * CELL_SIZE + 1,
                        (x + 1) * CELL_SIZE - 1,
                        (y + 1) * CELL_SIZE - 1,
                        fill=color,
                        outline=color,
                    )

    def predict(self):
        if not np.any(self.pixels):
            self.result.set("?")
            return

        image = self.prepare_image().reshape(1, GRID_SIZE, GRID_SIZE, 1)
        prediction = self.model.predict(image, verbose=0)[0]
        digit = int(np.argmax(prediction))
        self.result.set(str(digit))

    def prepare_image(self):
        rows, cols = np.where(self.pixels > 0)
        if rows.size == 0:
            return self.pixels

        top, bottom = rows.min(), rows.max()
        left, right = cols.min(), cols.max()
        digit = self.pixels[top : bottom + 1, left : right + 1]

        height, width = digit.shape
        if height > width:
            new_height = MNIST_DIGIT_SIZE
            new_width = max(1, round(width * MNIST_DIGIT_SIZE / height))
        else:
            new_width = MNIST_DIGIT_SIZE
            new_height = max(1, round(height * MNIST_DIGIT_SIZE / width))

        digit = tf.image.resize(
            digit[..., np.newaxis],
            (new_height, new_width),
            method="bilinear",
        ).numpy()[:, :, 0]

        centered = np.zeros_like(self.pixels)
        y = (GRID_SIZE - digit.shape[0]) // 2
        x = (GRID_SIZE - digit.shape[1]) // 2
        centered[y : y + digit.shape[0], x : x + digit.shape[1]] = digit
        return self.center_by_mass(centered)

    def center_by_mass(self, image):
        rows, cols = np.where(image > 0)
        if rows.size == 0:
            return image

        weights = image[rows, cols]
        center_y = np.average(rows, weights=weights)
        center_x = np.average(cols, weights=weights)
        shift_y = int(round((GRID_SIZE - 1) / 2 - center_y))
        shift_x = int(round((GRID_SIZE - 1) / 2 - center_x))
        shifted = np.zeros_like(image)

        src_y_start = max(0, -shift_y)
        src_y_end = min(GRID_SIZE, GRID_SIZE - shift_y)
        src_x_start = max(0, -shift_x)
        src_x_end = min(GRID_SIZE, GRID_SIZE - shift_x)

        dst_y_start = max(0, shift_y)
        dst_y_end = dst_y_start + (src_y_end - src_y_start)
        dst_x_start = max(0, shift_x)
        dst_x_end = dst_x_start + (src_x_end - src_x_start)

        shifted[dst_y_start:dst_y_end, dst_x_start:dst_x_end] = image[
            src_y_start:src_y_end, src_x_start:src_x_end
        ]
        return shifted

    def clear(self):
        self.pixels.fill(0)
        self.last_cell = None
        self.canvas.delete("all")
        self.draw_grid()
        self.result.set("?")


def main():
    model_path = sys.argv[1] if len(sys.argv) == 2 else MODEL_FILE
    if not os.path.exists(model_path):
        messagebox.showerror("Missing model", f"Could not find {model_path}")
        return

    root = tk.Tk()
    DigitDrawer(root, model_path)
    root.mainloop()


if __name__ == "__main__":
    main()
