from constants import device
from cnn_model import CNNDigitModel
import tkinter as tk
from tkinter import messagebox
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
import argparse
from mlp_model import DigitModel

MODELS_DIR = Path("./models")
BAR_MAX_WIDTH = 120
BAR_HEIGHT = 10
HIGHLIGHT_BG = "#1a3a5c"
DEFAULT_BG = "#f0f0f0"

def latest_checkpoint(models_dir=MODELS_DIR):
    checkpoints = sorted(models_dir.glob("model_*.pth"))
    return checkpoints[-1] if checkpoints else None
  
def get_model(model_type):
  if model_type == 'cnn':
    return {
      'model':CNNDigitModel,
      'path':Path("./models_cnn"),
      'shape':(1,28,28)
    }
    
  return {
    'model':DigitModel,
    'path':Path("./models"),
    'shape':(28*28,)
  }


class MNISTDrawingApp:
    def __init__(self, root,model_type):
        self.root = root
        self.root.title("28x28 Grayscale Painter")
        self.root.geometry("560x420")
        self.root.resizable(False, False)
        self.model_type = model_type
        self.modal_dict = get_model(model_type)

        # Configuration: Black background, White pen
        self.bg_color = "black"
        self.pen_color = "white"
        self.pen_size = 16  # Thicker pen helps when downscaling to 28x28
        self.old_x = None
        self.old_y = None

        # Display canvas size (comfortably scaled up by 10x)
        self.canvas_size = 280

        # Background PIL Image to track drawing pixels accurately
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)

        self.checkpoint_path = None
        self.model = self.load_model()

        # Toolbar Frame
        self.toolbar = tk.Frame(self.root, pady=10)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)

        self.save_btn = tk.Button(self.toolbar, text="Export 28x28", command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=20)

        self.clear_btn = tk.Button(self.toolbar, text="Clear Canvas", command=self.clear_canvas)
        self.clear_btn.pack(side=tk.RIGHT, padx=20)

        content = tk.Frame(self.root)
        content.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Canvas Widget
        self.canvas = tk.Canvas(
            content,
            width=self.canvas_size,
            height=self.canvas_size,
            bg=self.bg_color,
            highlightthickness=0,
        )
        self.canvas.pack(side=tk.LEFT, padx=(10, 10), pady=10)

        self.build_side_panel(content)

        # Mouse Event Bindings
        self.canvas.bind("<B1-Motion>", self.paint)
        self.canvas.bind("<ButtonRelease-1>", self.reset)

        self.update_predictions(empty=True)

    def load_model(self):
        checkpoint = latest_checkpoint(self.modal_dict['path'])
        if checkpoint is None:
            messagebox.showerror(
                "No model found",
                f"No checkpoints found in '{MODELS_DIR}'. Train a model first.",
            )
            return None

        model = self.modal_dict['model']().to(device=device)
        model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
        model.eval()
        self.checkpoint_path = checkpoint
        return model

    def build_side_panel(self, parent):
        self.panel = tk.Frame(parent, width=220, padx=8)
        self.panel.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        self.panel.pack_propagate(False)

        tk.Label(self.panel, text="Prediction", font=("Helvetica", 12, "bold")).pack(anchor=tk.W)

        self.prediction_label = tk.Label(
            self.panel,
            text="Draw a digit",
            font=("Helvetica", 18, "bold"),
            fg="#0a5c2e",
        )
        self.prediction_label.pack(anchor=tk.W, pady=(0, 8))

        self.digit_rows = []
        for digit in range(10):
            row = tk.Frame(self.panel, bg=DEFAULT_BG)
            row.pack(fill=tk.X, pady=1)

            digit_label = tk.Label(row, text=str(digit), width=2, anchor=tk.W, bg=DEFAULT_BG)
            digit_label.pack(side=tk.LEFT)

            bar_canvas = tk.Canvas(
                row,
                width=BAR_MAX_WIDTH,
                height=BAR_HEIGHT,
                bg="#d9d9d9",
                highlightthickness=0,
            )
            bar_canvas.pack(side=tk.LEFT, padx=4)
            bar_rect = bar_canvas.create_rectangle(0, 0, 0, BAR_HEIGHT, fill="#2b6cb0", width=0)

            pct_label = tk.Label(row, text="0.0%", width=6, anchor=tk.E, bg=DEFAULT_BG)
            pct_label.pack(side=tk.LEFT)

            self.digit_rows.append((row, digit_label, bar_canvas, bar_rect, pct_label))

        checkpoint_name = self.checkpoint_path.name if self.checkpoint_path else "none"
        self.checkpoint_label = tk.Label(
            self.panel,
            text=checkpoint_name + " " + self.model_type,
            font=("Helvetica", 8),
            fg="#666666",
            wraplength=200,
            justify=tk.LEFT,
        )
        self.checkpoint_label.pack(side=tk.BOTTOM, anchor=tk.W, pady=(12, 0))

    def paint(self, event):
        if self.old_x and self.old_y:
            # 1. Render on visible Tkinter screen canvas
            self.canvas.create_line(
                self.old_x, self.old_y, event.x, event.y,
                width=self.pen_size, fill=self.pen_color,
                capstyle=tk.ROUND, smooth=True
            )
            # 2. Replicate the identical stroke on the background PIL image
            self.draw.line(
                [self.old_x, self.old_y, event.x, event.y],
                fill="white", width=self.pen_size, joint="round"
            )
            self.update_predictions()
        self.old_x = event.x
        self.old_y = event.y

    def reset(self, event):
        self.old_x = None
        self.old_y = None
        self.update_predictions()

    def clear_canvas(self):
        self.canvas.delete("all")
        self.image = Image.new("L", (self.canvas_size, self.canvas_size), "black")
        self.draw = ImageDraw.Draw(self.image)
        self.update_predictions(empty=True)

    def canvas_is_empty(self):
        return self.image.getextrema()[1] == 0

    def update_predictions(self, empty=False):
        if empty or self.canvas_is_empty() or self.model is None:
            self.prediction_label.config(text="Draw a digit")
            self.render_probabilities([0.0] * 10, predicted=None)
            return

        resized = self.image.resize((28, 28), Image.Resampling.LANCZOS)
        x = torch.tensor(np.array(resized).reshape(self.modal_dict['shape']), dtype=torch.float32) / 255
        x = x.unsqueeze(0).to(device=device)

        with torch.no_grad():
            logits, _ = self.model(x)
            probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

        predicted = int(probs.argmax())
        confidence = float(probs[predicted]) * 100
        self.prediction_label.config(text=f"{predicted}  ({confidence:.1f}%)")
        self.render_probabilities(probs, predicted=predicted)

    def render_probabilities(self, probs, predicted=None):
        for digit, (row, digit_label, bar_canvas, bar_rect, pct_label) in enumerate(self.digit_rows):
            bg = HIGHLIGHT_BG if digit == predicted else DEFAULT_BG
            fg = "white" if digit == predicted else "black"
            row.config(bg=bg)
            digit_label.config(bg=bg, fg=fg)
            pct_label.config(bg=bg, fg=fg)

            width = max(0, int(float(probs[digit]) * BAR_MAX_WIDTH))
            bar_canvas.coords(bar_rect, 0, 0, width, BAR_HEIGHT)
            pct_label.config(text=f"{float(probs[digit]) * 100:.1f}%")

    def save_image(self):
        # Resize high-resolution canvas tracking image to target 28x28 pixels
        resized_image = self.image.resize((28, 28), Image.Resampling.LANCZOS)

        # Save output file
        filename = "drawing_28x28.png"
        resized_image.save(filename)
        messagebox.showinfo("Success", f"Image exported successfully as '{filename}'!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'mlp'])
    args = parser.parse_args()
    model_type = args.model
    root = tk.Tk()
    app = MNISTDrawingApp(root,model_type)
    root.mainloop()
