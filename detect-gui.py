import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
from ultralytics import YOLO


class DetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Detector UI")
        self.root.geometry("800x600")

        self.model_path = None
        self.img_path = None
        self.model = None

        # Buttons
        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)

        tk.Button(btn_frame, text="Select Model (.pt)", command=self.select_model).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Select Image", command=self.select_image).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Run Detection", command=self.run_detection).grid(row=0, column=2, padx=5)

        # Canvas for image
        self.canvas = tk.Canvas(root, width=640, height=480, bg="grey")
        self.canvas.pack(pady=10)

    def select_model(self):
        path = filedialog.askopenfilename(filetypes=[("Model files", "*.pt")])
        if path:
            self.model_path = path
            try:
                self.model = YOLO(self.model_path)
                messagebox.showinfo("Model Loaded", f"Loaded model:\n{self.model_path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

    def select_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
        if path:
            self.img_path = path
            img = Image.open(self.img_path)
            img.thumbnail((640, 480))
            self.tk_img = ImageTk.PhotoImage(img)
            self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)

    def run_detection(self):
        if not self.model:
            messagebox.showwarning("No Model", "Please select a .pt model first.")
            return
        if not self.img_path:
            messagebox.showwarning("No Image", "Please select an image first.")
            return
        # run inference
        results = self.model(self.img_path, conf=0.25)
        # get annotated image
        annotated = results[0].plot()  # returns numpy array
        # convert to PIL
        from PIL import Image
        img = Image.fromarray(annotated)
        img.thumbnail((640, 480))
        self.tk_img = ImageTk.PhotoImage(img)
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.tk_img)


if __name__ == "__main__":
    root = tk.Tk()
    app = DetectionApp(root)
    root.mainloop()
