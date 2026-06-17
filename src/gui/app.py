"""
Graphical User Interface for Image Enhancing Tool.
Built with tkinter for cross-platform compatibility.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import cv2
import numpy as np
from pathlib import Path
import threading
import sys
import os

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.pipeline import EnhancementPipeline, EnhancementConfig


class ImageEnhancerGUI:
    """Main GUI application for image enhancement."""
    
    def __init__(self, root):
        self.root = root
        self.root.title("AI Image Enhancer")
        self.root.geometry("1200x750")
        self.root.minsize(900, 600)
        
        self.original_image = None
        self.enhanced_image = None
        self.current_image_path = None
        self.pipeline = EnhancementPipeline()
        
        self._setup_styles()
        self._create_widgets()
        self._create_menu()
        self._setup_layout()
    
    def _setup_styles(self):
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.style.configure('TButton', padding=6)
        self.style.configure('Success.TButton', background='#4CAF50')
        self.style.configure('Primary.TButton', background='#2196F3')
    
    def _create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Image", command=self.open_image, accelerator="Ctrl+O")
        file_menu.add_command(label="Save Enhanced", command=self.save_image, accelerator="Ctrl+S")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.root.quit)
        
        enhance_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Enhance", menu=enhance_menu)
        enhance_menu.add_command(label="Full Pipeline", command=lambda: self.enhance_image('full'))
        enhance_menu.add_command(label="Traditional Only", command=lambda: self.enhance_image('traditional'))
        enhance_menu.add_command(label="AI Only", command=lambda: self.enhance_image('ai'))
        
        view_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="View", menu=view_menu)
        view_menu.add_command(label="Original", command=self.show_original)
        view_menu.add_command(label="Enhanced", command=self.show_enhanced)
        view_menu.add_separator()
        self.view_menu_compare = tk.BooleanVar(value=True)
        view_menu.add_checkbutton(label="Side by Side", variable=self.view_menu_compare, command=self.toggle_view)
    
    def _create_widgets(self):
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding=10)
        control_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 10))
        
        btn_frame = ttk.Frame(control_frame)
        btn_frame.pack(fill=tk.X)
        
        self.open_btn = ttk.Button(btn_frame, text="📂 Open Image", command=self.open_image, style='Primary.TButton')
        self.open_btn.pack(side=tk.LEFT, padx=2)
        
        self.enhance_btn = ttk.Button(btn_frame, text="✨ Enhance", command=self.enhance_pipeline, style='Success.TButton')
        self.enhance_btn.pack(side=tk.LEFT, padx=2)
        self.enhance_btn.config(state=tk.DISABLED)
        
        self.save_btn = ttk.Button(btn_frame, text="💾 Save", command=self.save_image)
        self.save_btn.pack(side=tk.LEFT, padx=2)
        self.save_btn.config(state=tk.DISABLED)
        
        # Settings
        settings_frame = ttk.Frame(control_frame)
        settings_frame.pack(fill=tk.X, pady=(10, 0))
        
        ttk.Label(settings_frame, text="Denoise:").pack(side=tk.LEFT, padx=(0, 5))
        self.denoise_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, variable=self.denoise_var).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(settings_frame, text="Sharpen:").pack(side=tk.LEFT, padx=(0, 5))
        self.sharpen_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, variable=self.sharpen_var).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(settings_frame, text="Color:").pack(side=tk.LEFT, padx=(0, 5))
        self.color_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, variable=self.color_var).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(settings_frame, text="Contrast:").pack(side=tk.LEFT, padx=(0, 5))
        self.contrast_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, variable=self.contrast_var).pack(side=tk.LEFT, padx=(0, 15))
        
        ttk.Label(settings_frame, text="AI Enhance:").pack(side=tk.LEFT, padx=(0, 5))
        self.ai_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, variable=self.ai_var).pack(side=tk.LEFT, padx=(0, 15))
        
        # Image display
        display_frame = ttk.Frame(main_frame)
        display_frame.pack(fill=tk.BOTH, expand=True)
        
        self.notebook = ttk.Notebook(display_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        self.original_frame = ttk.Frame(self.notebook)
        self.enhanced_frame = ttk.Frame(self.notebook)
        self.comparison_frame = ttk.Frame(self.notebook)
        
        self.notebook.add(self.original_frame, text="Original")
        self.notebook.add(self.enhanced_frame, text="Enhanced")
        self.notebook.add(self.comparison_frame, text="Side by Side")
        
        self.orig_canvas = tk.Canvas(self.original_frame, bg='#1e1e1e', highlightthickness=0)
        self.orig_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.enh_canvas = tk.Canvas(self.enhanced_frame, bg='#1e1e1e', highlightthickness=0)
        self.enh_canvas.pack(fill=tk.BOTH, expand=True)
        
        self.comp_canvas = tk.Canvas(self.comparison_frame, bg='#1e1e1e', highlightthickness=0)
        self.comp_canvas.pack(fill=tk.BOTH, expand=True)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        status_bar = ttk.Label(self.root, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def _setup_layout(self):
        self.root.bind('<Control-o>', lambda e: self.open_image())
        self.root.bind('<Control-s>', lambda e: self.save_image())
    
    def open_image(self):
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                       ("All files", "*.*")]
        )
        if not file_path:
            return
        
        self.current_image_path = file_path
        self.original_image = cv2.imread(file_path)
        if self.original_image is None:
            messagebox.showerror("Error", f"Cannot load image: {file_path}")
            return
        
        self.original_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2RGB)
        self.enhanced_image = None
        
        self._display_original()
        self.enhance_btn.config(state=tk.NORMAL)
        self.save_btn.config(state=tk.DISABLED)
        self.status_var.set(f"Loaded: {Path(file_path).name}")
    
    def _display_original(self):
        img = self._resize_for_display(self.original_image, self.orig_canvas)
        self.orig_photo = ImageTk.PhotoImage(img)
        self.orig_canvas.delete("all")
        self.orig_canvas.create_image(
            self.orig_canvas.winfo_width() // 2 or 400,
            self.orig_canvas.winfo_height() // 2 or 300,
            image=self.orig_photo, anchor=tk.CENTER
        )
    
    def _display_enhanced(self):
        if self.enhanced_image is None:
            return
        img = self._resize_for_display(self.enhanced_image, self.enh_canvas)
        self.enh_photo = ImageTk.PhotoImage(img)
        self.enh_canvas.delete("all")
        self.enh_canvas.create_image(
            self.enh_canvas.winfo_width() // 2 or 400,
            self.enh_canvas.winfo_height() // 2 or 300,
            image=self.enh_photo, anchor=tk.CENTER
        )
    
    def _display_comparison(self):
        if self.original_image is None or self.enhanced_image is None:
            return
        
        h, w = self.original_image.shape[:2]
        max_display_w = self.comp_canvas.winfo_width() // 2 - 20 or 300
        max_display_h = self.comp_canvas.winfo_height() - 20 or 250
        scale = min(max_display_w / w, max_display_h / h, 1.0)
        new_size = (int(w * scale), int(h * scale))
        
        orig_resized = cv2.resize(self.original_image, new_size, interpolation=cv2.INTER_AREA)
        enh_resized = cv2.resize(self.enhanced_image, new_size, interpolation=cv2.INTER_AREA)
        
        separator = np.ones((new_size[1], 5, 3), dtype=np.uint8) * 255
        combined = np.hstack([orig_resized, separator, enh_resized])
        
        comp_pil = Image.fromarray(combined)
        self.comp_photo = ImageTk.PhotoImage(comp_pil)
        self.comp_canvas.delete("all")
        self.comp_canvas.create_image(
            self.comp_canvas.winfo_width() // 2 or 400,
            self.comp_canvas.winfo_height() // 2 or 300,
            image=self.comp_photo, anchor=tk.CENTER
        )
        
        orig_w = new_size[0]
        enh_start = orig_w + 5
        self.comp_canvas.create_text(orig_w // 2, 15, text="Original", fill="white", font=("Arial", 10, "bold"))
        self.comp_canvas.create_text(orig_w + 5 + new_size[0] // 2, 15, text="Enhanced", fill="white", font=("Arial", 10, "bold"))
    
    def _resize_for_display(self, image, canvas):
        h, w = image.shape[:2]
        canvas.update_idletasks()
        max_w = canvas.winfo_width() - 20 or 500
        max_h = canvas.winfo_height() - 20 or 400
        scale = min(max_w / w, max_h / h, 1.0)
        if scale < 1.0:
            new_size = (int(w * scale), int(h * scale))
            resized = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
        else:
            resized = image
        return Image.fromarray(resized)
    
    def enhance_image(self, mode='full'):
        if self.original_image is None:
            return
        
        self.progress.pack(fill=tk.X, pady=(5, 0))
        self.progress.start()
        self.status_var.set("Enhancing...")
        self.enhance_btn.config(state=tk.DISABLED)
        
        config = EnhancementConfig()
        config.traditional['denoise']['enabled'] = self.denoise_var.get()
        config.traditional['sharpen']['enabled'] = self.sharpen_var.get()
        config.traditional['color_balance']['enabled'] = self.color_var.get()
        config.traditional['contrast']['enabled'] = self.contrast_var.get()
        
        if mode == 'traditional':
            config.pipeline_order = ['traditional']
            config.ai['model_path'] = None
        elif mode == 'ai':
            config.pipeline_order = ['ai']
        else:
            config.pipeline_order = ['traditional', 'ai']
            if not self.ai_var.get():
                config.pipeline_order = ['traditional']
        
        self.pipeline = EnhancementPipeline(config)
        
        def process():
            try:
                self.enhanced_image = self.pipeline.enhance(self.original_image.copy())
                self.root.after(0, self._on_enhance_complete)
            except Exception as e:
                self.root.after(0, lambda: self._on_enhance_error(str(e)))
        
        threading.Thread(target=process, daemon=True).start()
    
    def enhance_pipeline(self):
        self.enhance_image('full')
    
    def _on_enhance_complete(self):
        self.progress.stop()
        self.progress.pack_forget()
        self._display_enhanced()
        self._display_comparison()
        self.save_btn.config(state=tk.NORMAL)
        self.enhance_btn.config(state=tk.NORMAL)
        self.status_var.set("Enhancement complete!")
        self.notebook.select(self.enhanced_frame)
    
    def _on_enhance_error(self, error):
        self.progress.stop()
        self.progress.pack_forget()
        self.enhance_btn.config(state=tk.NORMAL)
        self.status_var.set("Enhancement failed")
        messagebox.showerror("Enhancement Error", error)
    
    def save_image(self):
        if self.enhanced_image is None:
            return
        
        file_path = filedialog.asksaveasfilename(
            title="Save Enhanced Image",
            defaultextension=".jpg",
            filetypes=[("JPEG", "*.jpg"), ("PNG", "*.png"), ("All files", "*.*")]
        )
        if not file_path:
            return
        
        save_img = cv2.cvtColor(self.enhanced_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(file_path, save_img)
        self.status_var.set(f"Saved: {Path(file_path).name}")
    
    def show_original(self):
        self.notebook.select(self.original_frame)
    
    def show_enhanced(self):
        self.notebook.select(self.enhanced_frame)
    
    def toggle_view(self):
        if self.view_menu_compare.get():
            self.notebook.add(self.comparison_frame, text="Side by Side")
        else:
            self.notebook.forget(self.comparison_frame)


def run_gui():
    root = tk.Tk()
    app = ImageEnhancerGUI(root)
    root.mainloop()


if __name__ == '__main__':
    run_gui()