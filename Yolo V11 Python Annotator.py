# yolo_v11_annotator.py
from box_item import BoxItem
from resizeHandle import MIN_BOX_SIZE, ResizeHandle
import sys, os, json, math
from typing import Callable, List, Tuple, Dict
from GraphicsView import GraphicsView
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QPushButton, QFileDialog,
    QGraphicsView, QGraphicsScene,
    QVBoxLayout, QHBoxLayout, QListWidget, QListWidgetItem, QLabel,
    QProgressDialog, QDoubleSpinBox, QComboBox, QSpinBox, QMessageBox,
    QDialog, QCheckBox, QGroupBox, QScrollArea, QSlider, QTableWidget,
    QTableWidgetItem, QHeaderView
)
from PySide6.QtGui import QPixmap, QMouseEvent, QKeyEvent, QImage, QAction, QKeySequence
from PySide6.QtCore import Qt, QRectF, QPointF, QThread, Signal, QEvent, QObject, QTimer
from ultralytics import YOLO
from save_worker import SaveWorker
import torch
import shutil  # Added for file copying

# For image similarity detection
import numpy as np
from PIL import Image
import imagehash  # May need to be installed with pip


# ─────────────────── threaded auto‑annotate worker ───────────────────
class AutoAnnotateWorker(QThread):
    progress = Signal(int, int)
    result = Signal(str, list)
    finished_ok = Signal()
    error = Signal(str)

    def __init__(self, image_paths: List[str], model,
                 conf_thresh: float, classes: set[int] | None):
        super().__init__()
        self.paths = image_paths
        self.model = model
        self.conf_th = conf_thresh
        self.allowed = classes
        # Class name mapping (will map 'bottle' to class ID 0)
        self.class_mapping = {"bottle": 0}
        # Batch size for processing multiple images at once - reduced to prevent memory issues
        self.batch_size = 2  # Even smaller batch size for high-res images
        # Set proper image size that's multiple of 32
        self.img_size = (1088, 1920)  # Height, Width format for YOLO (adjusted to be multiple of 32)

    def run(self):
        try:
            total = len(self.paths)
            # Process images in batches
            for batch_start in range(0, total, self.batch_size):
                batch_end = min(batch_start + self.batch_size, total)
                batch_paths = self.paths[batch_start:batch_end]

                # Filter out invalid image paths
                valid_paths = []
                for path in batch_paths:
                    if os.path.exists(path) and os.path.isfile(path):
                        # Check if image can be loaded
                        try:
                            import cv2
                            img = cv2.imread(path)
                            if img is not None and img.size > 0:
                                valid_paths.append(path)
                            else:
                                self.error.emit(f"Cannot read image: {path}")
                        except Exception as e:
                            self.error.emit(f"Error checking image {path}: {str(e)}")
                    else:
                        self.error.emit(f"File does not exist: {path}")

                if not valid_paths:
                    # Skip to next batch if no valid images
                    for i in range(batch_start, min(batch_start + self.batch_size, total)):
                        self.progress.emit(i + 1, total)
                    continue

                # Run prediction on batch with explicitly set size (multiple of 32)
                try:
                    results = self.model.predict(
                        source=valid_paths,
                        conf=self.conf_th,
                        save=False,
                        imgsz=self.img_size,  # Height, Width format for YOLO (multiple of 32)
                        verbose=False
                    )

                    # Process results for each image in batch
                    for idx, (path, res) in enumerate(zip(valid_paths, results)):
                        boxes = []
                        for b in res.boxes:
                            cls_id = int(b.cls.cpu())

                            # Get the class name from the model
                            class_name = self.model.names[
                                cls_id].lower() if cls_id in self.model.names else f"class_{cls_id}"

                            # Apply class mapping for bottle
                            if class_name == "bottle":
                                cls_id = 0  # Force bottle to be class 0

                            if self.allowed is not None and cls_id not in self.allowed:
                                continue

                            score = float(b.conf.cpu())
                            if score < self.conf_th:
                                continue
                            x1, y1, x2, y2 = b.xyxy.tolist()[0]
                            boxes.append((cls_id, score, x1, y1, x2 - x1, y2 - y1))

                        self.result.emit(path, boxes)
                        self.progress.emit(batch_start + idx + 1, total)

                except Exception as e:
                    import traceback
                    error_msg = f"Error in prediction: {str(e)}\n{traceback.format_exc()}"
                    self.error.emit(error_msg)
                    for i in range(batch_start, min(batch_start + self.batch_size, total)):
                        self.progress.emit(i + 1, total)

                # Force garbage collection after each batch
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self.finished_ok.emit()
        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


# ─────────────────── threaded model‑load worker ────────────────────
class ModelLoadWorker(QThread):
    finished_ok = Signal(object)  # emits the loaded YOLO model
    error = Signal(str)

    def __init__(self, model_path: str, half_precision=True):
        super().__init__()
        self.model_path = model_path
        self.half_precision = half_precision  # Use FP16 for faster inference

    def run(self):
        try:
            # Force garbage collection before loading a new model
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            device = "cuda:0" if torch.cuda.is_available() else "cpu"
            # Use "cuda:0" explicitly to ensure GPU usage when available
            model = YOLO(self.model_path)

            if device != "cpu":
                model.to(device)
                # Apply half precision for faster inference if requested and on GPU
                if self.half_precision and device != "cpu":
                    model.model.half()

            self.finished_ok.emit(model)
        except Exception as e:
            import traceback
            error_msg = f"Error loading model {self.model_path}: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


# ─────────────────── parallel model loader ────────────────────
class ParallelModelLoader(QObject):
    model_loaded = Signal(int, object)  # index, model
    all_finished = Signal()
    progress = Signal(int, int)  # loaded, total

    def __init__(self, model_paths, half_precision=True):
        super().__init__()
        self.model_paths = model_paths
        self.workers = []
        self.loaded_count = 0
        self.total = len(model_paths)
        self.half_precision = half_precision
        # Maximum number of models to load concurrently
        self.max_concurrent = 2  # Reduce this for higher resolution images
        self.pending_indices = list(range(self.total))
        self.active_workers = 0

    def start_loading(self):
        """Start loading models with controlled concurrency."""
        # Start loading up to max_concurrent models
        self._start_next_batch()

    def _start_next_batch(self):
        """Start loading the next batch of models."""
        # Start loading as many models as we can up to max_concurrent
        while self.active_workers < self.max_concurrent and self.pending_indices:
            i = self.pending_indices.pop(0)
            self._start_loading_model(i)

    def _start_loading_model(self, index):
        """Start loading a single model."""
        path = self.model_paths[index]
        worker = ModelLoadWorker(path, self.half_precision)
        worker.finished_ok.connect(lambda model, idx=index: self._model_loaded(idx, model))
        worker.error.connect(lambda err, idx=index: self._model_error(idx, err))
        self.workers.append(worker)
        self.active_workers += 1
        worker.start()

    def _model_loaded(self, index, model):
        """Handler for when a model is loaded successfully."""
        self.loaded_count += 1
        self.active_workers -= 1
        self.model_loaded.emit(index, model)
        self.progress.emit(self.loaded_count, self.total)

        # Start loading next model if any pending
        self._start_next_batch()

        # Check if all models are loaded
        if self.loaded_count == self.total:
            self.all_finished.emit()

    def _model_error(self, index, error):
        """Handler for when a model fails to load."""
        print(f"Error loading model at index {index}: {error}")
        self.loaded_count += 1
        self.active_workers -= 1
        self.progress.emit(self.loaded_count, self.total)

        # Start loading next model if any pending
        self._start_next_batch()

        # Check if all models are loaded (or failed)
        if self.loaded_count == self.total:
            self.all_finished.emit()

    def cancel(self):
        """Cancel all loading operations."""
        self.pending_indices = []  # Clear pending models
        for worker in self.workers:
            if worker.isRunning():
                worker.terminate()
                worker.wait(100)  # Wait a bit for termination


# Add this new class for caching annotations
class AnnotationCache:
    def __init__(self):
        self.cache = {}  # hash -> annotations

    def get_hash(self, image_path):
        """Generate a simple hash for the image based on path and modification time."""
        try:
            mtime = os.path.getmtime(image_path)
            return f"{image_path}_{mtime}"
        except:
            return image_path  # Fallback to just path if can't get mtime

    def has(self, image_path):
        """Check if annotations for this image are cached."""
        img_hash = self.get_hash(image_path)
        return img_hash in self.cache

    def get(self, image_path):
        """Get cached annotations for an image."""
        img_hash = self.get_hash(image_path)
        return self.cache.get(img_hash, [])

    def store(self, image_path, annotations):
        """Store annotations for an image in cache."""
        img_hash = self.get_hash(image_path)
        self.cache[img_hash] = annotations

    def clear(self):
        """Clear all cached annotations."""
        self.cache = {}


# Add this class for optimized image loading
class ImageLoaderWorker(QThread):
    image_loaded = Signal(str, QPixmap, int, int)  # path, pixmap, width, height
    error = Signal(str)

    def __init__(self, path):
        super().__init__()
        self.path = path

    def run(self):
        try:
            # Load image at reduced size first for faster display
            pixmap = QPixmap(self.path)
            w, h = pixmap.width(), pixmap.height()
            self.image_loaded.emit(self.path, pixmap, w, h)
        except Exception as e:
            self.error.emit(str(e))


# Add ImageSimilarityWorker class after other worker classes
class ImageSimilarityWorker(QThread):
    progress = Signal(int, int)  # current, total
    result = Signal(list)  # list of duplicate pairs
    finished_ok = Signal()
    error = Signal(str)

    def __init__(self, image_paths, threshold):
        super().__init__()
        self.image_paths = image_paths
        self.threshold = threshold

    def run(self):
        try:
            total = len(self.image_paths)
            if total < 2:
                self.finished_ok.emit()
                return

            # Dictionary to store image hashes
            hashes = {}

            # Compute hashes for all images
            for i, img_path in enumerate(self.image_paths):
                try:
                    # Load image and compute hash
                    img = Image.open(img_path)
                    img_hash = imagehash.phash(img)
                    hashes[img_path] = img_hash

                    # Report progress
                    self.progress.emit(i + 1, total)
                except Exception as e:
                    print(f"Error processing {img_path}: {str(e)}")

            # Find similar images
            duplicates = []
            paths = list(hashes.keys())

            for i in range(len(paths)):
                for j in range(i + 1, len(paths)):
                    path1, path2 = paths[i], paths[j]
                    hash1, hash2 = hashes[path1], hashes[path2]

                    # Calculate hash difference
                    diff = hash1 - hash2

                    # If difference is below threshold, consider as duplicate
                    if diff < self.threshold:
                        duplicates.append((path1, path2, diff))

            # Emit results
            self.result.emit(duplicates)
            self.finished_ok.emit()

        except Exception as e:
            import traceback
            error_msg = f"Error: {str(e)}\n{traceback.format_exc()}"
            self.error.emit(error_msg)


# ─────────────────────────── Main annotator ──────────────────────────
class Annotator(QMainWindow):
    DUP_TOL = 1.0  # pixel tolerance for "same" box

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO v11 Annotation Tool");
        self.showMaximized()

        # ------------- state containers --------------------------------------
        self.undo_stack: List[Callable[[], None]] = []
        self.model = None
        self.image_paths: List[str] = []
        self.curr_index = -1
        self.image_path = None
        self.annotations: List[BoxItem] = []
        self.annotations_store: Dict[str, List[Tuple]] = {}
        self.conf_thresh = 0.25
        self.class_filter: set[int] | None = None
        self.current_draw_class = 0  # Default class for new boxes
        self.annotation_cache = AnnotationCache()  # Add cache system
        self.use_half_precision = True  # Use half precision by default

        # Add image loading and caching
        self.image_loader = None
        self.image_cache = {}  # Cache for loaded images
        self.image_cache_max_size = 20  # Max number of images to keep in cache

        # Add model tracking
        self.model_status = {}  # Dictionary to track model status
        self.model_names = []  # List of model names/paths
        self.current_working_model_index = -1  # Index of currently working model

        # Add UI responsiveness timer
        self.debounce_timer = QTimer()
        self.debounce_timer.setSingleShot(True)
        self.debounce_timer.timeout.connect(self._delayed_ui_update)
        self.pending_ui_update = False

        # Define class names for the UI
        self.class_names = {
            0: "Bottle",
            1: "Plastic bottle",
            2: "Custom 1",
            3: "Custom 2",
            4: "Custom 3",
            5: "Custom 4",
            6: "Custom 5",
            7: "Custom 6",
            8: "Custom 7",
            9: "Custom 8"
        }

        # ---------------------- buttons & info bar -----------------------------
        open_btn = QPushButton("Open Images")
        open_btn.setToolTip("Open a set of new images (replaces current set)")
        add_photos_btn = QPushButton("Add Photos")
        add_photos_btn.setToolTip("Add more images to the existing set")

        load_proj_btn = QPushButton("Load Project")
        load_proj_btn.setToolTip("Load a saved annotation project")
        save_proj_btn = QPushButton("Save Project")
        save_proj_btn.setToolTip("Save current annotation project to a file")

        model_btn = QPushButton("Load Model")
        model_btn.setToolTip("Load a YOLO model for auto-annotation")
        save_btn = QPushButton("Save Annotations")
        save_btn.setToolTip("Save annotation data as YOLO format txt files")

        auto_curr_btn = QPushButton("Auto‑Annotate Current")
        auto_curr_btn.setToolTip("Run auto-annotation on the current image")
        auto_all_btn = QPushButton("Auto‑Annotate All")
        auto_all_btn.setToolTip("Run auto-annotation on all loaded images")

        try_all_models_btn = QPushButton("Try All Models")
        try_all_models_btn.setToolTip("Try all YOLO models in a folder to annotate images")

        clear_ann_btn = QPushButton("Clear Annotations")
        clear_ann_btn.setToolTip("Clear all annotations on the current image")
        delete_img_btn = QPushButton("Delete Current Image")
        delete_img_btn.setToolTip("Remove the current image from your dataset")

        prev_btn = QPushButton("← Prev")
        prev_btn.setToolTip("Go to previous image")
        next_btn = QPushButton("Next →")
        next_btn.setToolTip("Go to next image")

        conf_spin = QDoubleSpinBox()
        conf_spin.setRange(0.01, 1.0)
        conf_spin.setDecimals(2)
        conf_spin.setSingleStep(0.05)
        conf_spin.setValue(self.conf_thresh)
        conf_spin.setPrefix("Conf ≥ ")
        conf_spin.setToolTip("Confidence threshold for auto-annotation")
        conf_spin.valueChanged.connect(lambda v: setattr(self, 'conf_thresh', v))

        # Replace info label with navigation control
        self.total_images_label = QLabel("Images: 0")
        self.image_spin = QSpinBox()
        self.image_spin.setMinimum(1)
        self.image_spin.setMaximum(1)
        self.image_spin.setToolTip("Jump to specific image number")
        self.image_spin.valueChanged.connect(self.go_to_image)
        self.image_spin.setEnabled(False)
        self.max_images_label = QLabel("/ 0")

        self.class_combo = QComboBox()
        self.class_combo.setEnabled(False)  # enabled after model load
        self.class_combo.addItem("All classes", userData=None)
        self.class_combo.setToolTip("Filter specific classes for auto-annotation")

        def _combo_changed():
            data = self.class_combo.currentData()
            self.class_filter = None if data is None else {data}

        self.class_combo.currentIndexChanged.connect(_combo_changed)

        open_btn.clicked.connect(self.open_images);
        load_proj_btn.clicked.connect(self.load_project)
        model_btn.clicked.connect(self.load_model)
        auto_curr_btn.clicked.connect(self.auto_annotate_current);
        auto_all_btn.clicked.connect(self.auto_annotate_all)
        save_btn.clicked.connect(self.save_annotations);
        save_proj_btn.clicked.connect(self.save_project)
        add_photos_btn.clicked.connect(self.add_photos)
        try_all_models_btn.clicked.connect(self.try_all_models)
        clear_ann_btn.clicked.connect(self.clear_annotations)
        delete_img_btn.clicked.connect(self.delete_current_image)
        prev_btn.clicked.connect(self.prev_image);
        next_btn.clicked.connect(self.next_image)

        auto_curr_btn.setEnabled(False);
        auto_all_btn.setEnabled(False)
        self._auto_curr_btn, self._auto_all_btn = auto_curr_btn, auto_all_btn

        # Create button groups with separators
        file_group = QHBoxLayout()
        file_group.addWidget(open_btn)
        file_group.addWidget(add_photos_btn)
        file_group.addWidget(load_proj_btn)
        file_group.addWidget(save_proj_btn)
        file_group.addStretch(1)

        model_group = QHBoxLayout()
        model_group.addWidget(model_btn)
        model_group.addWidget(conf_spin)
        model_group.addWidget(self.class_combo)
        model_group.addWidget(auto_curr_btn)
        model_group.addWidget(auto_all_btn)
        model_group.addWidget(try_all_models_btn)
        model_group.addWidget(clear_ann_btn)
        model_group.addStretch(1)

        nav_group = QHBoxLayout()
        nav_group.addWidget(save_btn)
        nav_group.addWidget(delete_img_btn)
        nav_group.addStretch(1)
        nav_group.addWidget(prev_btn)
        nav_group.addWidget(self.image_spin)
        nav_group.addWidget(self.max_images_label)
        nav_group.addWidget(next_btn)

        # Main button layout
        btn_bar = QVBoxLayout()
        btn_bar.addLayout(file_group)
        btn_bar.addLayout(model_group)
        btn_bar.addLayout(nav_group)

        # ------------------ scene, view, list widget ---------------------------
        self.scene = QGraphicsScene();
        self.scene.selectionChanged.connect(self._update_handle_visibility)
        self.view = GraphicsView(self.scene);
        self.view.setFocusPolicy(Qt.StrongFocus)
        self.view.installEventFilter(self);
        self.view.viewport().installEventFilter(self)
        self.view.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.list_widget = QListWidget();
        self.list_widget.setMaximumWidth(220)
        self.list_widget.itemClicked.connect(self._list_clicked)

        # Add class selector UI
        class_selector_layout = QVBoxLayout()
        class_selector_layout.addWidget(QLabel("Change Box Class:"))

        self.box_class_combo = QComboBox()
        for class_id, class_name in self.class_names.items():
            self.box_class_combo.addItem(f"{class_id}: {class_name}", userData=class_id)
        self.box_class_combo.currentIndexChanged.connect(self._change_selected_box_class)
        class_selector_layout.addWidget(self.box_class_combo)

        apply_class_btn = QPushButton("Apply Class")
        apply_class_btn.clicked.connect(self._change_selected_box_class)
        class_selector_layout.addWidget(apply_class_btn)

        # Add checkbox to persist class
        self.use_persistent_class = QCheckBox("Use for new boxes")
        self.use_persistent_class.setChecked(True)
        class_selector_layout.addWidget(self.use_persistent_class)

        # Add selection info
        self.selection_info = QLabel("No box selected")
        class_selector_layout.addWidget(self.selection_info)

        # Add model status display
        model_status_group = QGroupBox("Model Status")
        model_status_layout = QVBoxLayout()

        self.model_status_list = QListWidget()
        self.model_status_list.setMaximumHeight(150)
        model_status_layout.addWidget(self.model_status_list)

        model_status_group.setLayout(model_status_layout)
        class_selector_layout.addWidget(model_status_group)

        # Add sort options
        sort_group = QGroupBox("Sort Images")
        sort_layout = QVBoxLayout()

        sort_by_empty = QPushButton("Sort: Empty First")
        sort_by_empty.setToolTip("Sort images with no annotations first")
        sort_by_empty.clicked.connect(lambda: self._sort_images("empty_first"))

        sort_by_most = QPushButton("Sort: Most Annotations")
        sort_by_most.setToolTip("Sort images by most annotations first")
        sort_by_most.clicked.connect(lambda: self._sort_images("most_first"))

        sort_by_least = QPushButton("Sort: Least Annotations")
        sort_by_least.setToolTip("Sort images by least annotations first")
        sort_by_least.clicked.connect(lambda: self._sort_images("least_first"))

        # Add button to remove duplicated images
        remove_duplicates_btn = QPushButton("Remove Duplicate Images")
        remove_duplicates_btn.setToolTip("Detect and remove similar images based on a threshold\n"
                                         "Requires PIL, numpy, and imagehash packages")
        remove_duplicates_btn.clicked.connect(self.remove_duplicate_images)

        # Add button to remove annotated images
        remove_annotated_btn = QPushButton("Remove & Save Annotated Images")
        remove_annotated_btn.setToolTip("Save annotated images and remove them from the current view")
        remove_annotated_btn.clicked.connect(self.save_and_remove_annotated_images)

        sort_layout.addWidget(sort_by_empty)
        sort_layout.addWidget(sort_by_most)
        sort_layout.addWidget(sort_by_least)
        sort_layout.addWidget(remove_duplicates_btn)
        sort_layout.addWidget(remove_annotated_btn)
        sort_group.setLayout(sort_layout)

        class_selector_layout.addWidget(sort_group)
        class_selector_layout.addStretch(1)

        # layout ----------------------------------------------------------------
        side_panel = QVBoxLayout()
        side_panel.addLayout(class_selector_layout)
        side_panel.addWidget(self.list_widget)

        lay = QVBoxLayout();
        lay.addLayout(btn_bar);
        main_content = QHBoxLayout();
        main_content.addWidget(self.view, 4);
        main_content.addLayout(side_panel, 1)
        lay.addLayout(main_content)
        root = QWidget();
        root.setLayout(lay);
        self.setCentralWidget(root)

        # misc state ------------------------------------------------------------
        self.drawing = False;
        self.start_point = QPointF();
        self.current_box = None
        self.save_worker = self.save_dialog = self.auto_worker = self.auto_dialog = None

        # ---------------- Add Keyboard Shortcuts ----------------------
        # Setup shortcuts
        self.shortcuts = {
            'next_image': QKeySequence(Qt.Key_Right),  # Right arrow
            'prev_image': QKeySequence(Qt.Key_Left),  # Left arrow
            'save_annotations': QKeySequence("Ctrl+S"),  # Ctrl+S
            'clear_annotations': QKeySequence("Ctrl+C"),  # Ctrl+C
            'auto_annotate': QKeySequence("Ctrl+A"),  # Ctrl+A
            'delete_image': QKeySequence("Ctrl+D"),  # Ctrl+D
            'class_0': QKeySequence("0"),  # 0-9 for class selection
            'class_1': QKeySequence("1"),
            'class_2': QKeySequence("2"),
            'class_3': QKeySequence("3"),
            'class_4': QKeySequence("4"),
            'class_5': QKeySequence("5"),
            'class_6': QKeySequence("6"),
            'class_7': QKeySequence("7"),
            'class_8': QKeySequence("8"),
            'class_9': QKeySequence("9"),
        }

        # Create shortcut actions
        self.shortcut_actions = {}
        for name, key_sequence in self.shortcuts.items():
            action = self.createShortcutAction(name, key_sequence)
            self.addAction(action)
            self.shortcut_actions[name] = action

        # Set tooltips to show shortcuts
        next_btn.setToolTip(f"Go to next image (Shortcut: {self.shortcuts['next_image'].toString()})")
        prev_btn.setToolTip(f"Go to previous image (Shortcut: {self.shortcuts['prev_image'].toString()})")
        save_btn.setToolTip(f"Save annotation data (Shortcut: {self.shortcuts['save_annotations'].toString()})")
        clear_ann_btn.setToolTip(f"Clear all annotations (Shortcut: {self.shortcuts['clear_annotations'].toString()})")
        auto_curr_btn.setToolTip(f"Run auto-annotation (Shortcut: {self.shortcuts['auto_annotate'].toString()})")
        delete_img_btn.setToolTip(f"Delete current image (Shortcut: {self.shortcuts['delete_image'].toString()})")

    def _delayed_ui_update(self):
        """Handle delayed UI updates to improve responsiveness."""
        if self.pending_ui_update:
            self.view.update()
            self.pending_ui_update = False

    def load_current_image(self):
        if self.curr_index == -1:
            return

        # clear scene and UI
        self.scene.clear()
        self.annotations.clear()
        self.list_widget.clear()
        self.undo_stack.clear()

        self.image_path = self.image_paths[self.curr_index]

        # Check if image is already in cache
        if self.image_path in self.image_cache:
            # Use cached image
            pixmap = self.image_cache[self.image_path]
            self.scene.addPixmap(pixmap)
            self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)
            self._restore_annotations()
            self._update_navigation_info()
        else:
            # Show loading indicator
            self.statusBar().showMessage(f"Loading {os.path.basename(self.image_path)}...")

            # Start async loading
            if self.image_loader and self.image_loader.isRunning():
                self.image_loader.terminate()

            self.image_loader = ImageLoaderWorker(self.image_path)
            self.image_loader.image_loaded.connect(self._on_image_loaded)
            self.image_loader.error.connect(self._on_image_load_error)
            self.image_loader.start()

            # Update navigation info
            self._update_navigation_info()

        # Preload adjacent images for smoother navigation
        self._preload_adjacent_images()

    def _on_image_loaded(self, path, pixmap, width, height):
        """Handle when an image is loaded by the worker."""
        if path != self.image_path:
            return  # User navigated away before loading completed

        # Add to cache
        self.image_cache[path] = pixmap

        # Trim cache if too large
        if len(self.image_cache) > self.image_cache_max_size:
            # Remove oldest items (not current and not neighbors)
            cache_paths = list(self.image_cache.keys())
            current_idx = self.curr_index

            # Keep current and neighboring images
            keep_indices = set([
                current_idx,
                max(0, current_idx - 1),
                min(len(self.image_paths) - 1, current_idx + 1)
            ])
            keep_paths = {self.image_paths[i] for i in keep_indices if 0 <= i < len(self.image_paths)}

            # Remove oldest items not in keep_paths
            for old_path in cache_paths:
                if old_path not in keep_paths and len(self.image_cache) > self.image_cache_max_size:
                    self.image_cache.pop(old_path, None)

        # Add to scene
        self.scene.addPixmap(pixmap)
        self.view.fitInView(self.scene.itemsBoundingRect(), Qt.KeepAspectRatio)

        # Restore annotations
        self._restore_annotations()

        # Clear status message
        self.statusBar().showMessage("")

    def _on_image_load_error(self, error_msg):
        """Handle image loading errors."""
        self.statusBar().showMessage(f"Error loading image: {error_msg}")

    def _restore_annotations(self):
        """Restore annotations for the current image."""
        # restore annotations
        for cls, conf, x, y, w_rect, h_rect in self.annotations_store.get(self.image_path, []):
            box = BoxItem(QRectF(x, y, w_rect, h_rect), cls, conf)
            self.scene.addItem(box)
            self.annotations.append(box)
            self.list_widget.addItem(f"Class {cls}, Conf {conf:.2f}")

    def _update_navigation_info(self):
        """Update navigation controls and window title."""
        # update info string & title
        total = len(self.image_paths)

        # Update navigation controls
        self.total_images_label.setText(f"Images: {total}")
        self.max_images_label.setText(f"/ {total}")

        # Update spinner (block signals to prevent recursion)
        self.image_spin.blockSignals(True)
        self.image_spin.setMinimum(1)
        self.image_spin.setMaximum(max(1, total))
        self.image_spin.setValue(self.curr_index + 1)
        self.image_spin.setEnabled(total > 0)
        self.image_spin.blockSignals(False)

        self.setWindowTitle(f"YOLO v11 Annotation Tool — "
                            f"{os.path.basename(self.image_path) if self.image_path else 'No Image'} "
                            f"({self.curr_index + 1}/{total})")
        self.view.setFocus()

    def resizeEvent(self, event):
        """Handle window resize events."""
        super().resizeEvent(event)
        # Only use debouncing if we've completed initialization
        if hasattr(self, 'debounce_timer') and self.debounce_timer is not None:
            self.pending_ui_update = True
            self.debounce_timer.start(100)  # 100ms delay

    # Add preloading for smoother navigation
    def next_image(self):
        if self.curr_index == -1 or self.curr_index >= len(self.image_paths) - 1:
            return
        self._store_current_annotations()
        self.curr_index += 1
        self.load_current_image()

        # Preload next image if available
        self._preload_adjacent_images()

    def prev_image(self):
        if self.curr_index <= 0:
            return
        self._store_current_annotations()
        self.curr_index -= 1
        self.load_current_image()

        # Preload previous image if available
        self._preload_adjacent_images()

    def _preload_adjacent_images(self):
        """Preload adjacent images for faster navigation."""
        # Preload next image
        next_idx = self.curr_index + 1
        if next_idx < len(self.image_paths) and self.image_paths[next_idx] not in self.image_cache:
            next_path = self.image_paths[next_idx]
            QTimer.singleShot(100, lambda: self._preload_image(next_path))

        # Preload previous image
        prev_idx = self.curr_index - 1
        if prev_idx >= 0 and self.image_paths[prev_idx] not in self.image_cache:
            prev_path = self.image_paths[prev_idx]
            QTimer.singleShot(100, lambda: self._preload_image(prev_path))

    def _preload_image(self, path):
        """Preload an image in the background."""
        if path in self.image_cache or not os.path.exists(path):
            return

        # Create a local reference to store the worker
        # This prevents it from being garbage collected while running
        preload_worker = ImageLoaderWorker(path)

        # Store reference to prevent garbage collection
        if not hasattr(self, '_preload_workers'):
            self._preload_workers = []
        self._preload_workers.append(preload_worker)

        # Connect with lambda to properly capture variables
        preload_worker.image_loaded.connect(
            lambda p, pixmap, w, h: self._handle_preloaded_image(p, pixmap, preload_worker))
        preload_worker.start()

    def _handle_preloaded_image(self, path, pixmap, worker):
        """Handle a preloaded image."""
        # Store in cache
        self.image_cache[path] = pixmap

        # Remove worker from list to allow garbage collection
        if hasattr(self, '_preload_workers') and worker in self._preload_workers:
            self._preload_workers.remove(worker)

    # ───────────────── duplicate‑box helpers ──────────────────────────
    @staticmethod
    def _boxes_equal(a: Tuple[float, float, float, float],
                     b: Tuple[float, float, float, float],
                     tol: float) -> bool:
        return (abs(a[0] - b[0]) <= tol and
                abs(a[1] - b[1]) <= tol and
                abs(a[2] - b[2]) <= tol and
                abs(a[3] - b[3]) <= tol)

    def _is_duplicate_box(self, path: str,
                          cls: int, x: float, y: float,
                          w: float, h: float) -> bool:
        # check stored boxes
        for c, _, sx, sy, sw, sh in self.annotations_store.get(path, []):
            if c == cls and self._boxes_equal((x, y, w, h), (sx, sy, sw, sh), self.DUP_TOL):
                return True
        # if path is current, also check live BoxItems (may include unsaved edits)
        if path == self.image_path:
            for box in self.annotations:
                if (box.cls == cls and
                        self._boxes_equal((x, y, w, h),
                                          (box.rect().x(), box.rect().y(),
                                           box.rect().width(), box.rect().height()),
                                          self.DUP_TOL)):
                    return True
        return False

    # ----------------- handle‑visibility helper -------------------------------
    def _update_handle_visibility(self):
        sel = [it for it in self.scene.selectedItems() if isinstance(it, BoxItem)]
        if len(sel) > 1:
            for it in sel[:-1]:
                it.setSelected(False)
            sel = sel[-1:]
        for box in self.annotations:
            (box.add_handles() if box in sel else box.remove_handles())

        # Update selection info
        if sel:
            box = sel[0]
            class_name = self.class_names.get(box.cls, f"Class {box.cls}")
            self.selection_info.setText(f"Selected: {class_name}\nConfidence: {box.conf:.2f}")

            # Update class combobox
            self.box_class_combo.blockSignals(True)
            index = self.box_class_combo.findData(box.cls)
            if index >= 0:
                self.box_class_combo.setCurrentIndex(index)
            self.box_class_combo.blockSignals(False)
        else:
            self.selection_info.setText("No box selected")

    def _change_selected_box_class(self):
        """Change the class of the selected box."""
        sel = [it for it in self.scene.selectedItems() if isinstance(it, BoxItem)]

        # Get selected class
        new_class = self.box_class_combo.currentData()

        # Update current draw class if option is checked
        if self.use_persistent_class.isChecked():
            self.current_draw_class = new_class

        if not sel:
            return

        box = sel[0]

        # Store old class for undo
        old_class = box.cls

        # Change class
        box.set_class(new_class)

        # Update list item
        if box in self.annotations:
            idx = self.annotations.index(box)
            self.list_widget.item(idx).setText(f"Class {new_class}, Conf {box.conf:.2f}")

        # Add undo operation
        def _undo_class_change(b=box, cls=old_class):
            b.set_class(cls)
            if b in self.annotations:
                idx = self.annotations.index(b)
                self.list_widget.item(idx).setText(f"Class {cls}, Conf {b.conf:.2f}")

        self._push_undo(_undo_class_change)

        # Update selection info
        class_name = self.class_names.get(new_class, f"Class {new_class}")
        self.selection_info.setText(f"Selected: {class_name}\nConfidence: {box.conf:.2f}")

    # ----------------- undo helper --------------------------------------------
    def _push_undo(self, fn):
        self.undo_stack.append(fn)

    # ----------------- keyboard shortcuts (Ctrl‑Z, Delete) --------------------
    def keyPressEvent(self, e: QKeyEvent):
        # undo
        if e.modifiers() & Qt.ControlModifier and e.key() == Qt.Key_Z:
            if self.undo_stack: self.undo_stack.pop()(); return

        # delete selected boxes
        if e.key() in (Qt.Key_Delete, Qt.Key_Backspace):
            selected = [it for it in self.scene.selectedItems() if isinstance(it, BoxItem)]
            if not selected:
                return
            # store info for undo
            stored = []
            for box in selected:
                if box in self.annotations:
                    idx = self.annotations.index(box)
                    label = self.list_widget.item(idx).text()
                    stored.append((box, idx, label))
            # perform deletion (reverse order to keep indices valid)
            for box, idx, _ in sorted(stored, key=lambda t: t[1], reverse=True):
                self.annotations.pop(idx)
                self.list_widget.takeItem(idx)
                self.scene.removeItem(box)

            # undo function
            def _undo_delete(data=stored):
                for box, idx, label in data:
                    self.scene.addItem(box)
                    self.annotations.insert(idx, box)
                    self.list_widget.insertItem(idx, label)

            self._push_undo(_undo_delete)
            return

        super().keyPressEvent(e)

    # ────────────────── event filter (mouse + keys) ──────────────────
    def eventFilter(self, src, ev):
        # ─── keyboard navigation: handle only once (on the view) ───
        if isinstance(ev, QKeyEvent) and src is self.view:
            if ev.type() != QEvent.KeyPress:  # ← ignore KeyRelease
                return False  # let others handle it

            if ev.isAutoRepeat():  # ← ignore auto‑repeats
                return True  # (consume, do nothing)

            if ev.key() in (Qt.Key_Right, Qt.Key_Down):
                self.next_image()
                return True

            if ev.key() in (Qt.Key_Left, Qt.Key_Up):
                self.prev_image()
                return True

        # mouse handling for drawing boxes
        if src is self.view.viewport() and isinstance(ev, QMouseEvent):
            pt = self.view.mapToScene(ev.position().toPoint())

            if ev.type() == QMouseEvent.MouseButtonPress and ev.button() == Qt.LeftButton:
                if any(isinstance(it, (BoxItem, ResizeHandle)) for it in self.scene.items(pt)):
                    return False
                self.drawing, self.start_point = True, pt
                self.current_box = BoxItem(QRectF(pt, pt), self.current_draw_class, 1.0)
                self.current_box.set_class(self.current_draw_class)
                self.scene.addItem(self.current_box)
                return True

            if ev.type() == QMouseEvent.MouseMove and self.drawing:
                self.current_box.setRect(QRectF(self.start_point, pt).normalized())
                return True

            if (ev.type() == QMouseEvent.MouseButtonRelease and
                    ev.button() == Qt.LeftButton and self.drawing):
                self.drawing = False
                self._finalize_manual_box(self.current_box)
                self.current_box = None
                return True

        return super().eventFilter(src, ev)

    # ----------------- small utilities ----------------------------------------
    def _set_conf_thresh(self, v: float):
        self.conf_thresh = v

    # ----------------------- model loading ---------------------------------
    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load YOLO Model", "", "PyTorch Model (*.pt)")
        if not path:
            return

        # ① Create a tiny modal "busy" dialog
        self.model_dialog = QProgressDialog(
            "Loading model…", "Cancel", 0, 0, self)
        self.model_dialog.setWindowTitle("Loading YOLO model")
        self.model_dialog.setWindowModality(Qt.ApplicationModal)
        self.model_dialog.setMinimumDuration(0)  # show immediately

        # ② Spin up the background worker
        self.model_worker = ModelLoadWorker(path, self.use_half_precision)
        self.model_worker.finished_ok.connect(self._model_loaded)
        self.model_worker.error.connect(self._model_error)
        self.model_dialog.canceled.connect(self.model_worker.terminate)
        self.model_worker.start()

    def _model_loaded(self, model):
        self.model_dialog.close()
        self.model = model
        self._auto_curr_btn.setEnabled(True)
        self._auto_all_btn.setEnabled(True)

        # ─── NEW: fill class list ───────────────────────────────
        names = model.names if isinstance(model.names, (list, tuple)) else list(model.names.values())
        self.class_combo.blockSignals(True)
        self.class_combo.clear();
        self.class_combo.addItem("All classes", userData=None)
        for i, n in enumerate(names): self.class_combo.addItem(f"{i}: {n}", userData=i)
        self.class_combo.setEnabled(True);
        self.class_combo.blockSignals(False)
        # ────────────────────────────────────────────────────────

        dev = "GPU" if next(model.model.parameters()).is_cuda else "CPU"
        self.statusBar().showMessage(f"Model loaded on {dev}")

    def _model_error(self, msg: str):
        self.model_dialog.close()
        self.statusBar().showMessage("Model load failed: " + msg)
        self.model = None
        self._auto_curr_btn.setEnabled(False)
        self._auto_all_btn.setEnabled(False)

    # ----------------- auto‑annotate helpers ----------------------------------
    def _start_auto_worker(self, paths: List[str], title: str):
        self.auto_dialog = QProgressDialog(f"{title}...", "Cancel", 0, len(paths), self)
        self.auto_dialog.setWindowTitle(title)
        self.auto_dialog.setWindowModality(Qt.ApplicationModal)
        self.auto_dialog.setMinimumDuration(0)

        self._auto_curr_btn.setEnabled(False)
        self._auto_all_btn.setEnabled(False)

        self.auto_worker = AutoAnnotateWorker(paths, self.model, self.conf_thresh, self.class_filter)
        self.auto_worker.progress.connect(lambda d, t: self.auto_dialog.setValue(d))
        self.auto_worker.result.connect(self._auto_result)
        self.auto_worker.finished_ok.connect(self._auto_finished)
        self.auto_worker.error.connect(self._auto_error)
        self.auto_dialog.canceled.connect(self.auto_worker.terminate)
        self.auto_worker.start()

    # ───────────────── auto‑annotate result handler ──────────────────
    def _auto_result(self, path: str, boxes: List[Tuple]):
        keep = []
        for cls, conf, x, y, w, h in boxes:
            if not self._is_duplicate_box(path, cls, x, y, w, h):
                keep.append((cls, conf, x, y, w, h))

        if not keep:
            return  # everything was dupes

        # store in regular storage
        self.annotations_store.setdefault(path, []).extend(keep)

        # Also store in cache
        all_boxes = self.annotations_store.get(path, [])
        self.annotation_cache.store(path, all_boxes)

        # if this image is on screen, draw new ones
        if path == self.image_path:
            for cls, conf, x, y, w, h in keep:
                box = BoxItem(QRectF(x, y, w, h), cls, conf)
                self.scene.addItem(box)
                self.annotations.append(box)
                self.list_widget.addItem(f"Class {cls}, Conf {conf:.2f}")
                self._push_undo(lambda b=box: self._remove_box_undo(b))

    # ───────────────── manual‑draw dedup check  ──────────────────────
    def _finalize_manual_box(self, box: BoxItem):
        """Called when a manually drawn box is finalized."""
        r = box.rect()

        # Set the class to current draw class
        box.cls = self.current_draw_class
        box.set_class(self.current_draw_class)

        if (r.width() < MIN_BOX_SIZE or r.height() < MIN_BOX_SIZE or
                self._is_duplicate_box(self.image_path, box.cls, r.x(), r.y(),
                                       r.width(), r.height())):
            # too small or duplicate → discard
            self.scene.removeItem(box)
            return

        self.annotations.append(box)
        self.list_widget.addItem(f"Class {box.cls}, Conf 1.00 (manual)")
        box.setSelected(True)

        # undo support
        def _undo_add(b=box):
            if b in self.annotations:
                idx = self.annotations.index(b)
                self.annotations.pop(idx)
                self.scene.removeItem(b)
                self.list_widget.takeItem(idx)

        self._push_undo(_undo_add)

    def _auto_finished(self):
        if self.auto_dialog:
            self.auto_dialog.close()
        self.statusBar().showMessage("Auto‑annotation completed.")
        self.auto_worker = self.auto_dialog = None
        self._auto_curr_btn.setEnabled(True)
        self._auto_all_btn.setEnabled(True)

    def _auto_error(self, msg: str):
        if self.auto_dialog:
            self.auto_dialog.close()
        self.statusBar().showMessage("Auto‑annotate failed: " + msg)
        self.auto_worker = self.auto_dialog = None
        self._auto_curr_btn.setEnabled(bool(self.model))
        self._auto_all_btn.setEnabled(bool(self.model))

    def _remove_box_undo(self, b: BoxItem):
        if b in self.annotations:
            idx = self.annotations.index(b)
            self.annotations.pop(idx)
            self.scene.removeItem(b)
            self.list_widget.takeItem(idx)

    # ----------------- public auto‑annotate actions ---------------------------
    def auto_annotate_current(self):
        if not self.model or self.curr_index == -1:
            self.statusBar().showMessage("Load a model and image first.")
            return
        self._start_auto_worker([self.image_path], "Auto‑annotating image")

    def auto_annotate_all(self):
        if not self.model or not self.image_paths:
            self.statusBar().showMessage("Load a model and images first.")
            return

        # Filter out already annotated images to only process unannotated ones
        unannotated_images = []
        for path in self.image_paths:
            # Skip images that already have annotations
            if path in self.annotations_store and self.annotations_store[path]:
                continue

            # Skip if already in cache
            if self.annotation_cache.has(path):
                # Apply cached annotations
                cached_boxes = self.annotation_cache.get(path)
                if cached_boxes:
                    self.annotations_store[path] = cached_boxes
                    # If current image, update display
                    if path == self.image_path:
                        self._refresh_displayed_annotations()
                continue

            # Skip current image being viewed (to avoid conflicts with user editing)
            if path == self.image_path:
                continue

            unannotated_images.append(path)

        if not unannotated_images:
            self.statusBar().showMessage("All other images already have annotations or are currently being edited.")
            return

        # Create a non-modal dialog that allows user to continue working
        self.auto_dialog = QProgressDialog(f"Auto-annotating in background...", "Cancel", 0, len(unannotated_images),
                                           self)
        self.auto_dialog.setWindowTitle("Background Auto-Annotation")
        self.auto_dialog.setWindowModality(Qt.NonModal)  # Non-modal to allow user to continue working
        self.auto_dialog.setMinimumDuration(0)

        self._auto_curr_btn.setEnabled(False)
        self._auto_all_btn.setEnabled(False)

        self.auto_worker = AutoAnnotateWorker(unannotated_images, self.model, self.conf_thresh, self.class_filter)
        self.auto_worker.progress.connect(lambda d, t: self.auto_dialog.setValue(d))
        self.auto_worker.result.connect(self._auto_result)
        self.auto_worker.finished_ok.connect(self._auto_finished)
        self.auto_worker.error.connect(self._auto_error)
        self.auto_dialog.canceled.connect(self.auto_worker.terminate)

        # Start the worker in background
        self.auto_worker.start()

        # Show a user-friendly message
        self.statusBar().showMessage(
            f"Auto-annotating {len(unannotated_images)} images in background. You can continue working.")

        # Keep auto buttons disabled until done
        self._auto_curr_btn.setEnabled(False)
        self._auto_all_btn.setEnabled(False)

    # ----------------- save‑annotations (existing threaded saver) -------------
    def save_annotations(self):
        if not self.image_paths:
            return

        # Store current annotations
        self._store_current_annotations()

        # Check if we have any annotations to save
        if not self.annotations_store:
            self.statusBar().showMessage("No annotations to save. Please annotate at least one image.")
            return

        # Ask user for destination directory
        dir_default = os.path.dirname(self.image_paths[0])
        dst_dir = QFileDialog.getExistingDirectory(
            self, "Select directory to save YOLO txt files", dir_default)
        if not dst_dir:
            return

        # Create a filtered version that includes all images (not just those with annotations)
        # We'll create empty txt files for images without annotations
        all_image_annotations = {path: self.annotations_store.get(path, []) for path in self.image_paths}

        # Create a filtered worker that saves all images
        self._run_threaded_save(dst_dir, title="Saving annotations", annotations=all_image_annotations)

    # ----------------- threaded save helpers ----------------------------------
    def _run_threaded_save(self, dst_dir: str, title: str, annotations=None):
        self.save_dialog = QProgressDialog(f"{title}...", "Cancel",
                                           0, len(self.image_paths), self)
        self.save_dialog.setWindowTitle(title)
        self.save_dialog.setWindowModality(Qt.ApplicationModal)
        self.save_dialog.setMinimumDuration(0)

        # Use provided annotations if available, otherwise use the stored ones
        data_to_save = annotations if annotations is not None else self.annotations_store

        self.save_worker = SaveWorker(self.image_paths, data_to_save, dst_dir)
        self.save_worker.progress.connect(
            lambda done, total: self.save_dialog.setValue(done))
        self.save_worker.finished_ok.connect(
            lambda d=dst_dir: self._save_done(d))
        self.save_worker.error.connect(self._save_error)
        self.save_dialog.canceled.connect(self.save_worker.terminate)
        self.save_worker.start()

    def _save_done(self, dst_dir):
        if self.save_dialog:
            self.save_dialog.close()
        self.statusBar().showMessage(
            f"Saved annotations for {len(self.image_paths)} images → {dst_dir}")
        self.save_worker = self.save_dialog = None

    def _save_error(self, msg):
        if self.save_dialog:
            self.save_dialog.close()
        self.statusBar().showMessage("Save failed: " + msg)
        self.save_worker = self.save_dialog = None

    # ─────────────────── I/O and inference ───────────────────────────
    def open_images(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Open Images", "", "Images (*.png *.jpg *.bmp)")
        if not paths:
            return
        self._store_current_annotations()
        self.image_paths = paths
        self.curr_index = 0
        self.load_current_image()

    def load_project(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Load Project", "", "Project Files (*.json)")
        if not path:
            return
        try:
            with open(path, "r") as f:
                data = json.load(f)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to load project: {e}")
            return

        self._store_current_annotations()
        self.image_paths = data.get("image_paths", [])
        self.annotations_store = {
            k: [tuple(item) for item in v] for k, v in data.get("annotations", {}).items()
        }
        if not self.image_paths:
            self.statusBar().showMessage("Project file contains no images.")
            return

        self.curr_index = 0
        self.load_current_image()
        self.statusBar().showMessage("Loaded project: " + os.path.basename(path))

    def add_photos(self):
        """Add more photos to the existing list of images."""
        if not self.image_paths:
            self.statusBar().showMessage("Please open some images first.")
            return

        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add More Images", "", "Images (*.png *.jpg *.bmp)")
        if not paths:
            return

        # Store current annotations before adding new images
        self._store_current_annotations()

        # Add new paths to the list
        self.image_paths.extend(paths)

        # Update the info label
        total = len(self.image_paths)
        self.total_images_label.setText(f"Images: {total}")
        self.max_images_label.setText(f"/ {total}")

        # Update spinner (block signals to prevent recursion)
        self.image_spin.blockSignals(True)
        self.image_spin.setMinimum(1)
        self.image_spin.setMaximum(max(1, total))
        self.image_spin.setValue(self.curr_index + 1)
        self.image_spin.setEnabled(total > 0)
        self.image_spin.blockSignals(False)

        self.statusBar().showMessage(f"Added {len(paths)} new images")

    # ───────────────────── saving (threaded) ─────────────────────────
    def save_project(self):
        if not self.image_paths:
            return
        self._store_current_annotations()

        default_name = "annotation_project.json"
        dst, _ = QFileDialog.getSaveFileName(
            self, "Save Project", default_name, "Project Files (*.json)")
        if not dst:
            return

        data = {
            "image_paths": self.image_paths,
            "annotations": {k: [list(t) for t in v] for k, v in self.annotations_store.items()}
        }
        try:
            with open(dst, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            self.statusBar().showMessage(f"Failed to save project: {e}")
            return

        # run threaded txt save
        self._run_threaded_save(os.path.dirname(dst), title="Saving project")

    # ────────────────── list/selection sync ──────────────────────────
    def _list_clicked(self, item: QListWidgetItem):
        idx = self.list_widget.row(item)
        self.annotations[idx].setSelected(True)

    # ─────────── image navigation (shared by keys & buttons) ─────────
    def go_to_image(self, index):
        """Navigate to a specific image index."""
        if not self.image_paths or self.curr_index == -1:
            return

        # Don't respond if the value was changed programmatically
        if index == self.curr_index + 1:
            return

        # Convert from 1-based to 0-based index
        target_idx = index - 1

        # Bounds check
        if target_idx < 0 or target_idx >= len(self.image_paths):
            # Reset to current value if out of bounds
            self.image_spin.blockSignals(True)
            self.image_spin.setValue(self.curr_index + 1)
            self.image_spin.blockSignals(False)
            return

        # Store current annotations before moving
        self._store_current_annotations()

        # Set new index and load image
        self.curr_index = target_idx
        self.load_current_image()

    def try_all_models(self):
        """Try all YOLO models in a folder to annotate images without annotations."""
        if not self.image_paths:
            self.statusBar().showMessage("Please open some images first.")
            return

        # Ask user for folder with models
        models_dir = QFileDialog.getExistingDirectory(
            self, "Select directory with YOLO models", "")
        if not models_dir:
            return

        # Find all .pt files (YOLO models)
        model_files = []
        for file in os.listdir(models_dir):
            if file.endswith(".pt"):
                model_files.append(os.path.join(models_dir, file))

        if not model_files:
            QMessageBox.warning(self, "No Models Found",
                                "No YOLO model files (.pt) found in the selected directory.")
            return

        # Show class selection dialog
        selected_classes = self._show_class_selection_dialog()
        if selected_classes is None:  # User cancelled
            return

        # Convert to set for faster lookup
        if selected_classes:
            self.class_filter = set(selected_classes)
        else:
            self.class_filter = None

        # Store current annotations
        self._store_current_annotations()

        # Initialize state for parallel processing
        self.model_files = model_files
        self.model_names = [os.path.basename(f) for f in model_files]  # Store model names
        self.loaded_models = [None] * len(model_files)  # Will store loaded models
        self.model_used = [False] * len(model_files)  # Track which models have been used
        self.current_working_model_index = -1  # Index of model currently annotating
        self.is_annotating = False  # Whether we're currently annotating
        self.unannotated_images = self._get_unannotated_images()

        # Initialize model status
        self.model_status = {i: "pending" for i in range(len(model_files))}
        self._update_model_status()

        # Filter out current image to avoid conflicts with editing
        self.unannotated_images = [path for path in self.unannotated_images if path != self.image_path]

        # If all images already have annotations, we're done
        if not self.unannotated_images:
            self.statusBar().showMessage("All images (except current) already have annotations.")
            return

        # Start the cascading model process
        self.statusBar().showMessage(f"Found {len(model_files)} models. Loading in background...")

        # Create a dialog to show progress
        self.cascade_dialog = QProgressDialog(
            "Loading models in parallel...", "Cancel", 0, len(model_files), self)
        self.cascade_dialog.setWindowTitle("Cascading Models")
        self.cascade_dialog.setWindowModality(Qt.NonModal)  # Non-modal to allow user to continue working
        self.cascade_dialog.setMinimumDuration(0)

        # Start parallel loading of all models
        self.parallel_loader = ParallelModelLoader(model_files, self.use_half_precision)
        self.parallel_loader.model_loaded.connect(self._on_parallel_model_loaded)
        self.parallel_loader.all_finished.connect(self._on_all_models_loaded)
        self.parallel_loader.progress.connect(
            lambda loaded, total: self.cascade_dialog.setValue(loaded))
        self.cascade_dialog.canceled.connect(self._cancel_parallel_loading)

        self.parallel_loader.start_loading()

        # Show a message that processing is happening in background
        self.statusBar().showMessage(
            "Models are loading and will annotate images in the background. You can continue working.")

    def _show_class_selection_dialog(self):
        """Show a dialog to select which classes to detect."""
        dialog = QDialog(self)
        dialog.setWindowTitle("Select Classes to Detect")

        # Define common YOLO classes
        common_classes = [
            "bottle", "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
            "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
            "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
            "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
            "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
            "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
            "teddy bear", "hair drier", "toothbrush"
        ]

        layout = QVBoxLayout()

        # Add instructions
        layout.addWidget(QLabel("Select classes to detect (bottle is selected by default):"))

        # Create a group box for better organization
        group_box = QGroupBox("Available Classes")
        group_layout = QVBoxLayout()

        # Create checkboxes for each class
        checkboxes = {}
        for class_name in common_classes:
            checkbox = QCheckBox(class_name)
            # Set bottle as checked by default
            if class_name == "bottle":
                checkbox.setChecked(True)
            checkboxes[class_name] = checkbox
            group_layout.addWidget(checkbox)

        group_box.setLayout(group_layout)

        # Add scrollable area for the checkboxes
        scroll_area = QScrollArea()
        scroll_area.setWidget(group_box)
        scroll_area.setWidgetResizable(True)
        scroll_area.setMinimumHeight(300)
        layout.addWidget(scroll_area)

        # Add buttons
        button_layout = QHBoxLayout()
        select_all_btn = QPushButton("Select All")
        deselect_all_btn = QPushButton("Deselect All")
        ok_btn = QPushButton("OK")
        cancel_btn = QPushButton("Cancel")

        button_layout.addWidget(select_all_btn)
        button_layout.addWidget(deselect_all_btn)
        button_layout.addStretch()
        button_layout.addWidget(ok_btn)
        button_layout.addWidget(cancel_btn)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # Connect buttons
        select_all_btn.clicked.connect(lambda: self._toggle_all_checkboxes(checkboxes, True))
        deselect_all_btn.clicked.connect(lambda: self._toggle_all_checkboxes(checkboxes, False))
        ok_btn.clicked.connect(dialog.accept)
        cancel_btn.clicked.connect(dialog.reject)

        # Show dialog
        result = dialog.exec_()

        if result == QDialog.Accepted:
            # Return list of selected class names
            selected = []
            for class_name, checkbox in checkboxes.items():
                if checkbox.isChecked():
                    # Map to class ID (bottles are always 0)
                    if class_name == "bottle":
                        selected.append(0)
                    else:
                        # Other classes get assigned sequential IDs
                        # For simplicity, we'll use the index in the common_classes list + 1
                        # (to avoid conflict with bottle's ID 0)
                        class_id = common_classes.index(class_name) + 1
                        selected.append(class_id)
            return selected
        else:
            return None

    def _toggle_all_checkboxes(self, checkboxes, state):
        """Toggle all checkboxes to the given state."""
        for checkbox in checkboxes.values():
            checkbox.setChecked(state)

    def _on_parallel_model_loaded(self, index, model):
        """Called when a model is loaded in parallel."""
        # Store the model
        self.loaded_models[index] = model

        # Update model status
        self.model_status[index] = "loaded"
        self._update_model_status()

        # Update dialog
        model_name = os.path.basename(self.model_files[index])
        self.cascade_dialog.setLabelText(f"Loaded model: {model_name}")

        # If we're not currently annotating, start with this model
        if not self.is_annotating:
            self._start_annotation_with_model(index)

    def _start_annotation_with_model(self, model_index):
        """Start annotating unannotated images with the specified model."""
        if self.is_annotating or not self.unannotated_images:
            return  # Already annotating or no images left

        # Check if the model is valid
        if self.loaded_models[model_index] is None:
            # Update status to failed
            self.model_status[model_index] = "failed"
            self._update_model_status()
            return

        # Filter out current image to avoid conflicts
        current_unannotated = [path for path in self.unannotated_images if path != self.image_path]

        if not current_unannotated:
            # All remaining unannotated images are either the current image or already processed
            self.is_annotating = False
            self.current_working_model_index = -1
            self._update_model_status()

            if self.cascade_dialog:
                self.cascade_dialog.close()
            QMessageBox.information(self, "Annotation Complete",
                                    f"Completed annotation except for the image you are currently viewing.")
            return

        # Set current state
        self.is_annotating = True
        self.current_working_model_index = model_index

        # Update model status
        self.model_status[model_index] = "working"
        self._update_model_status()

        # Clear previous model from memory if exists
        if hasattr(self, 'model') and self.model is not None and self.model != self.loaded_models[model_index]:
            # Set to None to help with garbage collection
            self.model = None

            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # Set the new model
        self.model = self.loaded_models[model_index]
        self.model_used[model_index] = True

        # Update dialog to non-modal to allow user to continue working
        model_name = os.path.basename(self.model_files[model_index])
        if self.cascade_dialog:
            self.cascade_dialog.setWindowModality(Qt.NonModal)
            self.cascade_dialog.setLabelText(
                f"Annotating with model {model_index + 1}/{len(self.model_files)}: {model_name} " +
                f"on {len(current_unannotated)} images in background")

        # Create a non-modal auto-annotation dialog
        self.auto_dialog = QProgressDialog(
            f"Auto-annotating with model {model_index + 1}/{len(self.model_files)}...",
            "Cancel", 0, len(current_unannotated), self)
        self.auto_dialog.setWindowTitle("Background Auto-Annotation")
        self.auto_dialog.setWindowModality(Qt.NonModal)  # Non-modal to allow user interaction
        self.auto_dialog.setMinimumDuration(0)

        # Start auto-annotation with this model
        self.auto_worker = AutoAnnotateWorker(current_unannotated, self.model, self.conf_thresh, self.class_filter)
        self.auto_worker.progress.connect(lambda d, t: self.auto_dialog.setValue(d))
        self.auto_worker.result.connect(self._auto_result)
        self.auto_worker.error.connect(self._auto_error)
        self.auto_dialog.canceled.connect(self.auto_worker.terminate)

        # Connect to completion signal
        try:
            # Attempt to disconnect any existing connections first
            self.auto_worker.finished_ok.disconnect(self._on_cascade_auto_complete)
        except (TypeError, RuntimeError):
            # No connections exist yet, that's fine
            pass

        # Connect the signal
        self.auto_worker.finished_ok.connect(self._on_cascade_auto_complete)

        # Start the worker thread
        self.auto_worker.start()

        # Show a status message that annotation is running in background
        self.statusBar().showMessage(
            f"Annotating with model {model_index + 1}/{len(self.model_files)} in background. You can continue working.")

    def _on_all_models_loaded(self):
        """Called when all models are loaded."""
        self.statusBar().showMessage("All models loaded.")

        # If no annotation is in progress but there are unannotated images,
        # try to start with any loaded model that hasn't been used yet
        if not self.is_annotating and self.unannotated_images:
            for i, model in enumerate(self.loaded_models):
                if model is not None and not self.model_used[i]:
                    self._start_annotation_with_model(i)
                    break

    def _on_cascade_auto_complete(self):
        """Called when auto-annotation with current model is complete."""
        # Update model status
        if self.current_working_model_index >= 0:
            self.model_status[self.current_working_model_index] = "passed"
            self._update_model_status()

        # Reset annotation state
        self.is_annotating = False
        current_model_index = self.current_working_model_index
        self.current_working_model_index = -1
        self._update_model_status()

        # Original auto-finished functionality
        if self.auto_dialog:
            self.auto_dialog.close()
        self.auto_worker = self.auto_dialog = None

        # Get updated list of unannotated images
        self.unannotated_images = self._get_unannotated_images()

        # Check if we're done
        if not self.unannotated_images:
            # All images now have annotations, we're done
            if self.cascade_dialog:
                self.cascade_dialog.close()
            QMessageBox.information(self, "Annotation Complete",
                                    f"Successfully annotated all images using {sum(self.model_used)} models.")

            # Clean up models to free memory
            for i in range(len(self.loaded_models)):
                if self.loaded_models[i] is not None:
                    self.loaded_models[i] = None

            # Force garbage collection
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return

        # Look for next unused model
        next_model_found = False
        for i, model in enumerate(self.loaded_models):
            if model is not None and not self.model_used[i]:
                # Clean up memory first
                if hasattr(self, 'model') and self.model is not None:
                    # Set to None to help with garbage collection
                    self.model = None

                    # Force garbage collection
                    import gc
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # Start with next model
                self._start_annotation_with_model(i)
                next_model_found = True
                break

        # If we've used all available models but still have unannotated images
        if not next_model_found:
            # Check if all models have been loaded
            if all(m is not None for m in self.loaded_models) or sum(self.model_used) == len(self.model_files):
                # We've tried all models but some images are still unannotated
                if self.cascade_dialog:
                    self.cascade_dialog.close()

                # Clean up models to free memory
                for i in range(len(self.loaded_models)):
                    if self.loaded_models[i] is not None:
                        self.loaded_models[i] = None

                # Force garbage collection
                import gc
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Ask user if they want to navigate to the first unannotated image
                result = QMessageBox.question(self, "Annotation Incomplete",
                                              f"Tried all {len(self.model_files)} models. {len(self.unannotated_images)} images still lack annotations. " +
                                              "Do you want to navigate to the first unannotated image?",
                                              QMessageBox.Yes | QMessageBox.No)

                if result == QMessageBox.Yes and self.unannotated_images:
                    first_empty = self.image_paths.index(self.unannotated_images[0])
                    self.curr_index = first_empty
                    self.load_current_image()
            # else: we're still loading models, so just wait

    def _auto_error(self, msg: str):
        """Handle auto-annotation errors."""
        # Update model status if this was during cascade annotation
        if self.current_working_model_index >= 0:
            self.model_status[self.current_working_model_index] = "failed"
            self._update_model_status()

        if self.auto_dialog:
            self.auto_dialog.close()
        self.statusBar().showMessage("Auto‑annotate failed: " + msg)
        self.auto_worker = self.auto_dialog = None
        self._auto_curr_btn.setEnabled(bool(self.model))
        self._auto_all_btn.setEnabled(bool(self.model))

        # If this was part of cascade annotation, try the next model
        if hasattr(self, 'is_annotating') and self.is_annotating:
            self.is_annotating = False
            self.current_working_model_index = -1
            self._update_model_status()

            # Try to start with next model
            for i, model in enumerate(self.loaded_models):
                if model is not None and not self.model_used[i]:
                    # Start with next model
                    self._start_annotation_with_model(i)
                    break

    def _cancel_parallel_loading(self):
        """Cancel the parallel model loading process."""
        if hasattr(self, 'parallel_loader'):
            self.parallel_loader.cancel()

            # Mark all pending models as skipped
            for i in range(len(self.model_status)):
                if self.model_status.get(i) == "pending":
                    self.model_status[i] = "skipped"

            # Update current working model
            self.current_working_model_index = -1
            self._update_model_status()

        if hasattr(self, 'auto_worker') and self.auto_worker and self.auto_worker.isRunning():
            self.auto_worker.terminate()

        self.is_annotating = False

    def _get_unannotated_images(self):
        """Get list of images that have no annotations yet."""
        unannotated = []
        for img_path in self.image_paths:
            # Check if image has no annotations
            if img_path not in self.annotations_store or not self.annotations_store[img_path]:
                unannotated.append(img_path)
        return unannotated

    def _try_next_model(self):
        """Legacy method, kept for compatibility."""
        pass

    def _on_cascade_model_loaded(self, model):
        """Legacy method, kept for compatibility."""
        pass

    def _on_cascade_model_error(self, error_msg):
        """Legacy method, kept for compatibility."""
        pass

    def _cancel_cascade(self):
        """Legacy method, kept for compatibility."""
        if hasattr(self, 'cascade_model_worker') and self.cascade_model_worker.isRunning():
            self.cascade_model_worker.terminate()

        if hasattr(self, 'auto_worker') and self.auto_worker and self.auto_worker.isRunning():
            self.auto_worker.terminate()

    def save_and_remove_annotated_images(self):
        """Save annotated images to a directory and remove them from the displayed list."""
        if not self.image_paths:
            self.statusBar().showMessage("No images loaded.")
            return

        # Store current annotations
        self._store_current_annotations()

        # Find images with annotations
        images_with_annotations = [
            path for path in self.image_paths
            if path in self.annotations_store and self.annotations_store[path]
        ]

        if not images_with_annotations:
            self.statusBar().showMessage("No images with annotations found.")
            return

        # Ask user for destination directory
        dst_dir = QFileDialog.getExistingDirectory(
            self, "Select directory to save annotated images", "")
        if not dst_dir:
            return

        # Create a progress dialog
        progress = QProgressDialog(
            "Saving annotated images...", "Cancel", 0, len(images_with_annotations), self)
        progress.setWindowTitle("Saving Images")
        progress.setWindowModality(Qt.ApplicationModal)

        # Copy images with annotations
        try:
            for i, src_path in enumerate(images_with_annotations):
                if progress.wasCanceled():
                    break

                # Get destination path
                filename = os.path.basename(src_path)
                dst_path = os.path.join(dst_dir, filename)

                # Copy the file using shutil
                shutil.copy2(src_path, dst_path)

                # Also save the corresponding annotation file
                txt_name = os.path.splitext(filename)[0] + ".txt"
                txt_dst = os.path.join(dst_dir, txt_name)

                # Get image dimensions
                pixmap = QPixmap(src_path)
                w_img, h_img = pixmap.width(), pixmap.height()

                with open(txt_dst, "w") as f:
                    for cls, conf, x, y, w_rect, h_rect in self.annotations_store[src_path]:
                        cx, cy = (x + w_rect / 2) / w_img, (y + h_rect / 2) / h_img
                        f.write(f"{cls} {cx:.6f} {cy:.6f} "
                                f"{w_rect / w_img:.6f} {h_rect / h_img:.6f}\n")

                progress.setValue(i + 1)

            # Now remove the saved images from view
            # Remember current position
            current_path = self.image_path
            current_idx = self.curr_index

            # Create a list of images without annotations
            self.image_paths = [path for path in self.image_paths if path not in images_with_annotations]

            # Update current index
            if not self.image_paths:
                # No images left
                self.scene.clear()
                self.annotations.clear()
                self.list_widget.clear()
                self.curr_index = -1
                self.image_path = None
                self.total_images_label.setText("Images: 0")
                self.max_images_label.setText("/ 0")
                self.image_spin.setValue(0)
                self.image_spin.setEnabled(False)
                self.setWindowTitle("YOLO v11 Annotation Tool")
                self.statusBar().showMessage(f"Deleted image: {os.path.basename(removed_path)}. No images remaining.")
                return

            # Try to keep position close to where user was
            if current_path in images_with_annotations:
                # If current image was removed, go to nearest available
                self.curr_index = min(current_idx, len(self.image_paths) - 1)
            else:
                # If current image was kept, find its new position
                try:
                    self.curr_index = self.image_paths.index(current_path)
                except ValueError:
                    self.curr_index = 0

            # Load the new current image
            self.load_current_image()

            self.statusBar().showMessage(f"Saved and removed {len(images_with_annotations)} annotated images.")
        except Exception as e:
            self.statusBar().showMessage(f"Error processing images: {str(e)}")
        finally:
            progress.close()

    def _sort_images(self, sort_type):
        """Sort images based on annotation status."""
        if not self.image_paths:
            return

        # Store current annotations
        self._store_current_annotations()

        # Get current image path to restore after sorting
        current_path = self.image_path

        # Create a list of (path, annotation_count) tuples
        image_data = []
        for path in self.image_paths:
            ann_count = len(self.annotations_store.get(path, []))
            image_data.append((path, ann_count))

        # Sort based on the requested criteria
        if sort_type == "empty_first":
            # Empty images first, then by path
            image_data.sort(key=lambda x: (x[1] > 0, x[0]))
        elif sort_type == "most_first":
            # Most annotations first
            image_data.sort(key=lambda x: (-x[1], x[0]))
        elif sort_type == "least_first":
            # Least annotations first (but not empty)
            image_data.sort(key=lambda x: (x[1] == 0, x[1], x[0]))

        # Update image paths with the new order
        self.image_paths = [path for path, _ in image_data]

        # Restore current index or go to first image
        try:
            self.curr_index = self.image_paths.index(current_path)
        except ValueError:
            self.curr_index = 0 if self.image_paths else -1

        # Load current image
        self.load_current_image()

        # Update status
        sort_name = sort_type.replace('_', ' ').title()
        self.statusBar().showMessage(f"Images sorted by: {sort_name}")

    # Add a method to update displayed annotations from store
    def _refresh_displayed_annotations(self):
        """Refresh displayed annotations from the annotation store."""
        if self.image_path not in self.annotations_store:
            return

        # Clear current annotations
        for box in list(self.annotations):
            self.scene.removeItem(box)
        self.annotations.clear()
        self.list_widget.clear()

        # Reload from store
        for cls, conf, x, y, w_rect, h_rect in self.annotations_store.get(self.image_path, []):
            box = BoxItem(QRectF(x, y, w_rect, h_rect), cls, conf)
            self.scene.addItem(box)
            self.annotations.append(box)
            self.list_widget.addItem(f"Class {cls}, Conf {conf:.2f}")

    def createShortcutAction(self, name, key_sequence):
        """Create a QAction for a keyboard shortcut."""
        action = QAction(self)
        action.setShortcut(key_sequence)

        # Connect the action to the appropriate method
        if name == 'next_image':
            action.triggered.connect(self.next_image)
        elif name == 'prev_image':
            action.triggered.connect(self.prev_image)
        elif name == 'save_annotations':
            action.triggered.connect(self.save_annotations)
        elif name == 'clear_annotations':
            action.triggered.connect(self.clear_annotations)
        elif name == 'auto_annotate':
            action.triggered.connect(self.auto_annotate_current)
        elif name == 'delete_image':
            action.triggered.connect(self.delete_current_image)
        elif name.startswith('class_'):
            class_id = int(name.split('_')[1])
            action.triggered.connect(lambda: self.set_current_class(class_id))

        return action

    def set_current_class(self, class_id):
        """Set the current drawing class when number keys are pressed."""
        if class_id in self.class_names:
            self.current_draw_class = class_id

            # Update class combobox
            index = self.box_class_combo.findData(class_id)
            if index >= 0:
                self.box_class_combo.setCurrentIndex(index)

            # Update selected box class if a box is selected
            sel = [it for it in self.scene.selectedItems() if isinstance(it, BoxItem)]
            if sel and self.use_persistent_class.isChecked():
                box = sel[0]
                old_class = box.cls
                box.set_class(class_id)

                # Update list item
                if box in self.annotations:
                    idx = self.annotations.index(box)
                    self.list_widget.item(idx).setText(f"Class {class_id}, Conf {box.conf:.2f}")

                # Add undo operation
                def _undo_class_change(b=box, cls=old_class):
                    b.set_class(cls)
                    if b in self.annotations:
                        idx = self.annotations.index(b)
                        self.list_widget.item(idx).setText(f"Class {cls}, Conf {b.conf:.2f}")

                self._push_undo(_undo_class_change)

            # Update status bar
            self.statusBar().showMessage(f"Selected class: {class_id} - {self.class_names[class_id]}")

    # ───────────────────────── persistence ───────────────────────────
    def _store_current_annotations(self):
        if self.curr_index == -1:
            return
        boxes: List[Tuple] = []
        for b in self.annotations:
            r = b.rect()
            boxes.append((b.cls, b.conf, r.x(), r.y(), r.width(), r.height()))
        self.annotations_store[self.image_path] = boxes

    def clear_annotations(self):
        """Clear all annotations on the current image."""
        if self.curr_index == -1 or not self.annotations:
            self.statusBar().showMessage("No annotations to clear.")
            return

        # Save annotations for undo
        stored_boxes = []
        for box in self.annotations:
            idx = self.annotations.index(box)
            label = self.list_widget.item(idx).text()
            stored_boxes.append((box, idx, label))

        # Clear annotations
        for box in list(self.annotations):  # use a copy to iterate
            self.scene.removeItem(box)
        self.annotations.clear()
        self.list_widget.clear()

        # Remove from storage if saved
        if self.image_path in self.annotations_store:
            del self.annotations_store[self.image_path]

        # Add undo capability
        def _undo_clear(data=stored_boxes):
            for box, idx, label in data:
                self.scene.addItem(box)
                self.annotations.insert(idx, box)
                self.list_widget.insertItem(idx, label)

            # Restore to storage
            boxes = []
            for b in self.annotations:
                r = b.rect()
                boxes.append((b.cls, b.conf, r.x(), r.y(), r.width(), r.height()))
            self.annotations_store[self.image_path] = boxes

        self._push_undo(_undo_clear)
        self.statusBar().showMessage("All annotations cleared.")

    def delete_current_image(self):
        """Remove the current image from the dataset."""
        if self.curr_index == -1:
            self.statusBar().showMessage("No image loaded.")
            return

        # Store image info for undo
        old_path = self.image_path
        old_index = self.curr_index
        old_annotations = []
        if old_path in self.annotations_store:
            old_annotations = self.annotations_store[old_path]

        # Save annotations of current image
        self._store_current_annotations()

        # Remove image from paths
        removed_path = self.image_paths.pop(self.curr_index)

        # Handle navigation
        if not self.image_paths:
            # No images left
            self.scene.clear()
            self.annotations.clear()
            self.list_widget.clear()
            self.curr_index = -1
            self.image_path = None
            self.total_images_label.setText("Images: 0")
            self.max_images_label.setText("/ 0")
            self.image_spin.setValue(0)
            self.image_spin.setEnabled(False)
            self.setWindowTitle("YOLO v11 Annotation Tool")
            self.statusBar().showMessage(f"Deleted image: {os.path.basename(removed_path)}. No images remaining.")
            return

        # Adjust index if necessary
        if self.curr_index >= len(self.image_paths):
            self.curr_index = len(self.image_paths) - 1

        # Load next image
        self.load_current_image()

        # Add undo capability
        def _undo_delete():
            self.image_paths.insert(old_index, old_path)
            self.curr_index = old_index
            if old_path in self.annotations_store:
                self.annotations_store[old_path] = old_annotations
            self.load_current_image()

        self._push_undo(_undo_delete)
        self.statusBar().showMessage(f"Deleted image: {os.path.basename(removed_path)}")

    # Add closeEvent method to the Annotator class
    def closeEvent(self, event):
        """Clean up threads before closing."""
        # Terminate any running worker threads
        if hasattr(self, 'image_loader') and self.image_loader and self.image_loader.isRunning():
            self.image_loader.terminate()
            self.image_loader.wait()

        if hasattr(self, 'model_worker') and self.model_worker and self.model_worker.isRunning():
            self.model_worker.terminate()
            self.model_worker.wait()

        if hasattr(self, 'auto_worker') and self.auto_worker and self.auto_worker.isRunning():
            self.auto_worker.terminate()
            self.auto_worker.wait()

        if hasattr(self, 'save_worker') and self.save_worker and self.save_worker.isRunning():
            self.save_worker.terminate()
            self.save_worker.wait()

        # Let the parent class handle the close event
        super().closeEvent(event)

    # Add method to update model status display
    def _update_model_status(self):
        """Update the model status display widget."""
        self.model_status_list.clear()

        if not self.model_names:
            self.model_status_list.addItem("No models loaded")
            return

        for i, model_name in enumerate(self.model_names):
            # Get shortened display name
            display_name = os.path.basename(model_name)

            # Get status for this model
            status = self.model_status.get(i, "Unknown")

            # Create color-coded display text
            if i == self.current_working_model_index:
                item_text = f"Model {i + 1}: {display_name} [WORKING]"
                item = QListWidgetItem(item_text)
                item.setForeground(Qt.blue)  # Working model in blue
            elif status == "passed":
                item_text = f"Model {i + 1}: {display_name} [PASSED]"
                item = QListWidgetItem(item_text)
                item.setForeground(Qt.darkGreen)  # Passed models in green
            elif status == "failed":
                item_text = f"Model {i + 1}: {display_name} [FAILED]"
                item = QListWidgetItem(item_text)
                item.setForeground(Qt.red)  # Failed models in red
            elif status == "skipped":
                item_text = f"Model {i + 1}: {display_name} [SKIPPED]"
                item = QListWidgetItem(item_text)
                item.setForeground(Qt.gray)  # Skipped models in gray
            elif status == "pending":
                item_text = f"Model {i + 1}: {display_name} [PENDING]"
                item = QListWidgetItem(item_text)
                item.setForeground(Qt.darkYellow)  # Pending models in yellow
            else:
                item_text = f"Model {i + 1}: {display_name} [{status}]"
                item = QListWidgetItem(item_text)

            self.model_status_list.addItem(item)

        # Make current working model visible
        if self.current_working_model_index >= 0 and self.current_working_model_index < len(self.model_names):
            self.model_status_list.scrollToItem(self.model_status_list.item(self.current_working_model_index))

    # Add this method to the Annotator class
    def remove_duplicate_images(self):
        """Detect and remove duplicate images based on a similarity threshold."""
        if not self.image_paths or len(self.image_paths) < 2:
            QMessageBox.information(self, "Not Enough Images",
                                    "Need at least 2 images to detect duplicates.")
            return

        # Check if required packages are installed
        try:
            import numpy as np
            from PIL import Image
            import imagehash
        except ImportError as e:
            missing_package = str(e).split("'")[1]
            QMessageBox.warning(self, "Missing Package",
                                f"Required package '{missing_package}' is not installed.\n\n"
                                f"Please install it using:\n"
                                f"pip install {missing_package}")
            return

        # Store current annotations
        self._store_current_annotations()

        # Create dialog for threshold selection
        dialog = QDialog(self)
        dialog.setWindowTitle("Duplicate Detection Settings")
        dialog.setMinimumWidth(400)
        layout = QVBoxLayout(dialog)

        # Add threshold slider
        layout.addWidget(QLabel("Similarity Threshold (lower = more strict):"))
        slider = QSlider(Qt.Horizontal)
        slider.setRange(0, 20)  # 0-20 range for hash difference
        slider.setValue(8)  # Default value
        slider.setTickPosition(QSlider.TicksBelow)
        slider.setTickInterval(1)

        slider_label = QLabel(f"Current: {slider.value()}")
        slider.valueChanged.connect(lambda v: slider_label.setText(f"Current: {v}"))

        layout.addWidget(slider)
        layout.addWidget(slider_label)
        layout.addWidget(QLabel("Lower values mean images must be more similar to be considered duplicates.\n"
                                "Higher values will detect more duplicates but may include false positives."))

        # Add buttons
        button_layout = QHBoxLayout()
        ok_button = QPushButton("OK")
        cancel_button = QPushButton("Cancel")
        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)
        layout.addLayout(button_layout)

        # Show dialog
        result = dialog.exec_()
        if result != QDialog.Accepted:
            return

        threshold = slider.value()

        # Create progress dialog
        progress = QProgressDialog(
            "Analyzing images for duplicates...", "Cancel", 0, len(self.image_paths), self)
        progress.setWindowTitle("Finding Duplicates")
        progress.setWindowModality(Qt.ApplicationModal)
        progress.setMinimumDuration(0)

        # Create worker
        self.similarity_worker = ImageSimilarityWorker(self.image_paths, threshold)
        self.similarity_worker.progress.connect(lambda d, t: progress.setValue(d))
        self.similarity_worker.result.connect(self._show_duplicate_selection)
        self.similarity_worker.error.connect(lambda e: self.statusBar().showMessage(f"Error: {e}"))
        progress.canceled.connect(self.similarity_worker.terminate)

        # Start worker
        self.similarity_worker.start()

    def _show_duplicate_selection(self, duplicates):
        """Show dialog for selecting which duplicates to remove."""
        if not duplicates:
            QMessageBox.information(self, "No Duplicates Found",
                                    "No duplicate images were found with the current threshold.")
            return

        # Create dialog for duplicate selection
        dialog = QDialog(self)
        dialog.setWindowTitle(f"Found {len(duplicates)} Duplicate Pairs")
        dialog.setMinimumWidth(800)
        dialog.setMinimumHeight(600)
        layout = QVBoxLayout(dialog)

        # Add info label
        layout.addWidget(QLabel(f"Found {len(duplicates)} duplicate image pairs. "
                                f"Select which images to remove from each pair:"))

        # Create table for duplicate selection
        table = QTableWidget(len(duplicates), 5)
        table.setHorizontalHeaderLabels(["Image 1", "Image 2", "Difference", "Remove 1", "Remove 2"])
        table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        table.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)

        # Add image preview area
        preview_layout = QHBoxLayout()
        preview1 = QLabel("Image 1 Preview")
        preview2 = QLabel("Image 2 Preview")
        preview1.setAlignment(Qt.AlignCenter)
        preview2.setAlignment(Qt.AlignCenter)
        preview1.setMinimumHeight(200)
        preview2.setMinimumHeight(200)
        preview_layout.addWidget(preview1)
        preview_layout.addWidget(preview2)

        # Add duplicate pairs to table
        for i, (path1, path2, diff) in enumerate(duplicates):
            # First image
            item1 = QTableWidgetItem(os.path.basename(path1))
            item1.setData(Qt.UserRole, path1)
            table.setItem(i, 0, item1)

            # Second image
            item2 = QTableWidgetItem(os.path.basename(path2))
            item2.setData(Qt.UserRole, path2)
            table.setItem(i, 1, item2)

            # Difference
            table.setItem(i, 2, QTableWidgetItem(str(diff)))

            # Remove checkboxes
            check1 = QTableWidgetItem()
            check1.setCheckState(Qt.Unchecked)
            table.setItem(i, 3, check1)

            check2 = QTableWidgetItem()
            check2.setCheckState(Qt.Unchecked)
            table.setItem(i, 4, check2)

        # Connect table selection to preview
        def update_preview():
            row = table.currentRow()
            if row >= 0:
                path1 = table.item(row, 0).data(Qt.UserRole)
                path2 = table.item(row, 1).data(Qt.UserRole)

                # Load and display images
                pixmap1 = QPixmap(path1)
                pixmap2 = QPixmap(path2)

                # Scale down if needed
                max_size = 200
                if pixmap1.width() > max_size or pixmap1.height() > max_size:
                    pixmap1 = pixmap1.scaled(max_size, max_size, Qt.KeepAspectRatio)

                if pixmap2.width() > max_size or pixmap2.height() > max_size:
                    pixmap2 = pixmap2.scaled(max_size, max_size, Qt.KeepAspectRatio)

                preview1.setPixmap(pixmap1)
                preview2.setPixmap(pixmap2)

        table.currentCellChanged.connect(lambda: update_preview())

        # Select first row by default to show first image pair
        if table.rowCount() > 0:
            table.selectRow(0)
            update_preview()

        layout.addWidget(table)
        layout.addLayout(preview_layout)

        # Add buttons
        button_layout = QHBoxLayout()

        select_first = QPushButton("Select All First Images")
        select_first.clicked.connect(lambda: self._select_duplicate_column(table, 3, Qt.Checked))

        select_second = QPushButton("Select All Second Images")
        select_second.clicked.connect(lambda: self._select_duplicate_column(table, 4, Qt.Checked))

        clear_selection = QPushButton("Clear Selection")
        clear_selection.clicked.connect(lambda: self._clear_duplicate_selection(table))

        ok_button = QPushButton("Remove Selected")
        cancel_button = QPushButton("Cancel")

        button_layout.addWidget(select_first)
        button_layout.addWidget(select_second)
        button_layout.addWidget(clear_selection)
        button_layout.addStretch()
        button_layout.addWidget(ok_button)
        button_layout.addWidget(cancel_button)

        layout.addLayout(button_layout)

        ok_button.clicked.connect(dialog.accept)
        cancel_button.clicked.connect(dialog.reject)

        # Show dialog
        result = dialog.exec_()
        if result != QDialog.Accepted:
            return

        # Process selected duplicates
        to_remove = set()
        for i in range(table.rowCount()):
            # Check if first image is selected for removal
            if table.item(i, 3).checkState() == Qt.Checked:
                path = table.item(i, 0).data(Qt.UserRole)
                to_remove.add(path)

            # Check if second image is selected for removal
            if table.item(i, 4).checkState() == Qt.Checked:
                path = table.item(i, 1).data(Qt.UserRole)
                to_remove.add(path)

        if not to_remove:
            QMessageBox.information(self, "No Images Selected",
                                    "No images were selected for removal.")
            return

        # Confirm removal
        confirm = QMessageBox.question(
            self, "Confirm Removal",
            f"Remove {len(to_remove)} duplicate images?",
            QMessageBox.Yes | QMessageBox.No)

        if confirm != QMessageBox.Yes:
            return

        # Remove duplicates
        self._remove_duplicate_images(to_remove)

    def _select_duplicate_column(self, table, column, state):
        """Select all checkboxes in a column."""
        for i in range(table.rowCount()):
            table.item(i, column).setCheckState(state)

    def _clear_duplicate_selection(self, table):
        """Clear all selection checkboxes."""
        for i in range(table.rowCount()):
            table.item(i, 3).setCheckState(Qt.Unchecked)
            table.item(i, 4).setCheckState(Qt.Unchecked)

    def _remove_duplicate_images(self, paths_to_remove):
        """Remove the selected duplicate images."""
        if not paths_to_remove:
            return

        # Store the current image
        current_path = self.image_path

        # Create a list of kept images
        kept_images = []
        removed_count = 0

        for path in self.image_paths:
            if path in paths_to_remove:
                # Remove from annotation store
                if path in self.annotations_store:
                    del self.annotations_store[path]
                removed_count += 1
            else:
                kept_images.append(path)

        # Update paths list
        self.image_paths = kept_images

        # Handle case when there are no more images
        if not self.image_paths:
            self.curr_index = -1
            self.image_path = None
            self.scene.clear()
            self.annotations.clear()
            self.list_widget.clear()
            self.image_spin.setEnabled(False)
            self.statusBar().showMessage(f"Removed {removed_count} duplicate images. No images remaining.")
            return

        # Try to find the previous current image in the new list
        if current_path in self.image_paths:
            self.curr_index = self.image_paths.index(current_path)
        else:
            # If removed, go to the first image
            self.curr_index = 0

        # Load current image
        self.load_current_image()

        # Show confirmation
        self.statusBar().showMessage(f"Removed {removed_count} duplicate images.")


# ────────────────────────── bootstrap ───────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv)
    gui = Annotator();
    gui.show()
    sys.exit(app.exec())
