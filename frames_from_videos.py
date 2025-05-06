import sys
import os
import cv2
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QPushButton, QLabel,
    QFileDialog, QVBoxLayout, QHBoxLayout, QWidget,
    QSpinBox, QProgressBar, QListWidget, QListWidgetItem,
    QMessageBox
)
from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QFont


class FrameExtractorThread(QThread):
    # Emit per-frame progress: current_frame, total_frames, base_name
    progress_updated = Signal(int, int, str)
    extraction_completed = Signal(str)  # base_name when done
    error_occurred = Signal(str)

    def __init__(self, video_path, output_dir, interval):
        super().__init__()
        self.video_path = video_path
        self.output_dir = output_dir
        self.interval = interval
        self.is_cancelled = False

    def run(self):
        try:
            base_name = os.path.splitext(os.path.basename(self.video_path))[0]
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.error_occurred.emit(f"Cannot open video: {self.video_path}")
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            frame_count = 0

            while frame_count < total_frames:
                if self.is_cancelled:
                    cap.release()
                    return

                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % self.interval == 0:
                    frame_name = f"{base_name}_frame_{frame_count:06d}.jpg"
                    cv2.imwrite(os.path.join(self.output_dir, frame_name), frame)

                frame_count += 1
                self.progress_updated.emit(frame_count, total_frames, base_name)

            cap.release()
            self.extraction_completed.emit(base_name)

        except Exception as e:
            self.error_occurred.emit(str(e))

    def cancel(self):
        self.is_cancelled = True


class FrameExtractorApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Video Frame Extractor")
        self.setMinimumSize(600, 600)

        self.file_paths = []
        self.output_dir = None
        self.threads = []
        self.completed_count = 0
        self.file_progress_bars = {}

        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        main_layout = QVBoxLayout(main_widget)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_label = QLabel("Video Frame Extractor")
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(title_label)

        # Interval selection
        interval_layout = QHBoxLayout()
        interval_label = QLabel("Extract every:")
        interval_layout.addWidget(interval_label)

        self.interval_spinbox = QSpinBox()
        self.interval_spinbox.setRange(1, 1000)
        self.interval_spinbox.setValue(30)
        self.interval_spinbox.setSuffix(" frames")
        interval_layout.addWidget(self.interval_spinbox)
        interval_layout.addStretch()
        main_layout.addLayout(interval_layout)

        # File and output folder selection
        file_btn_layout = QHBoxLayout()
        self.select_btn = QPushButton("Select Video Files")
        self.select_btn.setMinimumHeight(40)
        self.select_btn.clicked.connect(self.select_files)
        file_btn_layout.addWidget(self.select_btn)

        self.output_btn = QPushButton("Select Output Folder")
        self.output_btn.setMinimumHeight(40)
        self.output_btn.clicked.connect(self.select_output_folder)
        file_btn_layout.addWidget(self.output_btn)
        main_layout.addLayout(file_btn_layout)

        # File list with per-file progress bars
        self.file_list = QListWidget()
        self.file_list.setAlternatingRowColors(True)
        self.file_list.setMinimumHeight(200)
        main_layout.addWidget(self.file_list)

        # Status label
        self.current_file_label = QLabel("Ready")
        main_layout.addWidget(self.current_file_label)

        # Action buttons
        action_layout = QHBoxLayout()
        self.extract_btn = QPushButton("Extract Frames")
        self.extract_btn.setMinimumHeight(50)
        self.extract_btn.clicked.connect(self.extract_frames)
        action_layout.addWidget(self.extract_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setMinimumHeight(50)
        self.cancel_btn.clicked.connect(self.cancel_extraction)
        self.cancel_btn.setEnabled(False)
        action_layout.addWidget(self.cancel_btn)
        main_layout.addLayout(action_layout)

        self.setCentralWidget(main_widget)

    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        if files:
            self.file_paths = files
            self.file_list.clear()
            self.file_progress_bars.clear()

            for file in files:
                base_name = os.path.splitext(os.path.basename(file))[0]
                item = QListWidgetItem()
                widget = QWidget()
                layout = QHBoxLayout(widget)
                layout.setContentsMargins(5, 2, 5, 2)

                name_label = QLabel(base_name)
                progress_bar = QProgressBar()
                progress_bar.setRange(0, 1)
                progress_bar.setValue(0)
                progress_bar.setFormat("%p%%")

                layout.addWidget(name_label)
                layout.addWidget(progress_bar)

                self.file_list.addItem(item)
                self.file_list.setItemWidget(item, widget)
                self.file_progress_bars[base_name] = progress_bar

    def select_output_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.output_dir = folder
            self.current_file_label.setText(f"Output: {folder}")

    def extract_frames(self):
        if not self.file_paths:
            QMessageBox.warning(self, "No Files", "Please select at least one video file.")
            return
        if not self.output_dir:
            QMessageBox.warning(self, "No Output Folder", "Please select an output folder.")
            return

        interval = self.interval_spinbox.value()
        os.makedirs(self.output_dir, exist_ok=True)

        # Disable UI
        self.extract_btn.setEnabled(False)
        self.select_btn.setEnabled(False)
        self.output_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)

        # Reset state
        self.completed_count = 0
        self.threads = []

        # Launch one thread per file
        for video_path in self.file_paths:
            thread = FrameExtractorThread(video_path, self.output_dir, interval)
            thread.progress_updated.connect(self.update_file_progress)
            thread.error_occurred.connect(self.handle_error)
            thread.extraction_completed.connect(self.on_thread_done)
            self.threads.append(thread)
            thread.start()

    @Slot(int, int, str)
    def update_file_progress(self, current, total, base_name):
        bar = self.file_progress_bars.get(base_name)
        if bar:
            bar.setMaximum(total)
            bar.setValue(current)
            self.current_file_label.setText(f"{base_name}: {current}/{total} frames")

    @Slot(str)
    def on_thread_done(self, base_name):
        self.completed_count += 1
        # Mark completion bar at 100%
        bar = self.file_progress_bars.get(base_name)
        if bar:
            bar.setValue(bar.maximum())

        if self.completed_count == len(self.file_paths):
            self.current_file_label.setText("All extractions completed!")
            self.reset_ui_state()
            QMessageBox.information(self, "Extraction Complete",
                                    "All frames have been successfully extracted!")

    def cancel_extraction(self):
        for thread in self.threads:
            if thread.isRunning():
                thread.cancel()
        self.cancel_btn.setEnabled(False)
        self.current_file_label.setText("Cancelling...")

    @Slot(str)
    def handle_error(self, error_message):
        self.reset_ui_state()
        QMessageBox.critical(self, "Error", f"An error occurred: {error_message}")

    def reset_ui_state(self):
        self.extract_btn.setEnabled(True)
        self.select_btn.setEnabled(True)
        self.output_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)

    def closeEvent(self, event):
        for thread in self.threads:
            if thread.isRunning():
                thread.cancel()
                thread.wait()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = FrameExtractorApp()
    window.show()
    sys.exit(app.exec())
