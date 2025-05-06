# save_worker.py
import os
from PySide6.QtCore import QThread, Signal, QObject
from PySide6.QtGui  import QImage
import concurrent.futures
import multiprocessing

class SaveWorker(QThread):
    """Background thread to write YOLO-format .txt files for a list of images."""
    progress    = Signal(int, int)   # done, total
    finished_ok = Signal(str)        # dstDir
    error       = Signal(str)

    def __init__(self, image_paths, annotations_store, dst_dir):
        super().__init__()
        self.image_paths       = image_paths
        self.annotations_store = annotations_store
        self.dst_dir           = dst_dir
        # Use half of available CPU cores, but at least 1 and at most 8
        self.num_workers = max(1, min(8, multiprocessing.cpu_count() // 2))

    def run(self):
        try:
            total = len(self.image_paths)
            
            # Process images in chunks to avoid creating too many threads
            if total <= 20 or self.num_workers <= 1:
                # For small batches, use the original sequential method
                self._process_images_sequential()
            else:
                # For larger batches, use parallel processing
                self._process_images_parallel()
                
            self.finished_ok.emit(self.dst_dir)
        except Exception as e:
            self.error.emit(str(e))
            
    def _process_images_sequential(self):
        """Process images sequentially (original method)."""
        total = len(self.image_paths)
        for idx, img_path in enumerate(self.image_paths, 1):
            self._save_single_annotation(img_path)
            self.progress.emit(idx, total)
            
    def _process_images_parallel(self):
        """Process images in parallel using ThreadPoolExecutor."""
        total = len(self.image_paths)
        completed = 0
        
        # Create batches of images to process
        batch_size = max(1, total // (self.num_workers * 2))
        batches = [self.image_paths[i:i+batch_size] for i in range(0, total, batch_size)]
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            # Submit batches for processing
            future_to_batch = {
                executor.submit(self._process_batch, batch): batch 
                for batch in batches
            }
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_batch):
                batch = future_to_batch[future]
                try:
                    future.result()  # Check for exceptions
                    completed += len(batch)
                    self.progress.emit(completed, total)
                except Exception as e:
                    self.error.emit(f"Error processing batch: {str(e)}")
                    executor.shutdown(wait=False, cancel_futures=True)
                    raise
    
    def _process_batch(self, batch):
        """Process a batch of images."""
        for img_path in batch:
            self._save_single_annotation(img_path)
    
    def _save_single_annotation(self, img_path):
        """Save annotation for a single image."""
        img = QImage(img_path)
        if img.isNull():
            raise RuntimeError(f"Cannot read {img_path}")
        w_img, h_img = img.width(), img.height()

        txt_name = os.path.splitext(os.path.basename(img_path))[0] + ".txt"
        txt_path = os.path.join(self.dst_dir, txt_name)

        with open(txt_path, "w") as f:
            for cls, conf, x, y, w_rect, h_rect in self.annotations_store.get(img_path, []):
                cx, cy = (x + w_rect/2)/w_img, (y + h_rect/2)/h_img
                f.write(f"{cls} {cx:.6f} {cy:.6f} "
                        f"{w_rect/w_img:.6f} {h_rect/h_img:.6f}\n")
