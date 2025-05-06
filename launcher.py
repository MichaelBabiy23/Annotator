#!/usr/bin/env python3
"""
YOLO Annotation Tool Launcher
A simple GUI launcher that helps install and run the annotation tool.
"""

import os
import sys
import subprocess
import platform
import threading
import tkinter as tk
from tkinter import ttk, messagebox

class LauncherApp:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Annotation Tool Launcher")
        self.root.geometry("500x400")
        self.root.resizable(False, False)
        
        # Set icon if available
        try:
            if platform.system() == "Windows":
                self.root.iconbitmap("icon.ico")
            else:
                logo = tk.PhotoImage(file="icon.png")
                self.root.iconphoto(True, logo)
        except:
            pass  # Icon not found, continue without it
        
        self.setup_ui()
        self.check_env()
        
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="YOLO Annotation Tool",
            font=("Helvetica", 16, "bold")
        )
        title_label.pack(pady=(0, 10))
        
        # Status frame
        status_frame = ttk.LabelFrame(main_frame, text="Environment Status", padding="10")
        status_frame.pack(fill=tk.X, pady=10)
        
        # Status indicators
        self.python_status = ttk.Label(status_frame, text="Python: Checking...")
        self.python_status.pack(anchor=tk.W)
        
        self.venv_status = ttk.Label(status_frame, text="Virtual Environment: Checking...")
        self.venv_status.pack(anchor=tk.W)
        
        self.deps_status = ttk.Label(status_frame, text="Dependencies: Checking...")
        self.deps_status.pack(anchor=tk.W)
        
        # Buttons frame
        btn_frame = ttk.Frame(main_frame)
        btn_frame.pack(fill=tk.X, pady=20)
        
        # Install button
        self.install_btn = ttk.Button(
            btn_frame, 
            text="Install",
            command=self.install,
            state=tk.DISABLED
        )
        self.install_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Run button
        self.run_btn = ttk.Button(
            btn_frame, 
            text="Run",
            command=self.run,
            state=tk.DISABLED
        )
        self.run_btn.pack(side=tk.LEFT, padx=5, expand=True, fill=tk.X)
        
        # Progress and output
        self.progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.progress.pack(fill=tk.X, pady=10)
        
        # Output console
        console_frame = ttk.LabelFrame(main_frame, text="Output", padding="10")
        console_frame.pack(fill=tk.BOTH, expand=True)
        
        self.console = tk.Text(console_frame, height=8, wrap=tk.WORD, bg="#f0f0f0")
        self.console.pack(fill=tk.BOTH, expand=True)
        self.console.config(state=tk.DISABLED)
        
    def log(self, message, end="\n"):
        """Add message to console"""
        self.console.config(state=tk.NORMAL)
        self.console.insert(tk.END, message + end)
        self.console.see(tk.END)
        self.console.config(state=tk.DISABLED)
        self.root.update()
        
    def check_env(self):
        """Check environment status in a separate thread"""
        self.progress.start()
        threading.Thread(target=self._check_env_thread, daemon=True).start()
        
    def _check_env_thread(self):
        """Background thread for checking environment"""
        # Check Python
        try:
            if platform.system() == "Windows":
                python_version = subprocess.check_output(["python", "--version"], stderr=subprocess.STDOUT, text=True)
            else:
                python_version = subprocess.check_output(["python3", "--version"], stderr=subprocess.STDOUT, text=True)
            
            self.python_status.config(text=f"Python: {python_version.strip()}", foreground="green")
            self.log(f"Found {python_version.strip()}")
            python_ok = True
        except:
            self.python_status.config(text="Python: Not found", foreground="red")
            self.log("Python not found. Please install Python 3.6 or higher")
            python_ok = False
            
        # Check virtual environment
        venv_exists = os.path.exists("venv")
        if venv_exists:
            self.venv_status.config(text="Virtual Environment: Found", foreground="green")
            self.log("Virtual environment found")
        else:
            self.venv_status.config(text="Virtual Environment: Not found", foreground="orange")
            self.log("Virtual environment not found. Installation needed.")
            
        # Check dependencies
        if venv_exists:
            try:
                # Check if PySide6 is installed in the virtual environment
                if platform.system() == "Windows":
                    cmd = ["venv\\Scripts\\python", "-c", "import PySide6; print('PySide6 found')"]
                else:
                    cmd = ["./venv/bin/python3", "-c", "import PySide6; print('PySide6 found')"]
                    
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                if "PySide6 found" in result.stdout:
                    self.deps_status.config(text="Dependencies: Installed", foreground="green")
                    self.log("Dependencies are installed")
                    deps_ok = True
                else:
                    self.deps_status.config(text="Dependencies: Missing", foreground="orange")
                    self.log("Dependencies not fully installed")
                    deps_ok = False
            except:
                self.deps_status.config(text="Dependencies: Check failed", foreground="orange")
                self.log("Failed to check dependencies")
                deps_ok = False
        else:
            self.deps_status.config(text="Dependencies: Not installed", foreground="orange")
            self.log("Dependencies need to be installed")
            deps_ok = False
            
        # Update button states
        self.root.after(0, self._update_buttons, python_ok, venv_exists, deps_ok)
        self.progress.stop()
        
    def _update_buttons(self, python_ok, venv_exists, deps_ok):
        """Update button states based on environment check"""
        if python_ok:
            self.install_btn.config(state=tk.NORMAL)
            
        if venv_exists and deps_ok:
            self.run_btn.config(state=tk.NORMAL)
            
    def install(self):
        """Run installation script"""
        self.progress.start()
        self.install_btn.config(state=tk.DISABLED)
        self.run_btn.config(state=tk.DISABLED)
        self.log("\n--- Starting installation ---")
        threading.Thread(target=self._install_thread, daemon=True).start()
        
    def _install_thread(self):
        """Background thread for installation"""
        try:
            if platform.system() == "Windows":
                process = subprocess.Popen(
                    ["install.bat"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
            else:
                process = subprocess.Popen(
                    ["./install.sh"], 
                    stdout=subprocess.PIPE, 
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )
                
            # Stream output
            for line in iter(process.stdout.readline, ""):
                if line:
                    self.log(line.strip())
            
            process.wait()
            
            if process.returncode == 0:
                self.log("\nInstallation completed successfully!")
                self.root.after(0, lambda: self.check_env())
            else:
                self.log(f"\nInstallation failed with return code {process.returncode}")
                
        except Exception as e:
            self.log(f"\nError during installation: {str(e)}")
        finally:
            self.progress.stop()
            self.root.after(0, lambda: self.install_btn.config(state=tk.NORMAL))
    
    def run(self):
        """Run the annotation tool"""
        self.progress.start()
        self.log("\n--- Starting YOLO Annotation Tool ---")
        threading.Thread(target=self._run_thread, daemon=True).start()
        
    def _run_thread(self):
        """Background thread for running the application"""
        try:
            if platform.system() == "Windows":
                subprocess.Popen(["run.bat"])
            else:
                subprocess.Popen(["./run.sh"])
                
            self.log("Application launched!")
        except Exception as e:
            self.log(f"Error launching application: {str(e)}")
        finally:
            self.progress.stop()
            
if __name__ == "__main__":
    root = tk.Tk()
    app = LauncherApp(root)
    root.mainloop() 