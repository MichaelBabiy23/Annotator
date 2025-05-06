import tkinter as tk
from tkinter import filedialog, messagebox
import os

def remove_duplicates_in_file(filepath):
    """
    Read the file at `filepath`, remove duplicate lines while preserving order,
    and overwrite the file with the unique lines.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    seen = set()
    unique_lines = []
    for line in lines:
        if line not in seen:
            unique_lines.append(line)
            seen.add(line)
    with open(filepath, 'w', encoding='utf-8') as f:
        f.writelines(unique_lines)


def process_folder():
    """
    Open a dialog to select a folder, then process each file in the folder
    (and its subfolders), removing duplicate lines in-place.
    """
    folder = filedialog.askdirectory(title="Select Folder to Process")
    if not folder:
        return
    for root, dirs, files in os.walk(folder):
        for filename in files:
            filepath = os.path.join(root, filename)
            try:
                remove_duplicates_in_file(filepath)
            except Exception as e:
                print(f"Skipping {filepath}: {e}")
    messagebox.showinfo("Done", "Duplicate lines removed from all files in the selected folder.")


def main():
    root = tk.Tk()
    root.title("Remove Duplicate Lines GUI")
    root.geometry("300x100")
    btn = tk.Button(root, text="Select Folder", command=process_folder)
    btn.pack(expand=True, pady=20)
    root.mainloop()


if __name__ == '__main__':
    main()
