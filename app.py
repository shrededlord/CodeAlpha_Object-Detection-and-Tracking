import tkinter as tk
from tkinter import filedialog
from detector import process_video

def browse_file():
    filename = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4;*.avi")])
    if filename:
        process_video(filename)

root = tk.Tk()
root.title("Video Selection")

browse_button = tk.Button(
    root, text="Browse", command=browse_file, height=2, width=20, bg="light blue"
)
browse_button.pack(pady=10)

root.mainloop()
