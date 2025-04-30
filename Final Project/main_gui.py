import tkinter as tk
from tkinter import messagebox
import threading
import video_mode
import camera_mode

root = tk.Tk()
root.title("People Counter")
root.geometry("400x200")

status_label = tk.Label(root, text="Status: Waiting...", font=("Arial", 14))
status_label.pack(pady=20)

# Global state
thread_handle = None
stop_event = threading.Event()

def update_status(text):
    status_label.config(text=f"Status: {text}")

def stop_running_thread():
    global thread_handle, stop_event
    if thread_handle and thread_handle.is_alive():
        stop_event.set()
        thread_handle.join()
    stop_event.clear()

def start_video_mode():
    global thread_handle
    stop_running_thread()
    update_status("Running video mode...")
    thread_handle = threading.Thread(target=video_mode.run_video_mode, args=(update_status, stop_event), daemon=True)
    thread_handle.start()

def start_camera_mode():
    global thread_handle
    stop_running_thread()
    update_status("Running camera mode...")
    thread_handle = threading.Thread(target=camera_mode.run_video_mode, args=(update_status, stop_event), daemon=True)
    thread_handle.start()

def quit_running_mode():
    stop_running_thread()
    update_status("Stopped")

# Buttons
tk.Button(root, text="Load from Video", command=start_video_mode, width=20).pack(pady=5)
tk.Button(root, text="Load from Camera", command=start_camera_mode, width=20).pack(pady=5)
tk.Button(root, text="Quit Video/Camera", command=quit_running_mode, width=20).pack(pady=5)

root.mainloop()
