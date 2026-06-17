import tkinter as tk
from tkinter import filedialog, messagebox
import subprocess
import os

def select_image():
    filepath = filedialog.askopenfilename(
        title="選擇圖片",
        filetypes=[("圖片", "*.jpg *.jpeg *.png *.jfif"), ("All files", "*.*")]
    )
    if filepath:
        try:
            rel_path = os.path.relpath(filepath, os.getcwd())
            img_entry.delete(0, tk.END)
            img_entry.insert(0, rel_path)
        except Exception:
            img_entry.delete(0, tk.END)
            img_entry.insert(0, filepath)

def select_video():
    filepath = filedialog.askopenfilename(
        title="選擇影片",
        filetypes=[("影片", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
    )
    if filepath:
        try:
            rel_path = os.path.relpath(filepath, os.getcwd())
            vid_entry.delete(0, tk.END)
            vid_entry.insert(0, rel_path)
        except Exception:
            vid_entry.delete(0, tk.END)
            vid_entry.insert(0, filepath)

def run_image():
    filepath = img_entry.get().strip()
    if not filepath:
        messagebox.showwarning("警告", "請先選擇圖片！")
        return
    run_script("process_single_image.py", filepath)

def run_video():
    filepath = vid_entry.get().strip()
    if not filepath:
        messagebox.showwarning("警告", "請先選擇影片！")
        return
    run_script("process_enhanced_video.py", filepath)

def run_script(script, filepath):
    print(f"Executing: python3 {script} {filepath}")
    btn_run_img.config(state=tk.DISABLED)
    btn_run_vid.config(state=tk.DISABLED)
    root.update()
    
    try:
        subprocess.run(["python3", script, filepath], check=True)
        messagebox.showinfo("完成", f"處理完成！\n呼叫的腳本：{script}")
    except subprocess.CalledProcessError as e:
        messagebox.showerror("錯誤", f"執行失敗：\n{e}")
    finally:
        btn_run_img.config(state=tk.NORMAL)
        btn_run_vid.config(state=tk.NORMAL)

root = tk.Tk()
root.title("Fire & Smoke Detection Launcher")
root.geometry("550x250")

# Image Section
tk.Label(root, text="--- 圖片偵測 ---", font=("Arial", 10, "bold")).pack(pady=(10,0))
frame_img = tk.Frame(root)
frame_img.pack(pady=5)
img_entry = tk.Entry(frame_img, width=40, font=("Arial", 10))
img_entry.pack(side=tk.LEFT, padx=5)
img_entry.insert(0, "Deployment/assets/croatia_fire_dataset/cro_data_0.jpg")
tk.Button(frame_img, text="瀏覽圖片", command=select_image).pack(side=tk.LEFT, padx=2)
btn_run_img = tk.Button(frame_img, text="執行圖片", command=run_image, bg="#4CAF50", fg="white", font=("Arial", 10, "bold"))
btn_run_img.pack(side=tk.LEFT, padx=5)

# Video Section
tk.Label(root, text="--- 影片偵測 ---", font=("Arial", 10, "bold")).pack(pady=(15,0))
frame_vid = tk.Frame(root)
frame_vid.pack(pady=5)
vid_entry = tk.Entry(frame_vid, width=40, font=("Arial", 10))
vid_entry.pack(side=tk.LEFT, padx=5)
vid_entry.insert(0, "assets/videos/roomfire41.mp4")
tk.Button(frame_vid, text="瀏覽影片", command=select_video).pack(side=tk.LEFT, padx=2)
btn_run_vid = tk.Button(frame_vid, text="執行影片", command=run_video, bg="#2196F3", fg="white", font=("Arial", 10, "bold"))
btn_run_vid.pack(side=tk.LEFT, padx=5)

root.mainloop()
