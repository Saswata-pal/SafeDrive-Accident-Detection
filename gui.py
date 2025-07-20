import cv2
import pandas as pd
from ultralytics import YOLO
import cvzone
import numpy as np
from twilio.rest import Client
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os, json
import cloudinary
import cloudinary.uploader
import time
import threading
from dotenv import load_dotenv
import os

# ==============================
# CONFIGURATIONS
# ==============================

load_dotenv()

cloudinary.config(
    cloud_name=os.getenv("CLOUD_NAME"),
    api_key=os.getenv("CLOUD_API_KEY"),
    api_secret=os.getenv("CLOUD_API_SECRET")
)

account_sid = os.getenv("TWILIO_ACCOUNT_SID")
auth_token = os.getenv("TWILIO_AUTH_TOKEN")
twilio_phone_number = os.getenv("TWILIO_PHONE_NUMBER")

client = Client(account_sid, auth_token)


#  YOLO Model
model = YOLO('best1.pt')

#  CCTV registry JSON
CCTV_REGISTRY_FILE = "cctv_registry.json"

def load_cctv_registry():
    if os.path.exists(CCTV_REGISTRY_FILE):
        with open(CCTV_REGISTRY_FILE, "r") as f:
            return json.load(f)
    return {}

def save_cctv_registry(data):
    with open(CCTV_REGISTRY_FILE, "w") as f:
        json.dump(data, f, indent=4)

# Load current CCTV data
cctv_registry = load_cctv_registry()

# ==============================
#  CLOUDINARY + SMS FUNCTIONS
# ==============================
def upload_image_to_cloudinary(image_path):
    try:
        result = cloudinary.uploader.upload(image_path)
        return result.get('secure_url')
    except Exception as e:
        print("❌ Cloudinary upload failed:", e)
        return None

def broadcast_sms(cctv_id, image_url=None):
    """Send SMS to all police + hospitals registered for CCTV"""
    if cctv_id not in cctv_registry:
        print(f"❌ CCTV {cctv_id} not found in registry!")
        return

    details = cctv_registry[cctv_id]
    location = details.get("location", "Unknown Location")
    contacts = details.get("contacts", [])

    if not contacts:
        print(f"⚠️ No contacts registered for CCTV {cctv_id}")
        return

    # First alert (before uploading image)
    base_msg = f"🚨 Accident Detected!\n📍 CCTV: {cctv_id} ({location})\nAuthorities notified."
    for number in contacts:
        msg = client.messages.create(
            body=base_msg,
            from_=twilio_phone_number,
            to=number
        )
        print(f"✅ First SMS sent to {number}: {msg.sid}")

    # Send image URL separately after upload
    if image_url:
        for number in contacts:
            try:
                msg = client.messages.create(
                    body=image_url,  
                    from_=twilio_phone_number,
                    to=number
                )
                print(f"✅ Image link SMS sent to {number}: {msg.sid}")
            except Exception as e:
                print(f"❌ Error sending image link to {number}: {e}")


def process_alert(frame, cctv_id):
    # Save snapshot
    if not os.path.exists("accidents"):
        os.makedirs("accidents")
    filename = f"accidents/accident_frame_{len(os.listdir('accidents'))+1}.png"
    cv2.imwrite(filename, frame)
    print(f"✅ Accident frame saved as {filename}")

    # Immediate SMS (location only)
    broadcast_sms(cctv_id)

    # Async upload + follow-up SMS with image
    def upload_and_notify():
        image_url = upload_image_to_cloudinary(filename)
        if image_url:
            print(f"✅ Uploaded to Cloudinary: {image_url}")
            broadcast_sms(cctv_id, image_url=image_url)

    threading.Thread(target=upload_and_notify, daemon=True).start()

# ==============================
#  PROCESS VIDEO
# ==============================
def process_video(video_path, cctv_id):
    cap = cv2.VideoCapture(video_path)
    with open("coco1.txt", "r") as my_file:
        class_list = my_file.read().split("\n")

    count = 0
    accident_detected = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("✅ Video finished.")
            break

        count += 1
        if count % 5 != 0:  # speed optimization
            continue

        frame = cv2.resize(frame, (1020, 500))
        results = model.predict(frame, verbose=False)
        detections = results[0].boxes.data
        px = pd.DataFrame(detections).astype("float")

        for _, row in px.iterrows():
            x1, y1, x2, y2, _, cls_id = map(int, row[:6])
            c = class_list[cls_id]

            if c == "accident":
                color = (0, 0, 255)
                if not accident_detected:
                    accident_detected = True
                    threading.Thread(target=process_alert, args=(frame.copy(), cctv_id)).start()
            else:
                color = (0, 255, 0)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cvzone.putTextRect(frame, c, (x1, y1), 1, 1)

        cv2.imshow("Accident Detection (Demo Mode)", frame)

        # ESC to quit
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()

# ==============================
#  CCTV REGISTRY GUI
# ==============================
def manage_cctv_registry():
    reg_win = tk.Toplevel()
    reg_win.title("Manage CCTV Registry")
    reg_win.geometry("400x400")
    reg_win.resizable(False, False)

    tk.Label(reg_win, text="CCTV ID:").pack()
    entry_id = tk.Entry(reg_win)
    entry_id.pack()

    tk.Label(reg_win, text="Location:").pack()
    entry_loc = tk.Entry(reg_win)
    entry_loc.pack()

    tk.Label(reg_win, text="Contacts (comma separated):").pack()
    entry_contacts = tk.Entry(reg_win)
    entry_contacts.pack()

    def save_cctv():
        cctv_id = entry_id.get().strip()
        loc = entry_loc.get().strip()
        contacts = [x.strip() for x in entry_contacts.get().split(",") if x.strip()]

        if not cctv_id or not loc or not contacts:
            messagebox.showerror("Error", "All fields required!")
            return

        cctv_registry[cctv_id] = {
            "location": loc,
            "contacts": contacts
        }
        save_cctv_registry(cctv_registry)
        messagebox.showinfo("Saved", f"CCTV {cctv_id} registered successfully!")
        reg_win.destroy()

    tk.Button(reg_win, text="Save CCTV", command=save_cctv).pack(pady=10)

    # Show existing registry
    tk.Label(reg_win, text="Current Registered CCTVs:").pack(pady=5)
    txt = tk.Text(reg_win, height=10, width=40)
    txt.pack()
    for cid, info in cctv_registry.items():
        txt.insert(tk.END, f"{cid}: {info['location']} ({len(info['contacts'])} contacts)\n")

# ==============================
#  GUI UPLOAD VIDEO
# ==============================
def upload_video():
    if not cctv_registry:
        messagebox.showerror("Error", "No CCTV registered! Add one first.")
        return

    select_win = tk.Toplevel()
    select_win.title("Select CCTV for Video")
    select_win.geometry("300x200")

    tk.Label(select_win, text="Select CCTV:").pack(pady=5)
    cctv_var = tk.StringVar(value=list(cctv_registry.keys())[0])
    dropdown = ttk.Combobox(select_win, textvariable=cctv_var, values=list(cctv_registry.keys()))
    dropdown.pack()

    def choose_and_process():
        cctv_id = cctv_var.get()
        video_path = filedialog.askopenfilename(filetypes=[("MP4 files", "*.mp4")])
        if video_path:
            messagebox.showinfo("Processing", f"Video linked to CCTV {cctv_id} ({cctv_registry[cctv_id]['location']})")
            process_video(video_path, cctv_id)
        select_win.destroy()

    tk.Button(select_win, text="Upload Video", command=choose_and_process).pack(pady=10)

# ==============================
#  VIEW SNAPSHOTS
# ==============================
def view_images():
    folder = "accidents"
    if not os.path.exists(folder):
        messagebox.showwarning("Warning", "No accident images found!")
        return

    image_files = [f for f in os.listdir(folder) if f.endswith('.png')]
    if not image_files:
        messagebox.showwarning("Warning", "No accident images found!")
        return

    img_window = tk.Toplevel()
    img_window.title("Accident Snapshots")

    img_label = tk.Label(img_window)
    img_label.pack()

    def show_image(index):
        img_path = os.path.join(folder, image_files[index])
        img = Image.open(img_path).resize((400, 300), Image.Resampling.LANCZOS)
        img_tk = ImageTk.PhotoImage(img)
        img_label.config(image=img_tk)
        img_label.image = img_tk

    current_img = [0]
    show_image(0)

    ttk.Button(img_window, text="Next", command=lambda: (current_img.__setitem__(0, (current_img[0] + 1) % len(image_files)), show_image(current_img[0]))).pack(side=tk.RIGHT)
    ttk.Button(img_window, text="Previous", command=lambda: (current_img.__setitem__(0, (current_img[0] - 1) % len(image_files)), show_image(current_img[0]))).pack(side=tk.LEFT)

# ==============================
#  MAIN GUI
# ==============================
def create_gui():
    root = tk.Tk()
    #root.attributes("-topmost", True)  
    root.title("Accident Detection Demo")
    root.geometry("600x400")
    root.resizable(False, False)

    # def keep_all_windows_on_top():
    #     root.lift()
    #     root.attributes("-topmost", True)
    #     for window in root.winfo_children():
    #         if isinstance(window, tk.Toplevel):
    #             window.lift()
    #             window.attributes("-topmost", True)
    #     root.after(1000, keep_all_windows_on_top)

    # keep_all_windows_on_top()


    # Background image
    if os.path.exists("appArt.jpeg"):
        bg_img = Image.open("appArt.jpeg").resize((600, 400), Image.Resampling.LANCZOS)
        bg_photo = ImageTk.PhotoImage(bg_img)
        canvas = tk.Canvas(root, width=600, height=400)
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, image=bg_photo, anchor="nw")
    else:
        canvas = tk.Canvas(root, width=600, height=400)
        canvas.pack(fill="both", expand=True)

    label_quote = ttk.Label(root, text='"Speed thrills but kills"', font=("Helvetica", 16), background="white")
    canvas.create_window(300, 50, window=label_quote)

    ttk.Button(root, text="Upload Demo Video", command=upload_video).place(x=225, y=150)
    ttk.Button(root, text="View Accident Images", command=view_images).place(x=220, y=200)
    ttk.Button(root, text="Manage CCTV Registry", command=manage_cctv_registry).place(x=220, y=250)
    ttk.Button(root, text="Exit", command=root.quit).place(x=240, y=300)

    root.mainloop()

#  Run GUI
create_gui()
