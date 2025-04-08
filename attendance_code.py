import tkinter as tk
from tkinter import messagebox
import cv2
import face_recognition
import numpy as np
import os
from datetime import datetime

# Load known images
path = 'images'
images = []
classNames = []
myList = os.listdir(path)

for cl in myList:
    img = cv2.imread(f'{path}/{cl}')
    images.append(img)
    classNames.append(os.path.splitext(cl)[0])

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)
        if encode:
            encodeList.append(encode[0])
    return encodeList

encodeListKnown = findEncodings(images)
print(f"✅ Encodings complete for {len(encodeListKnown)} faces")

def markAttendance(name):
    if not os.path.exists('attendance.csv'):
        with open('attendance.csv', 'w') as f:
            f.write('Name,Time\n')

    with open('attendance.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = [line.split(',')[0] for line in myDataList]
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.write(f'{name},{dtString}\n')

def take_attendance():
    cap = cv2.VideoCapture(0)

    recognized = False
    unknown_timeout = 10
    unknown_start = datetime.now()

    cv2.namedWindow("Attendance", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Attendance", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    while True:
        success, img = cap.read()
        if not success:
            break

        imgS = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = [v * 4 for v in faceLoc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)
                recognized = True
                break  # Break after first recognized face

        cv2.imshow('Attendance', img)
        key = cv2.waitKey(1)

        if recognized:
            break

        if (datetime.now() - unknown_start).total_seconds() > unknown_timeout:
            print("⏳ Timeout: No known face found.")
            break

    cap.release()
    cv2.destroyAllWindows()
    if recognized:
        messagebox.showinfo("Success", f"✅ Attendance marked for {name}")
    else:
        messagebox.showwarning("Notice", "⚠️ No known face recognized.")

def upload_attendance():
    # Add logic to upload attendance.csv
    print("📤 Uploading attendance to web/app (TODO)")
    messagebox.showinfo("Upload", "Upload logic goes here.")

# ---------------- GUI -----------------
root = tk.Tk()
root.title("Attendance System")
root.attributes('-fullscreen', True)
root.configure(bg="#282c34")

tk.Label(root, text="Face Recognition Attendance System", font=("Arial", 30, "bold"), bg="#282c34", fg="white").pack(pady=40)

btn_frame = tk.Frame(root, bg="#282c34")
btn_frame.pack(pady=40)

take_btn = tk.Button(btn_frame, text="📷 Take Attendance", font=("Arial", 20), width=20, height=2, command=take_attendance, bg="#61afef", fg="white")
take_btn.grid(row=0, column=0, padx=40)

upload_btn = tk.Button(btn_frame, text="📤 Upload Attendance", font=("Arial", 20), width=20, height=2, command=upload_attendance, bg="#98c379", fg="white")
upload_btn.grid(row=0, column=1, padx=40)

# Escape key to exit full screen
root.bind("<Escape>", lambda e: root.attributes('-fullscreen', False))

root.mainloop()
