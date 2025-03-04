from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import render, redirect
import cv2
import base64
import numpy as np
from ultralytics import YOLO
import onnxruntime as ort  # Import ONNX runtime
import json
import logging
import os
import time
from collections import Counter
import firebase_admin
from firebase_admin import db

# 6969

# Load the YOLOv8 model outside the view function
# model = YOLO(r"/home/raspi/Downloads/my_yolov8_app/yolo11_48n.pt")
# model = YOLO(r"/home/raspi/Downloads/my_yolov8_app/yolo11_100n.onnx")
# model = YOLO(r"/home/raspi/Downloads/my_yolov8_app/yolo11_81_32.tflite")
model = YOLO(r"C:\my_yolov8_app\yolo11_107s.onnx")

def harga_buah_view(request):
    return render(request, 'hargarn.html')  # Sesuaikan dengan nama file HTML kamu

def realtime_view(request):
    return render(request, "realtime.html")  # Render template tanpa data langsung

def get_realtime_data(request):
    try:
        ref = db.reference("fruit")
        data = ref.get() or {}  
        return JsonResponse(data)  # Kirim sebagai JSON
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)

def get_price_description(fruit_name, request):
    ref = db.reference('fruit')  # Ambil referensi Firebase
    all_fruits = ref.get()  # Ambil semua data buah

    if all_fruits:
        for key, fruit in all_fruits.items():
            if fruit.get('nama') == fruit_name:
                return fruit['harga']  # Kembalikan harga buah
    return "Harga tidak tersedia"

def get_item_prices(request):
    ref = db.reference("fruit")  # Sesuaikan dengan struktur database Firebase
    items = ref.get()
    return JsonResponse(items, safe=False)

    
    
def combine_detections(detections):
    """
    Menggabungkan deteksi anggur yang berdekatan.
    """
    combined_detections = []
    ceri_detections = [d for d in detections if d['class'] == 'Anggur']
    used_indices = set()

    for i, det1 in enumerate(ceri_detections):
        if i in used_indices:
            continue

        for j, det2 in enumerate(ceri_detections[i+1:], i+1):
            if j in used_indices:
                continue

            center1 = np.array([(det1['bbox'][0] + det1['bbox'][2]) / 2, (det1['bbox'][1] + det1['bbox'][3]) / 2])
            center2 = np.array([(det2['bbox'][0] + det2['bbox'][2]) / 2, (det2['bbox'][1] + det2['bbox'][3]) / 2])
            distance = np.linalg.norm(center1 - center2)

            if distance < 50:  # Atur threshold jarak sesuai kebutuhan
                # Gabungkan deteksi
                combined_bbox = [min(det1['bbox'][0], det2['bbox'][0]), min(det1['bbox'][1], det2['bbox'][1]),
                                 max(det1['bbox'][2], det2['bbox'][2]), max(det1['bbox'][3], det2['bbox'][3])]
                combined_detections.append({'class': 'Anggur', 'confidence': max(det1['confidence'], det2['confidence']), 'bbox': combined_bbox})
                used_indices.update([i, j])
                break
        else:
            # Jika tidak ada deteksi yang digabungkan, tambahkan deteksi asli
            combined_detections.append(det1)

    # Tambahkan deteksi non-anggur ke hasil akhir
    combined_detections.extend([d for d in detections if d['class'] != 'Anggur'])

    return combined_detections

def realtime_view(request):
    ref = db.reference("fruit")  # Pastikan path sesuai dengan Firebase
    data = ref.get() or {}  # Ambil data, jika None ganti dengan {}
    print("Data dari Firebase:", data)  # Debugging, lihat data di terminal
    return render(request, "realtime.html", {"buah_data": data})


def detect_objects(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

            results = model(image)
            result = results[0]

            annotated_image = result.plot()
            detections = []
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0]
                    conf = box.conf[0]
                    cls = int(box.cls[0])
                    detections.append({
                        'class': model.names[cls],
                        'confidence': float(conf),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })

            detections = combine_detections(detections)

            _, original_img_encoded = cv2.imencode('.jpg', image)
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')

            _, annotated_img_encoded = cv2.imencode('.jpg', annotated_image)
            annotated_img_base64 = base64.b64encode(annotated_img_encoded).decode('utf-8')

            return JsonResponse({
                'detections': detections,
                'original_image': original_img_base64,
                'annotated_image': annotated_img_base64,
            })

    return JsonResponse({'error': 'No image uploaded'})

import time


from django.http import JsonResponse

def video_stream(request):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        return JsonResponse({'error': 'Cannot open camera'})

    ret, frame = cap.read()
    if not ret:
        return JsonResponse({'error': 'Cannot read frame from camera'})

    cap.release()

    results = model(frame)

    # Extract detected object names
    object_names =[]
    if isinstance(results, list):
        for result in results:
            annotated_image = result.plot()
            for r in result.boxes.cls:
                object_names.append(model.names[int(r)])
    else:
        annotated_image = results.plot()
        for r in results.boxes.cls:
            object_names.append(model.names[int(r)])

    # Encode the annotated image
    _, jpeg = cv2.imencode('.jpg', annotated_image)
    annotated_image_base64 = base64.b64encode(jpeg).decode('utf-8')

    # Hitung kuantitas objek
    object_counts = Counter(object_names)

    # Buat list objek dan kuantitas
    objects_with_quantity =[]
    for name, count in object_counts.items():
        objects_with_quantity.append({'name': name, 'quantity': count})

    # Ambil harga dan hitung harga total
    price_descriptions =[]
    total_prices =[]
    for obj in objects_with_quantity:
        price = get_price_description(obj['name'], request)
        if price is not None:
            price_descriptions.append(f"Rp. {price} per buah.")
            try:
                harga = int(price)
            except ValueError:
                harga = 0  # Jika tetap error, default ke 0
            kuantitas = int(obj['quantity'])
            total_price = harga * kuantitas
            total_prices.append(total_price)
        else:
            price_descriptions.append("Price information not available for " + obj['name'])
            total_prices.append(None)

    # Calculate the overall total price
    overall_total = sum(total_price for total_price in total_prices if total_price is not None)

    return JsonResponse({
        'objects': objects_with_quantity,
        'annotated_image': annotated_image_base64,
        'price_descriptions': price_descriptions,
        'total_prices': total_prices,
        'overall_total': overall_total
    })


def upload_image(request):
    return render(request, 'upload.html')

def detect_realtime(request):
    return render(request, 'realtime.html')

def home(request):
    return render(request, 'home.html')

    # Ambil harga buah dari Firebase
    price_data = {fruit: ref.child(fruit).get() for fruit in ['Pisang', 'Mangga', 'Apel', 'Anggur', 'Jeruk']}

    return render(request, 'change_price.html', {
        'message': message,
        'error': error,
        'password_correct': password_correct,
        'price_data': price_data,
    })

def logout(request):
    if request.method == 'POST':
        if request.session.get('password_correct'):
            del request.session['password_correct']
    return redirect('login')  # Redirect ke halaman login setelah logout

def about(request):
    return render(request, 'about.html')

def intro(request):
    return render(request, 'intro.html')


def detect_realtime(request):
    if request.method == 'POST':
        image_data = request.POST.get('image_data')
        if image_data:
            # Decode the base64 image data
            header, encoded = image_data.split(",", 1)
            image_bytes = base64.b64decode(encoded)
            nparr = np.frombuffer(image_bytes, np.uint8)
            image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            # Perform object detection
            results = model(image)
            detections =[]
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy
                    conf = box.conf
                    cls = int(box.cls)
                    detections.append({
                        'class': model.names[cls],
                        'confidence': float(conf),
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })

            # Gabungkan deteksi anggur yang berdekatan
            detections = combine_detections(detections)

            # Return detections as text
            detection_text = ""
            for detection in detections:
                label = f"Detected: {detection['class']} ({detection['confidence']*100:.0f}%)"
                detection_text += label + "<br>"  # Add a line break after each detection

            return JsonResponse({'detections': detection_text})

    return render(request, 'realtime.html')
    
