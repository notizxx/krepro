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








# Load the YOLOv8 model outside the view function
# model = YOLO(r"/home/raspi/Downloads/my_yolov8_app/yolo11_48n.pt")
# model = YOLO(r"/home/raspi/Downloads/my_yolov8_app/yolo11_81.onnx")
model = YOLO(r"/home/raspi/Downloads/my_yolov8_app/yolo11_81_32.tflite")


# Dictionary to store fruit names and their corresponding units
FRUIT_UNITS = {
    "Pisang": "kg",
    "Mangga": "kg",
    "Jeruk": "kg",
    "Anggur": "kg",
    "Apel": "kg",
    "Semangka": "kg",
}

# Konfigurasi logging
logging.basicConfig(filename='change_price.log', level=logging.DEBUG,
                    format='%(asctime)s - %(levelname)s - %(message)s')

def get_price_description(fruit_name, request):
    """
    Returns a price description for a given fruit, fetching the price from the session.
    """
    price = request.session.get(fruit_name, {}).get('price')

    price_data = {
        "Pisang": {
            "unit": "buah"
        },
        "Mangga": {
            "unit": "buah"
        },
        #... add more fruits and their unit information...
        "Jeruk": {
            "unit": "buah"
        },
        "Apel": {
            "unit": "buah"
        },
        "Semangka": {
            "unit": "buah"
        },
        
    }

    fruit_info = price_data.get(fruit_name)
    if fruit_info and price is not None:
        return f"Harga dari {fruit_name} adalah Rp. {price} per {fruit_info['unit']}."
    else:
        return "Price information not available for " + fruit_name

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

def detect_objects(request):
    if request.method == 'POST':
        if 'image' in request.FILES:
            image_file = request.FILES['image']
            image = cv2.imdecode(np.frombuffer(image_file.read(), np.uint8), cv2.IMREAD_COLOR)

            # Perform object detection
            results = model(image)  # Get the list of results
            result = results[0]     # Access the first element (YOLO object)

            annotated_image = result.plot()  # Use the YOLO object to plot

            # Process results (extract labels, bounding boxes, confidence scores)
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

            # Gabungkan deteksi anggur yang berdekatan
            detections = combine_detections(detections)

            # Draw bounding boxes and labels on the image
            for detection in detections:
                x1, y1, x2, y2 = detection['bbox']
                cv2.rectangle(annotated_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"Detected: {detection['class']} ({detection['confidence']*100:.0f}%)"
                cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        
            # Get price descriptions for detected objects
            price_descriptions = []
            for detection in detections:
                fruit_name = detection['class']
                price_description = get_price_description(fruit_name, request)
                price_descriptions.append(price_description)

            # Encode the original image
            _, original_img_encoded = cv2.imencode('.jpg', image)
            original_img_base64 = base64.b64encode(original_img_encoded).decode('utf-8')

            # Encode the annotated image
            _, annotated_img_encoded = cv2.imencode('.jpg', annotated_image)
            annotated_img_base64 = base64.b64encode(annotated_img_encoded).decode('utf-8')

            return JsonResponse({'detections': detections,
                                 'original_image': original_img_base64,
                                 'annotated_image': annotated_img_base64,
                                 'price_descriptions': price_descriptions})

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

    cap.release()  # Release the camera immediately after capturing

    results = model(frame)
    print(type(results))

    # Extract detected object names
    object_names =[]
    if isinstance(results, list):
        for result in results:
            annotated_image = result.plot()  # Generate the annotated image here
            for r in result.boxes.cls:
                object_names.append(model.names[int(r)])
    else:
        annotated_image = results.plot()  # Generate the annotated image here
        for r in results.boxes.cls:
            object_names.append(model.names[int(r)])

    # Encode the annotated image
    _, jpeg = cv2.imencode('.jpg', annotated_image)
    annotated_image_base64 = base64.b64encode(jpeg).decode('utf-8')

    # Return object names and the annotated image as JSON response
    return JsonResponse({'objects': object_names, 'annotated_image': annotated_image_base64})


def upload_image(request):
    return render(request, 'upload.html')

def detect_realtime(request):
    return render(request, 'realtime.html')

def home(request):
    return render(request, 'home.html')

def login(request):
    error = None
    if request.method == 'POST':
        password = request.POST.get('password')
        if password == "123":  # Ganti dengan password yang diinginkan
            request.session['password_correct'] = True
            return redirect('change_price')
        else:
            error = 'Incorrect password.'
    return render(request, 'login.html', {'error': error})

def change_price(request):
    message = None
    error = None
    password_correct = request.session.get('password_correct', False)

    logging.debug(f"Entering change_price view. password_correct: {password_correct}")
    logging.debug(f"Session data: {request.session.items()}")

    if not password_correct:
        return redirect('login')  # Redirect ke halaman login jika belum login

    if request.method == 'POST':
        fruit = request.POST.get('fruit')
        price = request.POST.get('price')
        logging.debug(f"Received fruit: {fruit}, price: {price}")

        if fruit and price:
            try:
                price = int(price)
                if price <= 0:  # Check if price is negative
                    error = 'Price cannot be negative.'
                    logging.error("Negative price value provided.")
                else:  # *** This else block was added ***
                    old_price = request.session.get(fruit, {}).get('price')
                    logging.debug(f"Old price for {fruit}: {old_price}")

                request.session[fruit] = {'price': price}
                message = 'Price updated successfully!'
                logging.debug(f"Price updated for {fruit}: {price}")
                logging.debug(f"Session data after update: {request.session.items()}")
            except ValueError:
                error = 'Invalid price value.'
                logging.error("Invalid price value provided.")

    price_data = {
        fruit: request.session.get(fruit, {}).get('price')
        for fruit in ['Pisang', 'Mangga', 'Apel', 'Semangka', 'Anggur', 'Jeruk']
    }

    logging.debug(f"Exiting change_price view. message: {message}, error: {error}")
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
    
