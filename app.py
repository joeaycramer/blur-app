import cv2
import numpy as np
import requests
import json
from flask import Flask, request, send_file, jsonify, render_template
from ultralytics import YOLO
from io import BytesIO
import os

# --- CONFIGURATION ---
# Your integrated API Key for Plate Recognizer (used for fallback)
PLATE_RECOGNIZER_API_TOKEN = os.getenv("PLATE_RECOGNIZER_API_TOKEN")
PLATE_RECOGNIZER_URL = 'https://api.platerecognizer.com/v1/plate-reader/'
REGION_CODE = "gb" 
# -----------------------------------

app = Flask(__name__, template_folder='templates')

# Load the custom YOLOv8 model
try:
    model = YOLO('license_plate_detector.pt')
except Exception as e:
    print(f"FATAL ERROR: Failed to load model. {e}")
    print("⚠️ Please run 'python3 download_model.py' first.")
    exit(1)


def blur_region(image, box):
    """
    Applies an inflated, feathered pixelated blur twice to the bounding box region,
    with custom settings and dynamic pixelation factor relative to plate size.
    """
    # --- CUSTOM SETTINGS ---
    INFLATION_RATIO = 0.35   # Expand box by 35% for feathering coverage
    FEATHER_RATIO = 0.20     # Feather the outer 20%
    PIXEL_DENSITY_PASS1 = 10 # Pass 1: standard strength (smaller number = larger blocks)
    PIXEL_DENSITY_PASS2 = 25 # Pass 2: stronger strength (smaller number = larger blocks)
    NUM_BLUR_PASSES = 2      # How many times to blur the ROI
    
    # 1. GET & INFLATE COORDINATES
    x1, y1, x2, y2 = map(int, box)
    h_img, w_img = image.shape[:2]

    # Calculate current box dimensions for dynamic padding
    box_w = x2 - x1
    box_h = y2 - y1
    pad_w = int(box_w * INFLATION_RATIO)
    pad_h = int(box_h * INFLATION_RATIO)
    
    # Apply padding and ensure coordinates stay within image bounds
    x1 = max(0, x1 - pad_w)
    y1 = max(0, y1 - pad_h)
    x2 = min(w_img, x2 + pad_w)
    y2 = min(h_img, y2 + pad_h)

    # Extract the ROI (Region of Interest)
    roi = image[y1:y2, x1:x2].copy()
    if roi.size == 0: return image

    h, w = roi.shape[:2]
    min_dim = min(h, w)
    
    # 2. PERFORM MULTIPLE PIXELATION PASSES ON THE ROI
    current_roi = roi.copy()
    
    for i in range(NUM_BLUR_PASSES):
        density = PIXEL_DENSITY_PASS1 if i == 0 else PIXEL_DENSITY_PASS2
        
        # Calculate dynamic factor based on plate size
        factor = max(3, int(min_dim / density)) 
        
        # Shrink the current ROI
        small = cv2.resize(current_roi, (w // factor, h // factor), interpolation=cv2.INTER_LINEAR)
        # Enlarge it back (Pixelation effect)
        pixelated = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Update current_roi for the next pass
        current_roi = pixelated 

    # 3. CREATE MASK (The "Feathering Gradient")
    mask = np.zeros((h, w), dtype=np.uint8)

    # Calculate safe fade margin (FEATHER_RATIO of shortest side, min 1 pixel)
    fade_pixels_max = int(min_dim * 0.40) 
    fade_pixels = int(min_dim * FEATHER_RATIO)
    fade_pixels = max(1, min(fade_pixels, fade_pixels_max)) 
    
    # Safety check: ensure inner box is drawable
    if w - (2 * fade_pixels) < 1 or h - (2 * fade_pixels) < 1:
        fade_pixels = max(1, min(w, h) // 3)

    # Draw solid white box in center
    cv2.rectangle(mask, (fade_pixels, fade_pixels), (w - fade_pixels, h - fade_pixels), 255, -1)

    # Blur the mask to create the smooth gradient for alpha blending
    ksize = fade_pixels * 2 + 1
    mask_blurred = cv2.GaussianBlur(mask, (ksize, ksize), 0)

    # 4. BLEND
    alpha = mask_blurred.astype(float) / 255.0
    alpha = np.dstack([alpha] * 3) # 3 channels for RGB blending

    # Final blend: (Double-Pixelated Image * Alpha) + (Original ROI * (1 - Alpha))
    soft_roi = (current_roi * alpha) + (roi * (1.0 - alpha))

    # 5. PASTE BACK
    image[y1:y2, x1:x2] = soft_roi.astype(np.uint8)
    
    return image


def call_plate_recognizer_api(image_bytes):
    """
    Calls the Plate Recognizer API to get bounding box coordinates (as fallback).
    """
    if PLATE_RECOGNIZER_API_TOKEN == "YOUR_API_TOKEN_HERE" or not PLATE_RECOGNIZER_API_TOKEN:
        print("❌ API KEY MISSING. Skipping fallback.")
        return []

    # Prepare request
    files = {'upload': ('image.jpg', image_bytes, 'image/jpeg')}
    headers = {'Authorization': f'Token {PLATE_RECOGNIZER_API_TOKEN}'}
    data = {'regions': [REGION_CODE]}

    try:
        response = requests.post(PLATE_RECOGNIZER_URL, files=files, headers=headers, data=data, timeout=10)
        response.raise_for_status() 
        data = response.json()
        api_boxes = []

        for result in data.get('results', []):
            box_data = result.get('box')
            if box_data:
                x1 = box_data['xmin']
                y1 = box_data['ymin']
                x2 = box_data['xmax']
                y2 = box_data['ymax']
                api_boxes.append([x1, y1, x2, y2])
        
        print(f"✅ Plate Recognizer (Fallback) found {len(api_boxes)} plates.")
        return api_boxes

    except requests.exceptions.RequestException as e:
        print(f"❌ Plate Recognizer API Error. Status: {response.status_code if 'response' in locals() else 'N/A'}. Error: {e}")
        return []


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/process', methods=['POST'])
def process_image():
    if 'file' not in request.files: return jsonify({"error": "No file uploaded"}), 400
    
    # Read the file stream once and store it
    file_stream = request.files['file'].read()
    
    # --- YOLO PROCESSING (Primary Method) ---
    file_bytes = np.frombuffer(file_stream, np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    
    if image is None: return jsonify({"error": "Invalid image file"}), 400

    # Run detection with TTA (augment=True) and low confidence threshold
    # In process_image():
    results = model(image, conf=0.25, augment=True, iou=0.5)
    
    detections_found = 0
    final_image = image.copy() 

    for result in results:
        for box in result.boxes:
            final_image = blur_region(final_image, box.xyxy[0])
            detections_found += 1
    
    # --- API FALLBACK LOGIC ---
    if detections_found == 0:
        print("⚠️ YOLO failed to find plates. Initiating API fallback...")
        
        api_boxes = call_plate_recognizer_api(file_stream) 
        
        for box in api_boxes:
            final_image = blur_region(final_image, box)
            detections_found += 1
            
        if detections_found > 0:
            print(f"✅ Fallback successful. Blurred {detections_found} plate(s).")

    # --- RETURN IMAGE ---
    _, buffer = cv2.imencode('.jpg', final_image)
    io_buf = BytesIO(buffer)
    io_buf.seek(0)
    
    # Pass detection count back to frontend for user feedback
    response = send_file(io_buf, mimetype='image/jpeg')
    response.headers['X-Detections-Count'] = str(detections_found)
    
    return response

if __name__ == '__main__':
    print("⚡ Server starting... Open http://127.0.0.1:5000 in your browser")
    app.run(debug=True, port=5000)