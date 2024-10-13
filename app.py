from flask import Flask, render_template, Response
import cv2
import dlib
import numpy as np
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation

app = Flask(__name__)

# Initialize dlib's face detector (HOG-based)
detector = dlib.get_frontal_face_detector()

# Initialize background subtractor (KNN) with parameters
backSub = cv2.createBackgroundSubtractorKNN(history=500, dist2Threshold=400, detectShadows=False)

# Load the pre-trained SegFormer model
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = SegformerImageProcessor.from_pretrained(model_name)
model = SegformerForSemanticSegmentation.from_pretrained(model_name)

@app.route('/')
def index():
    return render_template('index.html')

def generate_frames():
    cap = cv2.VideoCapture(0)  # Ensure correct camera index
    if not cap.isOpened():
        print("Error: Could not open video.")
        exit()
    
    # Get original dimensions of the frame
    ret, frame = cap.read()
    height, width = frame.shape[:2]
    aspect_ratio = width / height
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Maintain original aspect ratio
        target_width = 640
        target_height = int(target_width / aspect_ratio)
        frame = cv2.resize(frame, (target_width, target_height))
        
        # Increase brightness
        frame = cv2.convertScaleAbs(frame, alpha=1.2, beta=30)  # Adjust alpha (contrast) and beta (brightness)
        
        # Convert frame to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # Detect faces in the frame
        faces = detector(gray)
        
        # Apply background subtraction
        fgMask = backSub.apply(frame)
        
        for face in faces:
            x, y, w, h = (face.left(), face.top(), face.width(), face.height())
            face_roi = frame[y:y+h, x:x+w]
            mask_roi = fgMask[y:y+h, x:x+w]
            
            # Ensure mask_roi is not empty before processing
            if mask_roi.size > 0:
                # Apply Gaussian Blur to smooth the mask
                blurred_mask = cv2.GaussianBlur(mask_roi, (15, 15), 0)
                _, clean_mask = cv2.threshold(blurred_mask, 128, 255, cv2.THRESH_BINARY)
                
                # Morphological operations to remove noise and fill holes
                kernel = np.ones((5, 5), np.uint8)
                clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
                clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_OPEN, kernel)
                
                # Use inpainting to fill black patches
                clean_mask = cv2.inpaint(clean_mask, clean_mask, 3, cv2.INPAINT_TELEA)
                
                # Use SegFormer for semantic segmentation
                input_image = cv2.resize(frame, (512, 512))  # Reduce input size for faster processing
                inputs = processor(images=input_image, return_tensors="pt")
                with torch.no_grad():
                    outputs = model(**inputs)
                logits = outputs.logits
                semantic_mask = logits.argmax(dim=1).squeeze().cpu().numpy().astype(np.uint8)
                semantic_mask = cv2.resize(semantic_mask, (frame.shape[1], frame.shape[0]))
                
                # Align the mask properly with face region
                face_mask = semantic_mask[y:y+h, x:x+w]
                face_with_colors = cv2.bitwise_and(face_roi, face_roi, mask=face_mask)
                
                # Prepare segmented face as frame
                ret, buffer = cv2.imencode('.jpg', face_with_colors)
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    cap.release()

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)