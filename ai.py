# -*- coding: utf-8 -*-
import cv2
import boto3
from dotenv import load_dotenv
import os
import json
from datetime import datetime
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import platform

load_dotenv()

# Load DEBUG_MODE from environment
debug_mode = os.getenv("DEBUG_MODE", "false").lower() in ("true", "1", "yes")

client = boto3.client(
    'rekognition',
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION")
)

trash_labels = {"Plastic", "Bottle", "Can", "Paper", "Trash", "Garbage", "Battery", "Cardboard"}

# Color coding for trash types (BGR format for OpenCV)
# Red: Hazardous items
# Blue: Recyclable metals
# Green: Organic/Compostable
# Black: General waste
trash_color_mapping = {
    "Battery": (0, 0, 255),      # Red - Hazardous
    "Trash": (0, 0, 0),          # Black - General waste
    "Garbage": (0, 0, 0),        # Black - General waste
    "Can": (255, 0, 0),          # Blue - Recyclable metal
    "Bottle": (255, 0, 0),       # Blue - Recyclable
    "Plastic": (255, 0, 0),      # Blue - Recyclable
    "Paper": (0, 255, 0),        # Green - Organic/Compostable
    "Cardboard": (0, 255, 0),    # Green - Organic/Compostable
}

# States for debug mode
if debug_mode:
    STATE_CAMERA = 0
    STATE_REVIEW = 1
# States for non-debug mode
else:
    STATE_INSTRUCTION = 0
    STATE_LOADING = 1
    STATE_RESULT = 2


def detect_labels_bytes(image_bytes):
    response = client.detect_labels(
        Image={'Bytes': image_bytes},
        MaxLabels=10,
        MinConfidence=70
    )
    return response['Labels']


def put_text_utf8(image, text, position, font_scale=1, color=(0, 0, 0), thickness=1):
    """Draw text with UTF-8 support using PIL"""
    # Convert BGR to RGB for PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_image)

    # Try to load a system font that supports UTF-8
    try:
        # macOS system fonts
        font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", int(font_scale * 30))
    except:
        try:
            # Linux system fonts
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", int(font_scale * 30))
        except:
            # Fallback to default font
            font = ImageFont.load_default()

    # Draw text
    draw.text(position, text, font=font, fill=color)

    # Convert back to BGR
    return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)


def draw_instruction_screen(image):
    """Draw instruction screen for non-debug mode"""
    # Fill with white background
    image[:] = (255, 255, 255)

    text = "Çöpü Kameraya Gösterip SPACE Basınız"
    image = put_text_utf8(image, text, (int(image.shape[1] / 2) - 300, int(image.shape[0] / 2) - 20), 1.2, (0, 0, 0), 2)

    return image


def draw_loading_screen(image, frame_count):
    """Draw animated rotating spinner with white background"""
    # Fill with white background
    image[:] = (255, 255, 255)

    # Loading text
    text = "Yanıt Bekleniyor..."
    image = put_text_utf8(image, text, (int(image.shape[1] / 2) - 180, int(image.shape[0] / 2) - 100), 1.2, (0, 0, 0), 2)

    # Spinner center
    center_x = int(image.shape[1] / 2)
    center_y = int(image.shape[0] / 2) + 30

    # Simple but obvious rotating spinner
    radius = 60
    thickness = 8

    # Calculate rotation angle - visible rotation every frame
    angle = (frame_count * 6) % 360  # Fast rotation: 6 degrees per frame

    # Draw rotating gradient arcs
    for i in range(4):
        arc_angle = (angle + i * 90) % 360
        next_angle = (arc_angle + 60) % 360

        # Color gradient
        if i == 0:
            color = (66, 165, 245)  # Bright blue
        elif i == 1:
            color = (100, 150, 200)  # Medium blue
        elif i == 2:
            color = (140, 180, 220)  # Light blue
        else:
            color = (180, 210, 240)  # Very light blue

        cv2.ellipse(image, (center_x, center_y), (radius, radius), 0, int(arc_angle), int(next_angle), color, thickness)

    # Draw pulsing center circle
    pulse = int(20 + 10 * np.sin(frame_count * 0.1))
    cv2.circle(image, (center_x, center_y), pulse, (66, 165, 245), -1)
    cv2.circle(image, (center_x, center_y), pulse, (255, 255, 255), 2)

    return image


def draw_result_screen(image, labels):
    """Draw result screen with detected trash item and color-coded circle"""
    # Fill with white background
    image[:] = (255, 255, 255)

    # Filter to only trash items
    trash_items = [l for l in labels if l["Name"] in trash_labels]

    if trash_items:
        # Get highest confidence trash item
        top_item = max(trash_items, key=lambda x: x["Confidence"])
        item_name = top_item['Name']
        confidence = top_item['Confidence']

        # Get color for this item
        item_color = trash_color_mapping.get(item_name, (0, 0, 0))

        text = f"Algılanan: {item_name}"
        confidence_text = f"Güven: %{confidence:.1f}"
    else:
        text = "Çöp Algılanmadı"
        confidence_text = ""
        item_color = (200, 200, 200)  # Gray for no detection

    # Draw large color-coded circle at the top
    center_x = int(image.shape[1] / 2)
    circle_y = 100
    circle_radius = 60

    # Draw filled circle with item color
    cv2.circle(image, (center_x, circle_y), circle_radius, item_color, -1)
    # Draw border around circle
    cv2.circle(image, (center_x, circle_y), circle_radius, (0, 0, 0), 3)

    # Draw item text below circle
    image = put_text_utf8(image, text, (int(image.shape[1] / 2) - 200, int(image.shape[0] / 2) - 30), 1.5, (0, 0, 0), 2)

    if confidence_text:
        image = put_text_utf8(image, confidence_text, (int(image.shape[1] / 2) - 180, int(image.shape[0] / 2) + 30), 1.0, (0, 0, 0), 2)

    # Show legend for colors
    legend_y = int(image.shape[0] / 2) + 100
    legend_text = "KIRMIZI:Tehlikeli | MAVİ:Geri Dönüştürülebilir | YESİL:Organik | SİYAH:Genel"
    image = put_text_utf8(image, legend_text, (30, legend_y), 0.8, (100, 100, 100), 1)

    # Show instruction to go back
    text_back = "Devam Etmek İçin SPACE Tuşuna Basınız"
    image = put_text_utf8(image, text_back, (int(image.shape[1] / 2) - 280, int(image.shape[0] / 2) + 140), 0.9, (100, 100, 100), 1)

    return image


def save_result(raw_frame, labels):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = f"./Save/{timestamp}"
    os.makedirs(save_dir, exist_ok=True)

    # Save raw image (NOT annotated)
    cv2.imwrite(os.path.join(save_dir, "image.png"), raw_frame)

    # Clean JSON format
    formatted = []
    for l in labels:
        formatted.append({
            "name": l["Name"],
            "confidence": round(l["Confidence"], 2),
            "is_trash": l["Name"] in trash_labels
        })

    with open(os.path.join(save_dir, "result.json"), "w") as f:
        json.dump(formatted, f, indent=4)

    print(f"[SAVED] {save_dir}")


def main():
    if platform.system() == "Windows":
      cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
     cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if debug_mode:
        # ===== DEBUG MODE =====
        state = STATE_CAMERA
    else:
        # ===== NON-DEBUG MODE =====
        state = STATE_INSTRUCTION

    frame = None
    frozen_frame = None
    frozen_labels = None
    raw_capture = None
    frame_count = 0
    loading_frame_count = 0  # Track frames in loading state

    print("Hazır.")

    while True:
        frame_count += 1

        # ========== DEBUG MODE ==========
        if debug_mode:
            # CAMERA STATE
            if state == STATE_CAMERA:
                ret, frame = cap.read()
                if not ret:
                    continue

                cv2.imshow("TrashSys - Kamera", frame)

            # REVIEW STATE
            elif state == STATE_REVIEW:
                cv2.imshow("TrashSys - Sonuç", frozen_frame)

            key = cv2.waitKey(1) & 0xFF

            # ESC → always exit
            if key == 27:
                break

            # ================= CAMERA STATE =================
            if state == STATE_CAMERA:

                if key == 32:  # SPACE → capture
                    raw_capture = frame.copy()

                    ret2, buffer = cv2.imencode('.jpg', frame)
                    if not ret2:
                        print("Encode failed")
                        continue

                    image_bytes = buffer.tobytes()

                    try:
                        labels = detect_labels_bytes(image_bytes)
                    except Exception as e:
                        print("AWS error:", e)
                        continue

                    # Draw overlay
                    display_frame = frame.copy()
                    y0 = 30

                    for i, label in enumerate(labels):
                        name = label['Name']
                        confidence = label['Confidence']
                        color = (0, 0, 255) if name in trash_labels else (0, 255, 255)

                        text = f"{name} ({confidence:.1f}%)"
                        cv2.putText(
                            display_frame,
                            text,
                            (10, y0 + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            1,
                            color,
                            2
                        )

                    frozen_frame = display_frame
                    frozen_labels = labels

                    state = STATE_REVIEW
                    print("Captured → ENTER save | SPACE back")

            # ================= REVIEW STATE =================
            elif state == STATE_REVIEW:

                if key == 32:  # SPACE → back to camera
                    # --- Refresh camera completely ---
                    cv2.destroyAllWindows()                    # close all windows
                    cap.release()                              # close old stream
                    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)  # open fresh one
                    if not cap.isOpened():
                        print("Kamera yeniden açılamadı.")
                        break

                    frozen_frame = None
                    frozen_labels = None
                    raw_capture = None
                    state = STATE_CAMERA
                    print("Back to camera (camera refreshed)")

                elif key == 13:  # ENTER → save
                    if raw_capture is not None and frozen_labels is not None:
                        save_result(raw_capture, frozen_labels)

        # ========== NON-DEBUG MODE ==========
        else:
            # Create a blank canvas for display
            canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

            # INSTRUCTION STATE
            if state == STATE_INSTRUCTION:
                display = draw_instruction_screen(canvas.copy())
                cv2.imshow("TrashSys", display)

            # LOADING STATE
            elif state == STATE_LOADING:
                display = draw_loading_screen(canvas.copy(), frame_count)
                cv2.imshow("TrashSys", display)

            # RESULT STATE
            elif state == STATE_RESULT:
                display = draw_result_screen(canvas.copy(), frozen_labels)
                cv2.imshow("TrashSys", display)

            key = cv2.waitKey(1) & 0xFF

            # ESC → always exit
            if key == 27:
                break

            # ================= INSTRUCTION STATE =================
            if state == STATE_INSTRUCTION:
                if key == 32:  # SPACE → capture and process
                    ret, frame = cap.read()
                    if ret:
                        raw_capture = frame.copy()
                        state = STATE_LOADING
                        print("Capturing and analyzing...")

            # ================= LOADING STATE =================
            elif state == STATE_LOADING:
                loading_frame_count += 1

                # Show loading animation for at least 30 frames before processing
                if loading_frame_count > 30 and frame is not None:
                    ret2, buffer = cv2.imencode('.jpg', frame)
                    if ret2:
                        image_bytes = buffer.tobytes()

                        try:
                            labels = detect_labels_bytes(image_bytes)
                            frozen_labels = labels
                            state = STATE_RESULT
                            loading_frame_count = 0
                            print("Analysis complete")
                        except Exception as e:
                            print("AWS error:", e)
                            state = STATE_INSTRUCTION
                            loading_frame_count = 0

            # ================= RESULT STATE =================
            elif state == STATE_RESULT:
                if key == 32:  # SPACE → back to instruction
                    state = STATE_INSTRUCTION
                    frozen_labels = None
                    frame = None
                    print("Back to instruction screen")

    cap.release()
    cv2.destroyAllWindows()



if __name__ == "__main__":
    main()