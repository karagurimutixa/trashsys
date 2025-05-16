import cv2
import openai
import base64
import time
import os
from dotenv import load_dotenv

# .env dosyasını yükle
load_dotenv()
openai.api_key = os.getenv("OPENROUTER_API_KEY")
openai.base_url = "https://openrouter.ai/api/v1"  # OpenRouter URL'si

def encode_frame_to_base64(frame):
    _, buffer = cv2.imencode('.jpg', frame)
    return base64.b64encode(buffer).decode()

def classify_image(image_base64):
    messages = [
        {"role": "system", "content": "You are an assistant specialized in classifying images of trash. Respond only with the trash type like 'plastic', 'metal', 'organic', or 'paper'."},
        {"role": "user", "content": f"Classify this image: data:image/jpeg;base64,{image_base64}"}
    ]
    try:
        response = openai.ChatCompletion.create(
            model="deepseek/deepseek-chat-v3-0324:free",
            messages=messages
        )
        return response['choices'][0]['message']['content']
    except Exception as e:
        print("API Hatası:", e)
        return "Algılanamadı"

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Kamera açılamadı!")
        return

    last_classification_time = 0
    classification_interval = 3  # saniye

    detected_text = ""

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kare alınamadı!")
            break

        current_time = time.time()
        if current_time - last_classification_time > classification_interval:
            img_b64 = encode_frame_to_base64(frame)
            result = classify_image(img_b64)
            print("Algılama sonucu:", result)
            detected_text = result
            last_classification_time = current_time

        # Ekrana sonucu yaz
        cv2.putText(frame, f"Tespit: {detected_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Çöp Sınıflandırıcı', frame)

        # 'q' tuşuna basınca çık
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
