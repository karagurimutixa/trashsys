import cv2
import boto3

# AWS Rekognition client ayarları
client = boto3.client('rekognition',
                      aws_access_key_id='ACCESS_KEY_HERE',
                      aws_secret_access_key='SECRET_ACCESS_KEY_HERE',
                      region_name='us-east-1')

# Çöp türü etiketleri (örnek)
trash_labels = {"Plastic", "Bottle", "Can", "Paper", "Trash", "Garbage"}

def detect_labels_bytes(image_bytes):
    response = client.detect_labels(Image={'Bytes': image_bytes}, MaxLabels=10, MinConfidence=70)
    return response['Labels']

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Kamera açılamadı.")
        return

    print("Kamera açıldı. SPACE ile fotoğraf çek, ESC ile çık.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Kare alınamadı.")
            break

        cv2.imshow("TrashSys - Kamera", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC ile çıkış
            break

        if key == 32:  # SPACE ile fotoğraf çek
            # Frame'i jpeg formatına çevir
            ret2, buffer = cv2.imencode('.jpg', frame)
            if not ret2:
                print("Fotoğraf oluşturulamadı.")
                continue

            image_bytes = buffer.tobytes()

            # AWS Rekognition'dan etiketleri al
            labels = detect_labels_bytes(image_bytes)

            # Sonuçları yazdır ve renk seç
            print("Algılanan etiketler:")
            for label in labels:
                name = label['Name']
                confidence = label['Confidence']
                color = "KIRMIZI (Çöp)" if name in trash_labels else "SARI (Çöp değil)"
                print(f"- {name} ({confidence:.1f}%) => {color}")

            # Sonuçları frame üzerine yaz
            y0 = 30
            for i, label in enumerate(labels):
                name = label['Name']
                confidence = label['Confidence']
                color = (0, 0, 255) if name in trash_labels else (0, 255, 255)  # BGR
                text = f"{name} ({confidence:.1f}%)"
                cv2.putText(frame, text, (10, y0 + i * 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            cv2.imshow("TrashSys - Kamera", frame)
            cv2.waitKey(3000)  # 3 saniye göster
            print("Tekrar SPACE ile foto çek, ESC ile çık.")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
