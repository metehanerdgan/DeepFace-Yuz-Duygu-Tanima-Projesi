import cv2
from deepface import DeepFace

# Önceden eğitilmiş yüz algılama modelini yükle
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Web kameradan video çekimini başlat
cap = cv2.VideoCapture(0)

while True:
    # Web kameradan bir kare oku
    ret, frame = cap.read()

    # Kareyi gri tonlamaya dönüştür
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Gri tonlamalı kareyi tekrar RGB'ye dönüştür
    rgb_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2RGB)

    # Gri tonlamalı karede yüzleri algıla
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Algılanan yüzlerin üzerinden geç
    for (x, y, w, h) in faces:
        # RGB karede yüzün ilgi alanı (ROI) bölgesini çıkar
        face_roi = rgb_frame[y:y + h, x:x + w]

        # Yüz ROI'sini analiz ederek duyguları tespit et
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)

        # Tespit edilen baskın duyguyu al
        duygu = result[0]['dominant_emotion']

        # Orijinal karede yüzün etrafına bir dikdörtgen çiz
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        # Yüz dikdörtgeninin üstüne tespit edilen duygu metnini yaz
        cv2.putText(frame, duygu, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Duyguların gösterildiği kareyi ekranda göster
    cv2.imshow('Gerçek Zamanlı Duygu Tespiti', frame)

    # 'q' tuşuna basılırsa döngüyü kır ve işlemi sonlandır
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Video çekimini serbest bırak ve tüm pencereleri kapat
cap.release()
cv2.destroyAllWindows()
