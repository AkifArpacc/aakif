import screen_brightness_control as sbc
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2
import os
import random

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54)

# === GÖRSELLERİ YÜKLE ===
RESIM_KLASORU = "./resimler"
RESIM_BOYUTU = (60, 60)  # küçük ikonlar
resim_dosyalar = [f for f in os.listdir(RESIM_KLASORU) if f.endswith(('.png', '.jpg', '.jpeg'))]
yuklenen_resimler = []
for dosya in resim_dosyalar:
    yol = os.path.join(RESIM_KLASORU, dosya)
    img = cv2.imread(yol, cv2.IMREAD_UNCHANGED)
    if img is not None:
        img = cv2.resize(img, RESIM_BOYUTU, interpolation=cv2.INTER_AREA)
        yuklenen_resimler.append(img)

# === GÖRSELLERİN KONUMU ===
kullanilabilir_resimler = yuklenen_resimler.copy()
aktif_resim = None
aktif_konum = None

# === FONKSIYONLAR ===
def koordinat_getir(landmarks, indeks, h, w):
    landmark = landmarks[indeks]
    return int(landmark.x * w), int(landmark.y * h)

def rastgele_konum(ekran_boyutu):
    w, h = ekran_boyutu
    return random.randint(0, w - RESIM_BOYUTU[0]), random.randint(0, h - RESIM_BOYUTU[1])

def resim_yerlestir(ekran, resim, konum):
    x, y = konum
    h, w = resim.shape[:2]
    if resim.shape[2] == 4:
        alpha = resim[:, :, 3] / 255.0
        for c in range(3):
            ekran[y:y+h, x:x+w, c] = (1 - alpha) * ekran[y:y+h, x:x+w, c] + alpha * resim[:, :, c]
    else:
        ekran[y:y+h, x:x+w] = resim

def parmak_resme_degdi_mi(parmak_x, parmak_y, konum):
    x, y = konum
    return x <= parmak_x <= x + RESIM_BOYUTU[0] and y <= parmak_y <= y + RESIM_BOYUTU[1]

def draw_landmarks_on_image(rgb_image, detection_result):
    global aktif_resim, aktif_konum, yuklenen_resimler  # yuklenen_resimler'i kullanalım

    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)
    h, w, c = annotated_image.shape

    # Eğer aktif resim yoksa, yeni bir resim ve konum seç
    if aktif_resim is None:
        aktif_resim = random.choice(yuklenen_resimler)  # Her seferinde yeni bir resim seç
        aktif_konum = rastgele_konum((w, h))  # Rastgele bir konum belirle

    for idx in range(len(hand_landmarks_list)):
        hand_landmarks = hand_landmarks_list[idx]
        x1, y1 = koordinat_getir(hand_landmarks, 8, h, w)  # işaret parmağı ucu

        if aktif_resim is not None and aktif_konum is not None:
            if parmak_resme_degdi_mi(x1, y1, aktif_konum):
                # Eğer parmak resme dokunursa, resmi kaldır
                aktif_resim = None
                aktif_konum = None
                break  # sadece bir resim kaldırılacak

    if aktif_resim is not None and aktif_konum is not None:
        resim_yerlestir(annotated_image, aktif_resim, aktif_konum)

    return annotated_image


# === MEDIAPIPE MODEL YÜKLE ===
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options, num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)

# === KAMERA ===
cam = cv2.VideoCapture(0)
while cam.isOpened():
    basari, frame = cam.read()
    if basari:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
        detection_result = detector.detect(mp_image)
        annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
        cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

        key = cv2.waitKey(1)
        if key == ord('q') or key == ord('Q'):
            break

cam.release()
cv2.destroyAllWindows()
