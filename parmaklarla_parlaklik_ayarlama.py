#@markdown We implemented some functions to visualize the hand landmark detection results. <br/> Run the following cell to activate the functions.
import screen_brightness_control as sbc
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import numpy as np
import cv2

MARGIN = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
HANDEDNESS_TEXT_COLOR = (88, 205, 54) # vibrant green

def koordinat_getir(landmarks, indeks, h, w):
  landmark = landmarks[indeks]
  return int(landmark.x*w), int(landmark.y*h)

def draw_landmarks_on_image(rgb_image, detection_result):
  # eklem boğumlarının listesi
  hand_landmarks_list = detection_result.hand_landmarks
  # sağ el mi sol el mi mevcut göster
  handedness_list = detection_result.handedness
  annotated_image = np.copy(rgb_image)
  h, w, c = annotated_image.shape

  # Loop through the detected hands to visualize.
  for idx in range(len(hand_landmarks_list)):
    # idx ile belirtilen eldeki boğum noktalarını al
    hand_landmarks = hand_landmarks_list[idx]
    # işaret parmak ucu koordinatları
    x1, y1 = koordinat_getir(hand_landmarks, 8, h, w)
    # baş parmak ucu koordinatları
    x2, y2 = koordinat_getir(hand_landmarks, 4, h, w)
    renk = (255,255,0)
    # işaret parmağı ucunun olduğu yere daire koy
    annotated_image = cv2.circle(annotated_image, (x1,y1), 9, renk , 5)
    # işaret parmağı ucunun olduğu yere daire koy
    annotated_image = cv2.circle(annotated_image, (x2,y2), 9, renk, 5)
    # iki parmak arasına çizgi çiz
    annotated_image = cv2.line(annotated_image, (x1,y1), (x2,y2), renk, 5 )
    xort = (x1+x2)//2
    yort = (y1+y2)//2
    annotated_image = cv2.circle(annotated_image, (xort,yort), 9, (0,255,255), 5)
    # ikiparmak arası mesafe hesapla
    uzaklik = int(np.hypot(x2-x1, y2-y1))

    # Mesafeyi resimde göster
    annotated_image = cv2.putText(annotated_image, str(uzaklik), (xort, yort), cv2.FONT_HERSHEY_COMPLEX, 2, (255,0,0), 4)
    handedness = handedness_list[idx]
    # *** eklenen kısım burası ***
    # Mesafeyi 0-100 arası parlaklığa dönüştür
    min_uzaklik = 25
    max_uzaklik = 200
    parlaklik = int((uzaklik - min_uzaklik) / (max_uzaklik - min_uzaklik) * 100)
    parlaklik = max(0, min(100, parlaklik))
    # Parlaklık uygulama
    sbc.set_brightness(parlaklik)
       
    # Draw the hand landmarks.
    hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    hand_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
    ])
    
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      hand_landmarks_proto,
      solutions.hands.HAND_CONNECTIONS,
      solutions.drawing_styles.get_default_hand_landmarks_style(),
      solutions.drawing_styles.get_default_hand_connections_style())

    # Get the top left corner of the detected hand's bounding box.
    height, width, _ = annotated_image.shape

    x_coordinates = [landmark.x for landmark in hand_landmarks]
    y_coordinates = [landmark.y for landmark in hand_landmarks]
    text_x = int(min(x_coordinates) * width)
    text_y = int(min(y_coordinates) * height) - MARGIN

    # Draw handedness (left or right hand) on the image.
    cv2.putText(annotated_image, f"{handedness[0].category_name}",
                (text_x, text_y), cv2.FONT_HERSHEY_DUPLEX,
                FONT_SIZE, HANDEDNESS_TEXT_COLOR, FONT_THICKNESS, cv2.LINE_AA)

  return annotated_image


# STEP 1: Import the necessary modules.
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# STEP 2: Create an HandLandmarker object.
base_options = python.BaseOptions(model_asset_path='hand_landmarker.task')
options = vision.HandLandmarkerOptions(base_options=base_options,
                                       num_hands=2)
detector = vision.HandLandmarker.create_from_options(options)
# kameradan görüntü oku
cam = cv2.VideoCapture(0)
# kamera açık olduğu sürece
while cam.isOpened():
    # kameradan 1 frame oku
    basari, frame = cam.read()
    # eğer okuma başarılıysa
    if basari:
      frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
      mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)
      # STEP 4: Detect hand landmarks from the input image.
      detection_result = detector.detect(mp_image)

      # STEP 5: Process the classification result. In this case, visualize it.
      annotated_image = draw_landmarks_on_image(mp_image.numpy_view(), detection_result)
      cv2.imshow("Image", cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR))

      key = cv2.waitKey(1)  # 1 ms bekle
      # q tuşuna basıldıysa programı sonlandır
      if key == ord('q') or key == ord('Q'):
         exit(0)