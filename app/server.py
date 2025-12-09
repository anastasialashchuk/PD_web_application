from flask import Flask, render_template, Response
import cv2
import time
import math
import HandTrackingModule as htm

app = Flask(__name__)

class HandTracker:
    def __init__(self):  # Убрали параметр use_ffmpeg
        self.detector = htm.handDetector(detectionCon=0.7)
        self.cap = cv2.VideoCapture("udp://127.0.0.1:1234", cv2.CAP_FFMPEG)
        self.pTime = 0
        self.wCam, self.hCam = 640, 480

        if not self.cap.isOpened():
            print("Ошибка: не удалось открыть видеопоток")
            print("Убедитесь что:")
            print("1. FFmpeg запущен на Windows командой:")
            print('ffmpeg -f dshow -i video="Ваша камера" -vf scale=640:480 -f mpegts udp://127.0.0.1:1234')
            print("2. Брандмауэр разрешает соединения на порту 1234")
            raise RuntimeError("Could not open video source")

    def get_frame(self):
        success, img = self.cap.read()
        if not success:
            print("Не удалось получить кадр с камеры")
            return None

        img = self.detector.findHands(img)
        lmList = self.detector.findPosition(img, draw=True)

        if len(lmList) != 0:
            try:
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                length = math.hypot(x2 - x1, y2 - y1)
                cv2.putText(img, f'Distance: {int(length)}px', (10, 120),
                          cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
            except IndexError:
                pass

        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (10, 70),
                   cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        return img

    def release(self):
        if self.cap.isOpened():
            self.cap.release()

def gen_frames(tracker):
    while True:
        frame = tracker.get_frame()
        if frame is None:
            time.sleep(0.1)
            continue

        _, buffer = cv2.imencode('.jpg', frame)
        yield (b'--frame\r\n'
              b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route('/')
def index():
    return render_template('hand_tracking.html')

@app.route('/video_feed')
def video_feed():
    try:
        tracker = HandTracker()  # Убрали параметр use_ffmpeg
        return Response(gen_frames(tracker),
                      mimetype='multipart/x-mixed-replace; boundary=frame')
    except Exception as e:
        print(f"Ошибка видеопотока: {str(e)}")
        return "Ошибка видеопотока", 500

if __name__ == '__main__':
    print("Запуск сервера...")
    print("Перед запуском убедитесь, что на Windows выполняется:")
    print('ffmpeg -f dshow -i video="Ваша камера" -vf scale=640:480 -f mpegts udp://127.0.0.1:1234')
    app.run(host='0.0.0.0', port=5000, threaded=True)