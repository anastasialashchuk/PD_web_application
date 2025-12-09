import cv2
import time
import math
import HandTrackingModule as htm

def fdist():
    # Настройки камеры
    wCam, hCam = 640, 480

    # Инициализация камеры
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Ошибка: Не удалось открыть камеру")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, wCam)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, hCam)

    pTime = 0
    detector = htm.handDetector(detectionCon=0.7)

    while True:
        success, img = cap.read()
        if not success:
            print("Не удалось получить кадр с камеры")
            continue

        img = detector.findHands(img)
        lmList = detector.findPosition(img, draw=True)

        if len(lmList) != 0:
            try:
                # Кончики большого (4) и указательного (8) пальцев
                x1, y1 = lmList[4][1], lmList[4][2]
                x2, y2 = lmList[8][1], lmList[8][2]

                # Рисуем линию и точки между пальцами
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                cv2.circle(img, (x1, y1), 15, (255, 0, 255), cv2.FILLED)
                cv2.circle(img, (x2, y2), 15, (255, 0, 255), cv2.FILLED)
                cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 15, (255, 0, 255), cv2.FILLED)

                # Вычисляем расстояние между пальцами
                length = math.hypot(x2 - x1, y2 - y1)
                print(f"Расстояние между пальцами: {length:.1f} пикселей")

            except IndexError:
                print("Не обнаружены нужные точки руки")

        # Отображаем FPS
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps)}', (40, 50),
                   cv2.FONT_HERSHEY_COMPLEX, 1, (255, 0, 0), 3)

        cv2.imshow("Hand Tracking", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    fdist()