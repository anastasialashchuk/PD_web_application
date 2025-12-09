import time
import cv2
import mediapipe as mp

# Инициализация MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils


def select_video_source():
    print("\nВыберите источник видео:")
    print("1. Веб-камера")
    print("2. Видеофайл")
    choice = input("Введите номер (1/2): ").strip()

    if choice == "1":
        camera_id = input("Введите номер камеры (обычно 0): ").strip()
        return int(camera_id) if camera_id.isdigit() else 0
    elif choice == "2":
        file_path = input("Введите путь к видеофайлу: ").strip()
        return file_path
    else:
        print("Некорректный выбор. Используется камера 0.")
        return 0


def select_fps_limit():
    print("\nОграничить FPS?")
    print("1. Нет (максимальная скорость)")
    print("2. Да (указать вручную)")
    choice = input("Введите номер (1/2): ").strip()

    if choice == "2":
        fps = input("Введите желаемый FPS (например, 30): ").strip()
        return int(fps) if fps.isdigit() else None
    return None


def main():
    # Выбор источника видео
    video_source = select_video_source()

    # Ограничение FPS
    target_fps = select_fps_limit()
    frame_delay = 1.0 / target_fps if target_fps else 0

    # Инициализация модели
    print("\nИнициализация MediaPipe Hands...")
    start_init = time.time()
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    )
    init_time = (time.time() - start_init) * 1000
    print(f"Модель инициализирована за {init_time:.2f} мс")

    # Открытие видео
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видео источник!")
        return

    # Статистика
    total_frames = 0
    total_processing_time = 0
    min_time = float('inf')
    max_time = 0

    print("\nОбработка начата. Нажмите 'Q' для выхода...")
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Обработка кадра
            start_time = time.perf_counter()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)
            process_time = (time.perf_counter() - start_time) * 1000

            # Обновление статистики
            total_processing_time += process_time
            total_frames += 1
            min_time = min(min_time, process_time)
            max_time = max(max_time, process_time)

            # Отрисовка результатов
            if results.multi_hand_landmarks:
                for landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),
                        mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                    )

            # Вывод информации на кадр
            current_fps = 1000 / process_time if process_time > 0 else 0
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {total_frames}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Time: {process_time:.2f} ms", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # Отображение кадра
            cv2.imshow('MediaPipe Hands', frame)

            # Ограничение FPS
            if frame_delay > 0:
                time.sleep(frame_delay)

            # Выход по 'Q'
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        # Вывод статистики
        if total_frames > 0:
            avg_time = total_processing_time / total_frames
            print("\n--- Статистика ---")
            print(f"Всего кадров: {total_frames}")
            print(f"Среднее время: {avg_time:.2f} мс/кадр")
            print(f"Минимальное время: {min_time:.2f} мс")
            print(f"Максимальное время: {max_time:.2f} мс")

        # Освобождение ресурсов
        cap.release()
        cv2.destroyAllWindows()
        hands.close()


if __name__ == "__main__":
    main()

