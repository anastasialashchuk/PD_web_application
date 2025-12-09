from flask import Flask, render_template, Response, request, jsonify
import cv2
import time
import math
import mediapipe as mp
import numpy as np
import base64
import os
import json
from datetime import datetime
from .processing_page import processing_bp
from flask import Flask, send_from_directory

app = Flask(__name__, static_url_path='', static_folder='static')
# Инициализация Flask
#app = Flask(__name__)
app.register_blueprint(processing_bp, url_prefix='/processing')
# Настройки
VIDEO_OUTPUT_FOLDER = "recordings"
SAVED_FOLDER = os.path.join(VIDEO_OUTPUT_FOLDER, "saved")
os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SAVED_FOLDER, exist_ok=True)
TARGET_FPS = 30
MIN_HAND_CONFIDENCE = 0.6
AUTO_RECORD_DELAY = 2.0


class HandTracker:
    def __init__(self, source="browser"):
        self.hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=MIN_HAND_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.source = source
        self.video_writer = None
        self.json_writer = None
        self.frame_count = 0
        self.pTime = 0
        self.wCam, self.hCam = 640, 480
        self.last_frame_time = 0
        self.frame_interval = 1.0 / TARGET_FPS
        self.hand_detected_time = 0
        self.recording_started = False
        self.current_output_file = ""
        self.current_json_file = ""
        self.last_settings = {
            'confidence': MIN_HAND_CONFIDENCE,
            'delay': AUTO_RECORD_DELAY
        }

    def update_settings(self, confidence, delay):
        self.last_settings = {
            'confidence': float(confidence),
            'delay': float(delay)
        }
        self.hands.min_detection_confidence = float(confidence)

    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_output_file = os.path.join(VIDEO_OUTPUT_FOLDER, f"temp_recording_{timestamp}.avi")
        self.current_json_file = os.path.join(VIDEO_OUTPUT_FOLDER, f"temp_recording_{timestamp}.json")

        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        self.video_writer = cv2.VideoWriter(self.current_output_file, fourcc, TARGET_FPS, (self.wCam, self.hCam))

        # Initialize JSON file with empty list
        with open(self.current_json_file, 'w') as f:
            json.dump([], f)

        self.frame_count = 0
        self.recording_started = True
        return self.current_output_file

    def stop_recording(self):
        if self.video_writer is not None:
            self.video_writer.release()
            self.video_writer = None

        frames_recorded = self.frame_count
        self.frame_count = 0
        self.recording_started = False
        return frames_recorded

    def finalize_recording(self):
        if not self.current_output_file or not os.path.exists(self.current_output_file):
            return None, None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_filename = f"final_recording_{timestamp}.avi"
        json_filename = f"final_recording_{timestamp}.json"

        video_saved_path = os.path.join(SAVED_FOLDER, video_filename)
        json_saved_path = os.path.join(SAVED_FOLDER, json_filename)

        os.rename(self.current_output_file, video_saved_path)
        os.rename(self.current_json_file, json_saved_path)

        self.current_output_file = ""
        self.current_json_file = ""
        return video_saved_path, json_saved_path

    def MPjson(self, frame, hand_landmarks, handedness, frame_number):
        """Функция для формирования JSON данных аналогично LeapMotion"""
        POINTS = ['CENTRE', 'THUMB_MCP', 'THUMB_PIP', 'THUMB_DIP', 'THUMB_TIP',
                  'FORE_MCP', 'FORE_PIP', 'FORE_DIP', 'FORE_TIP',
                  'MIDDLE_MCP', 'MIDDLE_PIP', 'MIDDLE_DIP', 'MIDDLE_TIP',
                  'RING_MCP', 'RING_PIP', 'RING_DIP', 'RING_TIP',
                  'LITTLE_MCP', 'LITTLE_PIP', 'LITTLE_DIP', 'LITTLE_TIP']

        hand_label = handedness.classification[0].label
        translate_hand = {'Left': 'right hand', 'Right': 'left hand'}

        dict_points = {}
        for id, landmark in enumerate(hand_landmarks.landmark):
            if id == 0:  # CENTRE point
                cords = {
                    "X": round(landmark.x, 3),
                    "Y": round(landmark.y, 3),
                    "Z": round(landmark.z, 3),
                    "X1": round(landmark.x, 3),
                    "Y1": round(landmark.y, 3),
                    "Z1": round(landmark.z, 3),
                    "W": 0,
                    "Wx": 0,
                    "Wy": 0,
                    "Wz": 0,
                    "Angle": 0
                }
            else:
                cords = {
                    "X1": round(landmark.x, 3),
                    "Y1": round(landmark.y, 3),
                    "Z1": round(landmark.z, 3),
                    "X": round(landmark.x, 3),
                    "Y": round(landmark.y, 3),
                    "Z": round(landmark.z, 3),
                    "W": 0,
                    "Angle": 0
                }
            dict_points[POINTS[id]] = cords

        return {
            translate_hand[hand_label]: dict_points,
            'frame': frame_number
        }

    def save_json_data(self, json_data):
        """Сохраняет данные JSON в файл"""
        if not self.current_json_file or not self.recording_started:
            return

        try:
            # Read existing data
            existing_data = []
            if os.path.exists(self.current_json_file):
                with open(self.current_json_file, 'r') as f:
                    existing_data = json.load(f)

            # Append new data
            existing_data.extend(json_data)

            # Write back to file
            with open(self.current_json_file, 'w') as f:
                json.dump(existing_data, f)
        except Exception as e:
            print(f"Error saving JSON data: {e}")

    def auto_record_control(self, has_hand):
        current_time = time.time()

        if has_hand:
            if not self.recording_started:
                if self.hand_detected_time == 0:
                    self.hand_detected_time = current_time
                elif current_time - self.hand_detected_time > self.last_settings['delay']:
                    if not self.recording_started:
                        self.start_recording()
            else:
                self.hand_detected_time = current_time
        else:
            if self.recording_started and current_time - self.hand_detected_time > self.last_settings['delay']:
                self.stop_recording()
            self.hand_detected_time = 0

    def process_image(self, img):
        current_time = time.time()

        if current_time - self.last_frame_time < self.frame_interval:
            return img, False, []

        self.last_frame_time = current_time

        img = cv2.resize(img, (self.wCam, self.hCam))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)

        has_hand = results.multi_hand_landmarks is not None
        self.auto_record_control(has_hand)

        json_data = []
        if has_hand:
            for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
                self.mp_draw.draw_landmarks(
                    img, hand_landmarks, mp.solutions.hands.HAND_CONNECTIONS,
                    self.mp_draw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                    self.mp_draw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2)
                )

                # Generate JSON data
                json_data.append(self.MPjson(img, hand_landmarks, handedness, self.frame_count))

                h, w, c = img.shape
                landmarks = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]

                try:
                    thumb = landmarks[4]
                    index = landmarks[8]

                    cv2.circle(img, thumb, 15, (255, 0, 255), cv2.FILLED)
                    cv2.circle(img, index, 15, (255, 0, 255), cv2.FILLED)
                    cv2.line(img, thumb, index, (255, 0, 255), 3)

                    length = math.hypot(index[0] - thumb[0], index[1] - thumb[1])
                    cv2.putText(img, f'Distance: {int(length)}px', (10, 120),
                                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
                except IndexError:
                    pass

        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime

        status_text = f"FPS: {int(fps)} | "
        status_text += "RECORDING" if self.recording_started else "Waiting for hand..."
        cv2.putText(img, status_text, (10, 70),
                    cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)

        if self.recording_started and self.video_writer is not None:
            self.video_writer.write(img)
            if json_data:  # Only save if we have hand data
                self.save_json_data(json_data)
            self.frame_count += 1

        return img, self.recording_started, json_data

    def process_uploaded_frame(self, image_data):
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_img, is_recording, json_data = self.process_image(img)
        return processed_img, is_recording, json_data


# Глобальный экземпляр трекера
tracker = HandTracker()


@app.route('/')
def index():
    return render_template('main/hand_tracking.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        confidence = request.headers.get('X-Confidence', MIN_HAND_CONFIDENCE)
        delay = request.headers.get('X-Delay', AUTO_RECORD_DELAY)
        tracker.update_settings(confidence, delay)

        data = request.json
        if 'image' not in data:
            return jsonify({'status': 'error', 'message': 'No image data'}), 400

        processed_img, is_recording, json_data = tracker.process_uploaded_frame(data['image'])
        _, buffer = cv2.imencode('.jpg', processed_img)

        response_data = {
            'status': 'success',
            'processed_image': f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}",
            'is_recording': is_recording,
            'frame_count': tracker.frame_count,
            'hand_data': json_data
        }

        if is_recording:
            response_data['output_file'] = tracker.current_output_file

        return jsonify(response_data)
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/save_recording', methods=['POST'])
def save_recording():
    try:
        # Finalize video and JSON recording
        frames_recorded = tracker.stop_recording()
        video_saved_path, json_saved_path = tracker.finalize_recording()

        if not video_saved_path:
            return jsonify({'status': 'error', 'message': 'No active recording to save'}), 400

        return jsonify({
            'status': 'success',
            'video_saved_path': video_saved_path,
            'json_saved_path': json_saved_path,
            'frame_count': frames_recorded,
            'message': 'Recording saved successfully'
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, threaded=True, debug=True)