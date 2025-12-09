from flask import Flask, render_template, Response, request, jsonify
import cv2
import time
import mediapipe as mp
import numpy as np
import base64
import os
import json
from datetime import datetime
from flask import send_from_directory

app = Flask(__name__, static_url_path='', static_folder='static')

# Settings
VIDEO_OUTPUT_FOLDER = "recordings"
SAVED_FOLDER = os.path.join(VIDEO_OUTPUT_FOLDER, "saved")
os.makedirs(VIDEO_OUTPUT_FOLDER, exist_ok=True)
os.makedirs(SAVED_FOLDER, exist_ok=True)
TARGET_FPS = 30
MIN_FACE_CONFIDENCE = 0.5
AUTO_RECORD_DELAY = 2.0


class FaceTracker:
    def __init__(self, source="browser"):
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=MIN_FACE_CONFIDENCE,
            min_tracking_confidence=0.5
        )
        self.mp_drawing = mp.solutions.drawing_utils
        self.drawing_spec = mp.solutions.drawing_utils.DrawingSpec(thickness=1, circle_radius=1)
        self.source = source
        self.video_writer = None
        self.frame_count = 0
        self.pTime = 0
        self.wCam, self.hCam = 640, 480
        self.last_frame_time = 0
        self.frame_interval = 1.0 / TARGET_FPS
        self.face_detected_time = 0
        self.recording_started = False
        self.current_output_file = ""
        self.current_json_file = ""
        self.last_settings = {
            'confidence': MIN_FACE_CONFIDENCE,
            'delay': AUTO_RECORD_DELAY
        }

    def update_settings(self, confidence, delay):
        self.last_settings = {
            'confidence': float(confidence),
            'delay': float(delay)
        }
        self.face_mesh.min_detection_confidence = float(confidence)

    def start_recording(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.current_output_file = os.path.join(VIDEO_OUTPUT_FOLDER, f"temp_recording_{timestamp}.mp4")
        self.current_json_file = os.path.join(VIDEO_OUTPUT_FOLDER, f"temp_recording_{timestamp}.json")

        # Initialize video writer with MP4 format
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
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
        video_filename = f"recording_{timestamp}.mp4"
        json_filename = f"recording_{timestamp}.json"

        video_saved_path = os.path.join(SAVED_FOLDER, video_filename)
        json_saved_path = os.path.join(SAVED_FOLDER, json_filename)

        os.rename(self.current_output_file, video_saved_path)
        os.rename(self.current_json_file, json_saved_path)

        self.current_output_file = ""
        self.current_json_file = ""
        return video_saved_path, json_saved_path

    def MPjson(self, face_landmarks, frame_number):
        """Generate JSON data with facial landmarks"""
        landmarks_data = {}
        for id, landmark in enumerate(face_landmarks.landmark):
            landmarks_data[str(id)] = {
                "X": round(landmark.x, 3),
                "Y": round(landmark.y, 3),
                "Z": round(landmark.z, 3)
            }

        return {
            'landmarks': landmarks_data,
            'frame': frame_number,
            'timestamp': time.time()
        }

    def save_json_data(self, json_data):
        """Save JSON data to file"""
        if not self.current_json_file or not self.recording_started:
            return

        try:
            existing_data = []
            if os.path.exists(self.current_json_file):
                with open(self.current_json_file, 'r') as f:
                    existing_data = json.load(f)

            existing_data.append(json_data)

            with open(self.current_json_file, 'w') as f:
                json.dump(existing_data, f, indent=2)
        except Exception as e:
            print(f"Error saving JSON data: {e}")

    def auto_record_control(self, has_face):
        current_time = time.time()

        if has_face:
            if not self.recording_started:
                if self.face_detected_time == 0:
                    self.face_detected_time = current_time
                elif current_time - self.face_detected_time > self.last_settings['delay']:
                    self.start_recording()
            else:
                self.face_detected_time = current_time
        else:
            if self.recording_started and current_time - self.face_detected_time > self.last_settings['delay']:
                self.stop_recording()
            self.face_detected_time = 0

    def process_image(self, img):
        current_time = time.time()

        if current_time - self.last_frame_time < self.frame_interval:
            return img, False, None

        self.last_frame_time = current_time

        img = cv2.resize(img, (self.wCam, self.hCam))
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(img_rgb)

        has_face = results.multi_face_landmarks is not None
        self.auto_record_control(has_face)

        json_data = None
        if has_face:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face landmarks
                self.mp_drawing.draw_landmarks(
                    image=img,
                    landmark_list=face_landmarks,
                    connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=self.drawing_spec
                )

                # Generate JSON data
                json_data = self.MPjson(face_landmarks, self.frame_count)

        # Calculate FPS
        cTime = time.time()
        fps = 1 / (cTime - self.pTime)
        self.pTime = cTime

        # Display status
        status_text = f"FPS: {int(fps)} | "
        status_text += "RECORDING" if self.recording_started else "Ready"
        cv2.putText(img, status_text, (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Write frame if recording
        if self.recording_started and self.video_writer is not None:
            self.video_writer.write(img)
            if json_data:
                self.save_json_data(json_data)
            self.frame_count += 1

        return img, self.recording_started, json_data

    def process_uploaded_frame(self, image_data):
        nparr = np.frombuffer(base64.b64decode(image_data.split(',')[1]), np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        processed_img, is_recording, json_data = self.process_image(img)
        return processed_img, is_recording, json_data


# Global tracker instance
tracker = FaceTracker()


@app.route('/')
def index():
    return render_template('main/face_tracking.html')


@app.route('/process_frame', methods=['POST'])
def process_frame():
    try:
        # Debug: Log that we received a request
        app.logger.debug("Received process_frame request")

        # Get settings from headers with defaults
        confidence = request.headers.get('X-Confidence', str(MIN_FACE_CONFIDENCE))
        delay = request.headers.get('X-Delay', str(AUTO_RECORD_DELAY))

        # Debug: Log received settings
        app.logger.debug(f"Settings - Confidence: {confidence}, Delay: {delay}")

        # Update tracker settings
        tracker.update_settings(confidence, delay)

        # Verify image data exists
        if 'image' not in request.json:
            app.logger.error("No image data in request")
            return jsonify({'status': 'error', 'message': 'No image data'}), 400

        # Debug: Log that we're processing image
        app.logger.debug("Processing image data")

        # Process the uploaded frame
        processed_img, is_recording, json_data = tracker.process_uploaded_frame(request.json['image'])

        # Encode the processed image
        _, buffer = cv2.imencode('.jpg', processed_img)
        if buffer is None:
            raise ValueError("Failed to encode image")

        # Prepare response
        response_data = {
            'status': 'success',
            'processed_image': f"data:image/jpeg;base64,{base64.b64encode(buffer).decode()}",
            'is_recording': is_recording,
            'frame_count': tracker.frame_count,
            'face_data': json_data
        }

        # Debug: Log successful processing
        app.logger.debug("Frame processed successfully")

        return jsonify(response_data)

    except Exception as e:
        # Detailed error logging
        app.logger.error(f"Error processing frame: {str(e)}", exc_info=True)
        return jsonify({
            'status': 'error',
            'message': str(e),
            'details': 'Failed to process frame'
        }), 500


@app.route('/save_recording', methods=['POST'])
def save_recording():
    try:
        frames_recorded = tracker.stop_recording()
        video_saved_path, json_saved_path = tracker.finalize_recording()

        if not video_saved_path:
            return jsonify({'status': 'error', 'message': 'No active recording to save'}), 400

        return jsonify({
            'status': 'success',
            'video_path': video_saved_path,
            'json_path': json_saved_path,
            'frame_count': frames_recorded
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/recordings/<path:filename>')
def download_file(filename):
    return send_from_directory(SAVED_FOLDER, filename, as_attachment=True)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5005, threaded=True, debug=True)