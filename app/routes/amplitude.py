
amplitude = Blueprint('amplitude', __name__)
@amplitude.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')
