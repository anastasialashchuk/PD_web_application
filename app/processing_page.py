from flask import Blueprint, render_template, jsonify, request, send_from_directory, current_app
import os
import json
from datetime import datetime
import pandas as pd
import matplotlib
from typing import List, Dict  # Добавлен импорт типов

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import shutil
from hydra.utils import instantiate
import sys
from pathlib import Path
import base64
from io import BytesIO
import time
from functools import wraps
from scipy import stats
import seaborn as sns

sys.path.append(str(Path(__file__).parent))

processing_bp = Blueprint('processing', __name__, template_folder='templates')

# Конфигурация для AutoMarking
PROCESSING_CONFIG = {
    'automarking': {
        'dataset_type': ['default'],
        'default': {
            'path_to_directory': 'recordings/saved',
            'folder_name': 'final_recording_',
            'number': []
        },
        'MP': {
            'input_folder': '',
            'output_folder': 'mp_points',
            'exercise': ['FT', 'OC', 'PS'],
            'threshold_data_length': 10,
            'timestamp': 'frame',
            'auto_alg_class': {
                '_target_': 'autoalg.SignalProcessing',
                'frac': 0.1,
                'order_min': 5,
                'order_max': 5
            }
        },
        'mode': ['MP'],
        'image_save': False,  # Теперь сохраняем графики в base64
        'feature_config': {
            'frame_rate': 30,  # Частота кадров в секунду
            'min_frames': 5,  # Минимальное количество кадров для анализа
            'start': 0,  # Начало временного интервала
            'stop': 10,  # Конец временного интервала
            'k': 1  # Коэффициент нормализации
        }
    }
}


def timing_decorator(func):
    """Декоратор для измерения времени выполнения функции"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        if isinstance(result, dict):
            result['processing_time'] = execution_time
        return result

    return wrapper


class Feature():
    def __init__(self, hand_data: list, config: dict):
        self.hand_data = hand_data
        self.config = config
        self.frames = len(hand_data)

    def get_joint_trajectory(self, joint_name: str, coord: str) -> List[float]:
        """Получить траекторию движения конкретного сустава по одной координате"""
        return [frame["left hand"][joint_name][coord] for frame in self.hand_data]

    def get_velocity(self, joint_name: str, coord: str) -> List[float]:
        """Вычислить скорость изменения координаты сустава"""
        trajectory = self.get_joint_trajectory(joint_name, coord)
        if len(trajectory) < 2:
            return []
        return [trajectory[i + 1] - trajectory[i] for i in range(len(trajectory) - 1)]

    def get_min_max_points(self, joint_name: str) -> Dict[str, List[float]]:
        """Получить точки минимумов и максимумов для сустава"""
        x_traj = self.get_joint_trajectory(joint_name, "X")
        y_traj = self.get_joint_trajectory(joint_name, "Y")

        # Простая реализация поиска экстремумов на каждом кадре
        max_points = []
        min_points = []
        for i in range(1, len(y_traj) - 1):
            if (y_traj[i] > y_traj[i - 1] and y_traj[i] > y_traj[i + 1]):
                max_points.append({'X': x_traj[i], 'Y': y_traj[i], 'Type': 1})
            elif (y_traj[i] < y_traj[i - 1] and y_traj[i] < y_traj[i + 1]):
                min_points.append({'X': x_traj[i], 'Y': y_traj[i], 'Type': 0})

        return {'max': max_points, 'min': min_points}


class NumAF(Feature):
    """Количество максимумов"""

    def calc(self, joint_name: str) -> int:
        extrema = self.get_min_max_points(joint_name)
        return len(extrema['max'])


class NumA(Feature):
    """Скорректированное количество максимумов"""

    def calc(self, joint_name: str) -> int:
        extrema = self.get_min_max_points(joint_name)
        max_points = extrema['max']
        if not max_points:
            return 0
        dp = sorted(max_points, key=lambda k: k['X'])[-1]['X']
        if dp < self.config['stop']:
            return round(len(max_points) * self.config['stop'] / dp) if dp != 0 else 0
        else:
            return len(max_points)


class Length(Feature):
    """Длина движения"""

    def calc(self, joint_name: str) -> float:
        extrema = self.get_min_max_points(joint_name)
        min_points = extrema['min']
        if not min_points or len(min_points) < 2:
            return 0.0
        if len(extrema['max']) == 10:
            return min_points[-1]['X'] - min_points[0]['X']
        else:
            length = (min_points[-1]['X'] - min_points[0]['X']) * 10 / len(extrema['max']) if len(
                extrema['max']) != 0 else 0.0
            return length


class AvgFrq(Feature):
    """Средняя частота"""

    def calc(self, joint_name: str) -> float:
        extrema = self.get_min_max_points(joint_name)
        max_points = extrema['max']
        if len(max_points) < 2:
            return 0.0
        result = []
        for i in range(len(max_points) - 1):
            diff = max_points[i + 1]['X'] - max_points[i]['X']
            if diff != 0:
                result.append(1 / diff)
        if not result:
            return 0.0
        return sum(result) / len(result) * self.config['k']


class VarFrq(Feature):
    """Вариабельность частоты"""

    def calc(self, joint_name: str) -> float:
        extrema = self.get_min_max_points(joint_name)
        max_points = extrema['max']
        if len(max_points) < 2:
            return 0.0
        result = []
        for i in range(len(max_points) - 1):
            diff = max_points[i + 1]['X'] - max_points[i]['X']
            if diff != 0:
                result.append(1 / diff)
        if not result or np.mean(result) == 0:
            return 0.0
        return (np.std(result) / np.mean(result)) * self.config['k']


class AvgVopen(Feature):
    """Средняя скорость открытия"""

    def calc(self, joint_name: str) -> float:
        extrema = self.get_min_max_points(joint_name)
        max_points = extrema['max']
        min_points = extrema['min']
        if len(max_points) == 0 or len(min_points) == 0:
            return 0.0
        result = []
        for i in range(min(len(max_points), len(min_points))):
            x_diff = max_points[i]['X'] - min_points[i]['X']
            if x_diff != 0:
                result.append((max_points[i]['Y'] - min_points[i]['Y']) / x_diff)
        if not result:
            return 0.0
        return sum(result) / len(result)


class AvgVclose(Feature):
    """Средняя скорость закрытия"""

    def calc(self, joint_name: str) -> float:
        extrema = self.get_min_max_points(joint_name)
        max_points = extrema['max']
        min_points = extrema['min']
        if len(max_points) < 1 or len(min_points) < 2:
            return 0.0
        result = []
        for i in range(min(len(max_points), len(min_points) - 1)):
            x_diff = min_points[i + 1]['X'] - max_points[i]['X']
            if x_diff != 0:
                result.append((max_points[i]['Y'] - min_points[i + 1]['Y']) / x_diff)
        if not result:
            return 0.0
        return sum(result) / len(result)


class AvgA(Feature):
    """Средняя амплитуда"""

    def calc(self, joint_name: str) -> float:
        extrema = self.get_min_max_points(joint_name)
        max_points = extrema['max']
        min_points = extrema['min']
        if len(max_points) == 0 or len(min_points) == 0:
            return 0.0
        result = []
        for i in range(min(len(max_points), len(min_points))):
            result.append(max_points[i]['Y'] - min_points[i]['Y'])
        for i in range(min(len(max_points), len(min_points) - 1)):
            result.append(max_points[i]['Y'] - min_points[i + 1]['Y'])
        if not result:
            return 0.0
        return sum(result) / len(result)


class VarA(Feature):
    """Вариабельность амплитуды"""

    def calc(self, joint_name: str) -> float:
        extrema = self.get_min_max_points(joint_name)
        max_points = extrema['max']
        min_points = extrema['min']
        if len(max_points) < 2 or len(min_points) < 2:
            return 0.0
        result1 = [max_points[i]['Y'] - min_points[i]['Y']
                   for i in range(min(len(max_points), len(min_points)))]
        result2 = [max_points[i]['Y'] - min_points[i + 1]['Y']
                   for i in range(min(len(max_points), len(min_points) - 1))]

        mean1 = np.mean(result1) if result1 else 0.0
        mean2 = np.mean(result2) if result2 else 0.0

        std1 = (np.std(result1) / mean1) * self.config['k'] if mean1 != 0 else 0.0
        std2 = (np.std(result2) / mean2) * self.config['k'] if mean2 != 0 else 0.0

        return np.mean([std1, std2])


class VarVopen(Feature):
    """Вариабельность скорости открытия"""

    def calc(self, joint_name: str) -> float:
        extrema = self.get_min_max_points(joint_name)
        max_points = extrema['max']
        min_points = extrema['min']
        if len(max_points) < 2 or len(min_points) < 2:
            return 0.0
        result = []
        for i in range(min(len(max_points), len(min_points))):
            x_diff = max_points[i]['X'] - min_points[i]['X']
            if x_diff != 0:
                result.append((max_points[i]['Y'] - min_points[i]['Y']) / x_diff)
        if not result or np.mean(result) == 0:
            return 0.0
        return (np.std(result) / np.mean(result)) * self.config['k']


class VarVclose(Feature):
    """Вариабельность скорости закрытия"""

    def calc(self, joint_name: str) -> float:
        extrema = self.get_min_max_points(joint_name)
        max_points = extrema['max']
        min_points = extrema['min']
        if len(max_points) < 2 or len(min_points) < 3:
            return 0.0
        result = []
        for i in range(min(len(max_points), len(min_points) - 1)):
            x_diff = min_points[i + 1]['X'] - max_points[i]['X']
            if x_diff != 0:
                result.append((max_points[i]['Y'] - min_points[i + 1]['Y']) / x_diff)
        if not result or np.mean(result) == 0:
            return 0.0
        return (np.std(result) / np.mean(result)) * self.config['k']


class DecA(Feature):
    """Декремент амплитуды"""

    def calc(self, joint_name: str) -> float:
        extrema = self.get_min_max_points(joint_name)
        min_points = extrema['min']
        if not min_points:
            return 0.0

        start1 = self.config['start']
        max_x = max(p['X'] for p in min_points)
        stop1 = round((max_x - start1) / 4 + start1)

        # Получаем точки для первого интервала
        maxX1 = [p for p in extrema['max'] if start1 <= p['X'] <= stop1]
        minX1 = [p for p in extrema['min'] if start1 <= p['X'] <= stop1]
        amplitude1 = AvgA(self.hand_data, self.config).calc(joint_name) if maxX1 and minX1 else 0.0

        start4 = round(3 * (max_x - start1) / 4 + start1)
        stop4 = round(max_x + 1)

        # Получаем точки для четвертого интервала
        maxX4 = [p for p in extrema['max'] if start4 <= p['X'] <= stop4]
        minX4 = [p for p in extrema['min'] if start4 <= p['X'] <= stop4]
        amplitude4 = AvgA(self.hand_data, self.config).calc(joint_name) if maxX4 and minX4 else 0.0

        return round(amplitude4 / amplitude1, 4) if amplitude1 != 0 else 0.0


class DataProcessor:
    def __init__(self, config):
        self.config = config
        self.processing_stats = {}  # Для хранения статистики обработки
        self.finger_joints = {
            'Thumb': 'THUMB_TIP',
            'Index': 'FORE_TIP',
            'Middle': 'MIDDLE_TIP',
            'Ring': 'RING_TIP',
            'Little': 'LITTLE_TIP'
        }

    def get_recordings_list(self):
        recordings = []
        path = self.config['automarking']['default']['path_to_directory']
        for file in os.listdir(path):
            if file.endswith('.json'):
                file_path = os.path.join(path, file)
                created = datetime.fromtimestamp(os.path.getctime(file_path)).strftime('%Y-%m-%d %H:%M:%S')
                size = os.path.getsize(file_path) / 1024
                recordings.append({
                    'name': file,
                    'path': file_path,
                    'created': created,
                    'size': f"{size:.2f} KB"
                })
        return sorted(recordings, key=lambda x: x['created'], reverse=True)

    def create_plot_image(self, values, frame, maxP, minP, maxA, minA, title):
        """Создает график и возвращает его в формате base64"""
        try:
            plt.figure(figsize=(10, 5))
            plt.plot(frame, values, label='Signal', linewidth=1)

            if len(maxP) > 0:
                plt.plot(maxP, maxA, 'ro', markersize=5, label='Maxima')
            if len(minP) > 0:
                plt.plot(minP, minA, 'bo', markersize=5, label='Minima')

            plt.xlabel('Frame Number')
            plt.ylabel('Angle (degrees)')
            plt.title(title)
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.tight_layout()

            # Сохраняем график в буфер
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            plt.close()

            # Кодируем в base64
            image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()

            return f"data:image/png;base64,{image_base64}"
        except Exception as e:
            print(f"Error creating plot: {e}")
            return None

    def write_point_hand(self, path, file, maxP, minP, maxA, minA, frac, order_min, order_max):
        datapoint = []
        for i in range(len(maxP)):
            datapoint.append({"Type": 1, "Scale": 1.0, "Brush": "#FFFF0000", "X": float(maxP[i]), "Y": maxA[i]})
        for i in range(len(minP)):
            datapoint.append({"Type": 0, "Scale": 1.0, "Brush": "#FF0000FF", "X": float(minP[i]), "Y": minA[i]})
        datapoint = sorted(datapoint, key=lambda k: k['X'])
        file_point = file.split('.json')[0] + '_'.join(['_point', str(frac), str(order_min), str(order_max)]) + '.json'
        if len(datapoint) != 0:
            if not os.path.isdir(path):
                os.makedirs(path)
            with open(os.path.join(path, file_point), 'w') as f:
                json.dump(datapoint, f)

    def plot_image(self, values, frame, maxPointX, minPointX, maxPointY, minPointY, path_to_save, title):
        try:
            plt.figure(figsize=(12, 6))
            plt.plot(frame, values, label='Signal')
            plt.plot(maxPointX, maxPointY, 'ro', label='Maxima')
            plt.plot(minPointX, minPointY, 'bo', label='Minima')
            plt.xlabel('Frame')
            plt.ylabel('Value')
            plt.title(title)
            plt.legend()
            plt.grid(True)

            os.makedirs(os.path.dirname(path_to_save), exist_ok=True)
            plt.savefig(path_to_save, bbox_inches='tight')
            plt.close()
            return True
        except Exception as e:
            print(f"Error saving plot: {e}")
            return False

    def compute_angle(self, data, point1, point2, vertex_point, y_norm=1):
        x1 = data[point1]['X1'] - data[vertex_point]['X']
        y1 = data[point1]['Y1'] - data[vertex_point]['Y']
        z1 = data[point1]['Z1'] - data[vertex_point]['Z']

        x2 = data[point2]['X1'] - data[vertex_point]['X']
        y2 = data[point2]['Y1'] - data[vertex_point]['Y']
        z2 = data[point2]['Z1'] - data[vertex_point]['Z']

        len1 = np.sqrt(x1 ** 2 + y_norm * y1 ** 2 + z1 ** 2)
        len2 = np.sqrt(x2 ** 2 + y_norm * y2 ** 2 + z2 ** 2)

        cos_val = (x1 * x2 + y1 * y2 + z1 * z2) / (len1 * len2)
        return np.arccos(cos_val) * 180 / np.pi

    def signal_exersice_MP(self, data, hand, exersice):
        if exersice == '1':
            return self.signal_FT_angle(data, hand)
        elif exersice == '2':
            return self.signal_OC_angle(data, hand)
        elif exersice == '3':
            return self.signal_PS_mp(data, hand)
        return [], []

    def signal_FT_angle(self, data, hand):
        frame = []
        values = []
        point1 = 'THUMB_TIP'
        point2 = 'FORE_TIP'
        vertex_point = 'THUMB_MCP'
        for frame_data in data:
            if hand in frame_data.keys():
                angle = self.compute_angle(frame_data[hand], point1, point2, vertex_point)
                values.append(angle)
                frame.append(frame_data['frame'])
        return values, frame

    def signal_OC_angle(self, data, hand):
        frame = []
        values = []
        point1 = 'CENTRE'
        point2 = 'MIDDLE_TIP'
        vertex_point = 'MIDDLE_MCP'
        for frame_data in data:
            if hand in frame_data.keys():
                angle = self.compute_angle(frame_data[hand], point1, point2, vertex_point)
                values.append(angle)
                frame.append(frame_data['frame'])
        return values, frame

    def signal_PS_mp(self, data, hand):
        frame = []
        values = []
        point1 = 'LITTLE_TIP'
        point2 = 'RING_TIP'
        for frame_data in data:
            if hand in frame_data.keys():
                x1 = frame_data[hand][point1]['X1'] - frame_data[hand][point2]['X1']
                y1 = frame_data[hand][point1]['Y1'] - frame_data[hand][point2]['Y1']
                z1 = frame_data[hand][point1]['Z1'] - frame_data[hand][point2]['Z1']

                x2, y2, z2 = 0, 1, 0

                len1 = np.sqrt(x1 ** 2 + y1 ** 2 + z1 ** 2)
                len2 = np.sqrt(x2 ** 2 + y2 ** 2 + z2 ** 2)

                cos_val = (x1 * x2 + y1 * y2 + z1 * z2) / (len1 * len2)
                angle = np.arccos(cos_val) * 180 / np.pi
                values.append(angle)
                frame.append(frame_data['frame'])
        return values, frame

    def signal_MP(self, file, exersice, hand_type):
        hand_dict = {'L': 'left hand', 'R': 'right hand'}
        hand = hand_dict[hand_type]
        with open(file) as f:
            data = json.load(f)
        values, frame = self.signal_exersice_MP(data, hand, exersice)
        if len(values) < self.config['automarking']['MP']['threshold_data_length']:
            del hand_dict[hand_type]
            hand = hand_dict[list(hand_dict.keys())[0]]
            values, frame = self.signal_exersice_MP(data, hand, exersice)
        return values, frame

    def auto_point_MP(self, values, frame):
        auto_alg_class = instantiate(self.config['automarking']['MP']['auto_alg_class'])
        return auto_alg_class.get_point(values, frame)

    @timing_decorator
    def process_recording(self, file_path, exercise='1'):
        """Обрабатывает запись и возвращает данные с графиками в base64"""
        results = []
        exercise_names = {'1': 'Finger Tapping', '2': 'Open/Close', '3': 'Pronation/Supination'}

        try:
            with open(file_path) as f:
                data = json.load(f)

            hand = 'left hand'  # Начинаем с левой руки
            if hand not in data[0]:
                hand = 'right hand'

            values, frame_data = self.signal_exersice_MP(data, hand, exercise)

            if len(values) > self.config['automarking']['MP']['threshold_data_length']:
                # Модифицированный поиск экстремумов на каждом кадре
                maxP, minP = [], []
                maxA, minA = [], []

                for i in range(1, len(values) - 1):
                    if values[i] > values[i - 1] and values[i] > values[i + 1]:
                        maxP.append(frame_data[i])
                        maxA.append(values[i])
                    elif values[i] < values[i - 1] and values[i] < values[i + 1]:
                        minP.append(frame_data[i])
                        minA.append(values[i])

                # Создаем график
                plot_url = self.create_plot_image(
                    values, frame_data,
                    maxP, minP, maxA, minA,
                    f"{exercise_names[exercise]} - {os.path.basename(file_path)}"
                )

                if plot_url:
                    results.append({
                        'exercise': exercise,
                        'exercise_name': exercise_names[exercise],
                        'plot_url': plot_url,
                        'max_points': len(maxP),
                        'min_points': len(minP)
                    })

        except Exception as e:
            print(f"Error processing recording: {e}")

        return {
            'recording': os.path.basename(file_path),
            'results': results
        }

    def calculate_features(self, file_path):
        """Вычисляет параметры движения для файла"""
        try:
            with open(file_path) as f:
                data = json.load(f)

            # Проверяем структуру данных
            if not isinstance(data, list) or len(data) < 1 or "left hand" not in data[0]:
                return None

            config = self.config['automarking']['feature_config']

            # Словарь для хранения значений по каждому параметру для всех пальцев
            file_features = {
                'NumAF': [],
                'NumA': [],
                'Length': [],
                'AvgFrq': [],
                'VarFrq': [],
                'AvgVopen': [],
                'AvgVclose': [],
                'AvgA': [],
                'VarA': [],
                'VarVopen': [],
                'VarVclose': [],
                'DecA': []
            }

            # Вычисляем параметры для каждого пальца
            for finger, joint in self.finger_joints.items():
                file_features['NumAF'].append(NumAF(data, config).calc(joint))
                file_features['NumA'].append(NumA(data, config).calc(joint))
                length = Length(data, config).calc(joint)
                file_features['Length'].append(length if length != 0 else 1.0)  # Избегаем деления на 0
                file_features['AvgFrq'].append(AvgFrq(data, config).calc(joint))
                file_features['VarFrq'].append(VarFrq(data, config).calc(joint))
                file_features['AvgVopen'].append(AvgVopen(data, config).calc(joint))
                file_features['AvgVclose'].append(AvgVclose(data, config).calc(joint))
                file_features['AvgA'].append(AvgA(data, config).calc(joint))
                file_features['VarA'].append(VarA(data, config).calc(joint))
                file_features['VarVopen'].append(VarVopen(data, config).calc(joint))
                file_features['VarVclose'].append(VarVclose(data, config).calc(joint))
                file_features['DecA'].append(DecA(data, config).calc(joint))

            # Вычисляем средние значения по всем пальцам для каждого параметра
            avg_num_af = np.mean(file_features['NumAF']) if file_features['NumAF'] else 0.0
            avg_num_a = np.mean(file_features['NumA']) if file_features['NumA'] else 0.0
            avg_length = np.mean(file_features['Length']) if file_features['Length'] else 1.0

            result = {
                'NumAF': avg_num_af,
                'NumA/Length': avg_num_a / avg_length if avg_length != 0 else 0.0,
                'AvgFrq': np.mean(file_features['AvgFrq']) if file_features['AvgFrq'] else 0.0,
                'VarFrq': np.mean(file_features['VarFrq']) if file_features['VarFrq'] else 0.0,
                'AvgVopen': np.mean(file_features['AvgVopen']) if file_features['AvgVopen'] else 0.0,
                'AvgVclose': np.mean(file_features['AvgVclose']) if file_features['AvgVclose'] else 0.0,
                'AvgA': np.mean(file_features['AvgA']) if file_features['AvgA'] else 0.0,
                'VarA': np.mean(file_features['VarA']) if file_features['VarA'] else 0.0,
                'VarVopen': np.mean(file_features['VarVopen']) if file_features['VarVopen'] else 0.0,
                'VarVclose': np.mean(file_features['VarVclose']) if file_features['VarVclose'] else 0.0,
                'DecA': np.mean(file_features['DecA']) if file_features['DecA'] else 0.0
            }

            return result

        except Exception as e:
            print(f"Error calculating features: {e}")
            return None

    def get_processing_stats(self):
        """Возвращает статистику времени обработки"""
        return self.processing_stats


processor = DataProcessor(PROCESSING_CONFIG)


@processing_bp.route('/')
def processing_page():
    recordings = processor.get_recordings_list()
    return render_template('main/processing.html', recordings=recordings)


@processing_bp.route('/process', methods=['POST'])
def process_file():
    try:
        data = request.json
        file_path = data.get('file_path')
        exercise = data.get('exercise', '1')  # По умолчанию Finger Tapping
        if not file_path or not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404

        result = processor.process_recording(file_path, exercise)
        return jsonify({'status': 'success', 'result': result})

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@processing_bp.route('/calculate_features', methods=['POST'])
def calculate_features():
    try:
        data = request.json
        file_path = data.get('file_path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({'status': 'error', 'message': 'File not found'}), 404

        features = processor.calculate_features(file_path)
        if features is None:
            return jsonify({'status': 'error', 'message': 'Failed to calculate features'}), 400

        return jsonify({
            'status': 'success',
            'features': features,
            'recording': os.path.basename(file_path)
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@processing_bp.route('/static/recordings/saved/processed/<path:filename>')
def serve_processed(filename):
    processed_dir = os.path.join(current_app.root_path, 'static', 'recordings', 'saved', 'processed')
    return send_from_directory(processed_dir, filename)


@processing_bp.route('/download')
def download_file():
    file_path = request.args.get('file')
    if not file_path or not os.path.exists(file_path):
        return "File not found", 404
    return send_from_directory(os.path.dirname(file_path), os.path.basename(file_path), as_attachment=True)


@processing_bp.route('/get_results', methods=['GET'])
def get_results():
    try:
        recording_name = request.args.get('recording')
        if not recording_name:
            return jsonify({'status': 'error', 'message': 'Recording name not provided'}), 400

        base_name = recording_name.split('.json')[0]
        results_dir = os.path.join(PROCESSING_CONFIG['automarking']['default']['path_to_directory'],
                                   'processed', base_name)

        if not os.path.exists(results_dir):
            return jsonify({'status': 'error', 'message': 'Results not found'}), 404

        results = []
        for file in sorted(os.listdir(results_dir)):
            if file.endswith('.png'):
                exercise = file.split('_')[1].split('.')[0]
                points_file = os.path.join(results_dir, f'points_exercise_{exercise}.json')
                results.append({
                    'exercise': exercise,
                    'plot_url': f'/static/recordings/saved/processed/{base_name}/{file}',
                    'points_file': points_file if os.path.exists(points_file) else None
                })

        return jsonify({
            'status': 'success',
            'recording': recording_name,
            'results': results
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)}), 500


@processing_bp.route('/processing_time', methods=['GET'])
def get_processing_time():
    """Возвращает статистику времени обработки записей"""
    stats = processor.get_processing_stats()
    return jsonify({
        'status': 'success',
        'processing_stats': stats
    })