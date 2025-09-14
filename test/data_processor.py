"""
Модуль для обработки и анализа геотреков поездок
Обеспечивает анонимизацию данных и базовую аналитику
"""

import pandas as pd
import numpy as np
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class GeoTrackProcessor:
    """Класс для обработки геотреков с обеспечением анонимизации"""
    
    def __init__(self, data_path: str = None):
        """
        Инициализация процессора геотреков
        
        Args:
            data_path: путь к файлу с данными
        """
        self.data = None
        self.processed_data = None
        self.anonymization_params = {
            'noise_level': 0.0001,  # Уровень шума для координат
            'time_bucket': 300,     # Временные интервалы в секундах
            'min_points': 3         # Минимальное количество точек для трека
        }
        
        if data_path:
            self.load_data(data_path)
    
    def load_data(self, data_path: str) -> None:
        """Загрузка данных из CSV файла"""
        try:
            self.data = pd.read_csv(data_path)
            print(f"Загружено {len(self.data)} записей")
            self._validate_data()
        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            raise
    
    def create_sample_data(self) -> None:
        """Создание тестовых данных на основе предоставленного примера"""
        sample_data = {
            'randomized_id': [
                7637058049336049989, 1259981924615926140, 1259981924615926140,
                7180852955221959108, -6683155579225977143, -9163058962347897266,
                5965568696283616614, 5965568696283616614, 8912987557492744692,
                8912987557492744692, 8912987557492744692, 8912987557492744692,
                8912987557492744692, 4946418821398555409, -1167793641114800051,
                -1497322719701473200, -1863396568162172735, 1167443348261484352,
                -7848207225311059017, 7333684189108563626
            ],
            'lat': [
                51.09546, 51.0982, 51.09846, 51.0897785, 51.0887817, 51.09454,
                51.09869, 51.10038, 51.0944199, 51.0935884, 51.0927862, 51.0906913,
                51.0988013, 51.1018477, 51.09336, 51.10087, 51.0935, 51.097872,
                51.09696, 51.0997679
            ],
            'lng': [
                71.42753, 71.41295, 71.41212, 71.4284693, 71.4174617, 71.40761,
                71.40596, 71.40663, 71.4043796, 71.4040839, 71.4038026, 71.4030265,
                71.4058674, 71.4135955, 71.42396, 71.41263, 71.4281437, 71.417,
                71.42592, 71.4297613
            ],
            'alt': [
                350.53102, 348.80161, 349.27388, 314, 325.30001831054688, 349.61183,
                351.53058, 350.8547, 321.4000244140625, 321.4000244140625, 321.4000244140625,
                321.4000244140625, 321.4000244140625, 323.789928326188, 348.98415,
                351.17814, 320.60000610351562, 324.9000244140625, 350.78107, 325.31671142578125
            ],
            'spd': [
                0.20681, 0, 4.34501, 14.326102256774902, 0.00060214666882529855, 4.01778,
                11.16366, 12.83405, 15.889509201049805, 15.39943790435791, 14.697993278503418,
                14.682913780212402, 5.5047812461853027, 14.880088806152344, 2.31088,
                0.07523, 0.00712977908551693, 1.9036011695861816, 2.80749, 11.19477653503418
            ],
            'azm': [
                13.60168, 265.677, 307.2453, 192.12367248535156, 0, 170.38123,
                247.88129, 216.54187, 191.99342346191406, 190.94401550292969, 192.68167114257812,
                192.52548217773438, 194.46064758300781, 282.00006103515625, 16.03293,
                136.51695, 0, 192.13235473632812, 204.63948, 281.99520874023438
            ]
        }
        
        # Генерируем дополнительные данные для демонстрации
        np.random.seed(42)
        n_tracks = 50
        n_points_per_track = np.random.randint(5, 20, n_tracks)
        
        all_data = []
        for i in range(n_tracks):
            track_id = np.random.randint(-9223372036854775808, 9223372036854775807)
            start_lat = 51.08 + np.random.uniform(-0.02, 0.02)
            start_lng = 71.40 + np.random.uniform(-0.02, 0.02)
            
            for j in range(n_points_per_track[i]):
                lat = start_lat + np.random.normal(0, 0.005)
                lng = start_lng + np.random.normal(0, 0.005)
                alt = 320 + np.random.uniform(-20, 20)
                spd = np.random.uniform(0, 20)
                azm = np.random.uniform(0, 360)
                
                all_data.append({
                    'randomized_id': track_id,
                    'lat': lat,
                    'lng': lng,
                    'alt': alt,
                    'spd': spd,
                    'azm': azm
                })
        
        self.data = pd.DataFrame(all_data)
        print(f"Создано {len(self.data)} записей для {n_tracks} треков")
    
    def _validate_data(self) -> None:
        """Валидация загруженных данных"""
        required_columns = ['randomized_id', 'lat', 'lng', 'alt', 'spd', 'azm']
        
        for col in required_columns:
            if col not in self.data.columns:
                raise ValueError(f"Отсутствует обязательная колонка: {col}")
        
        # Проверка на наличие данных
        if len(self.data) == 0:
            raise ValueError("Данные пусты")
        
        print("Данные успешно валидированы")
    
    def anonymize_data(self) -> pd.DataFrame:
        """
        Анонимизация данных для защиты приватности
        
        Returns:
            Анонимизированный DataFrame
        """
        if self.data is None:
            raise ValueError("Данные не загружены")
        
        anonymized = self.data.copy()
        
        # Добавляем шум к координатам
        noise_lat = np.random.normal(0, self.anonymization_params['noise_level'], len(anonymized))
        noise_lng = np.random.normal(0, self.anonymization_params['noise_level'], len(anonymized))
        
        anonymized['lat'] += noise_lat
        anonymized['lng'] += noise_lng
        
        # Округляем координаты для дополнительной анонимизации
        anonymized['lat'] = anonymized['lat'].round(5)
        anonymized['lng'] = anonymized['lng'].round(5)
        
        # Округляем скорость
        anonymized['spd'] = anonymized['spd'].round(2)
        
        # Округляем азимут
        anonymized['azm'] = anonymized['azm'].round(1)
        
        self.processed_data = anonymized
        print("Данные анонимизированы")
        return anonymized
    
    def calculate_track_metrics(self) -> pd.DataFrame:
        """
        Расчет метрик для каждого трека
        
        Returns:
            DataFrame с метриками треков
        """
        if self.processed_data is None:
            self.anonymize_data()
        
        track_metrics = []
        
        for track_id in self.processed_data['randomized_id'].unique():
            track_data = self.processed_data[self.processed_data['randomized_id'] == track_id]
            
            if len(track_data) < self.anonymization_params['min_points']:
                continue
            
            # Сортируем по времени (предполагаем, что данные идут в хронологическом порядке)
            track_data = track_data.reset_index(drop=True)
            
            # Расчет расстояния между точками
            distances = []
            for i in range(1, len(track_data)):
                lat1, lng1 = track_data.iloc[i-1]['lat'], track_data.iloc[i-1]['lng']
                lat2, lng2 = track_data.iloc[i]['lat'], track_data.iloc[i]['lng']
                dist = self._haversine_distance(lat1, lng1, lat2, lng2)
                distances.append(dist)
            
            # Метрики трека
            total_distance = sum(distances)
            avg_speed = track_data['spd'].mean()
            max_speed = track_data['spd'].max()
            duration = len(track_data)  # Предполагаем 1 секунда между точками
            start_lat, start_lng = track_data.iloc[0]['lat'], track_data.iloc[0]['lng']
            end_lat, end_lng = track_data.iloc[-1]['lat'], track_data.iloc[-1]['lng']
            
            track_metrics.append({
                'track_id': track_id,
                'total_distance_km': total_distance,
                'avg_speed_kmh': avg_speed,
                'max_speed_kmh': max_speed,
                'duration_points': duration,
                'start_lat': start_lat,
                'start_lng': start_lng,
                'end_lat': end_lat,
                'end_lng': end_lng,
                'point_count': len(track_data)
            })
        
        return pd.DataFrame(track_metrics)
    
    def _haversine_distance(self, lat1: float, lng1: float, lat2: float, lng2: float) -> float:
        """
        Расчет расстояния между двумя точками по формуле Хаверсинуса
        
        Args:
            lat1, lng1: координаты первой точки
            lat2, lng2: координаты второй точки
            
        Returns:
            Расстояние в километрах
        """
        R = 6371  # Радиус Земли в км
        
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lng = np.radians(lng2 - lng1)
        
        a = (np.sin(delta_lat/2)**2 + 
             np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lng/2)**2)
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def get_heatmap_data(self, grid_size: float = 0.001) -> pd.DataFrame:
        """
        Подготовка данных для тепловой карты
        
        Args:
            grid_size: размер ячейки сетки в градусах
            
        Returns:
            DataFrame с данными для тепловой карты
        """
        if self.processed_data is None:
            self.anonymize_data()
        
        # Создаем сетку
        min_lat = self.processed_data['lat'].min()
        max_lat = self.processed_data['lat'].max()
        min_lng = self.processed_data['lng'].min()
        max_lng = self.processed_data['lng'].max()
        
        lat_bins = np.arange(min_lat, max_lat + grid_size, grid_size)
        lng_bins = np.arange(min_lng, max_lng + grid_size, grid_size)
        
        # Группируем точки по ячейкам сетки
        self.processed_data['lat_bin'] = pd.cut(self.processed_data['lat'], lat_bins)
        self.processed_data['lng_bin'] = pd.cut(self.processed_data['lng'], lng_bins)
        
        heatmap_data = self.processed_data.groupby(['lat_bin', 'lng_bin']).agg({
            'randomized_id': 'nunique',  # Количество уникальных треков
            'spd': 'mean',              # Средняя скорость
            'lat': 'mean',              # Центр ячейки по широте
            'lng': 'mean'               # Центр ячейки по долготе
        }).reset_index()
        
        heatmap_data.columns = ['lat_bin', 'lng_bin', 'trip_count', 'avg_speed', 'center_lat', 'center_lng']
        heatmap_data = heatmap_data[heatmap_data['trip_count'] > 0]
        
        return heatmap_data
    
    def detect_anomalies(self) -> pd.DataFrame:
        """
        Обнаружение аномальных треков (необычные маршруты, резкие отклонения)
        
        Returns:
            DataFrame с аномальными треками
        """
        if self.processed_data is None:
            self.anonymize_data()
        
        track_metrics = self.calculate_track_metrics()
        
        # Критерии аномалий
        speed_threshold = track_metrics['max_speed_kmh'].quantile(0.95)
        distance_threshold = track_metrics['total_distance_km'].quantile(0.95)
        
        anomalies = track_metrics[
            (track_metrics['max_speed_kmh'] > speed_threshold) |
            (track_metrics['total_distance_km'] > distance_threshold)
        ].copy()
        
        anomalies['anomaly_type'] = 'high_speed'
        anomalies.loc[anomalies['total_distance_km'] > distance_threshold, 'anomaly_type'] = 'long_distance'
        anomalies.loc[
            (anomalies['max_speed_kmh'] > speed_threshold) & 
            (anomalies['total_distance_km'] > distance_threshold), 
            'anomaly_type'
        ] = 'both'
        
        return anomalies

if __name__ == "__main__":
    # Демонстрация работы процессора
    processor = GeoTrackProcessor()
    processor.create_sample_data()
    
    # Анонимизация данных
    anonymized_data = processor.anonymize_data()
    print(f"Анонимизировано {len(anonymized_data)} записей")
    
    # Расчет метрик
    metrics = processor.calculate_track_metrics()
    print(f"Рассчитаны метрики для {len(metrics)} треков")
    
    # Данные для тепловой карты
    heatmap_data = processor.get_heatmap_data()
    print(f"Подготовлены данные для тепловой карты: {len(heatmap_data)} ячеек")
    
    # Обнаружение аномалий
    anomalies = processor.detect_anomalies()
    print(f"Обнаружено {len(anomalies)} аномальных треков")
