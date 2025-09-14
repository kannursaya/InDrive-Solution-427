"""
Модуль для обнаружения аномалий и анализа узких мест в геотреках
"""

import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from scipy import stats
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

class AnomalyDetector:
    """Класс для обнаружения аномалий в геотреках"""
    
    def __init__(self, data_processor):
        """
        Инициализация детектора аномалий
        
        Args:
            data_processor: экземпляр GeoTrackProcessor
        """
        self.processor = data_processor
        self.anomalies = None
        self.bottlenecks = None
        
    def detect_speed_anomalies(self, speed_threshold: float = None) -> pd.DataFrame:
        """
        Обнаружение аномалий по скорости
        
        Args:
            speed_threshold: порог скорости для аномалий
            
        Returns:
            DataFrame с аномальными треками по скорости
        """
        if self.processor.processed_data is None:
            self.processor.anonymize_data()
        
        metrics = self.processor.calculate_track_metrics()
        
        if speed_threshold is None:
            # Автоматический выбор порога на основе статистики
            speed_threshold = metrics['max_speed_kmh'].quantile(0.95)
        
        speed_anomalies = metrics[metrics['max_speed_kmh'] > speed_threshold].copy()
        speed_anomalies['anomaly_type'] = 'high_speed'
        speed_anomalies['severity'] = speed_anomalies['max_speed_kmh'] / speed_threshold
        
        return speed_anomalies
    
    def detect_distance_anomalies(self, distance_threshold: float = None) -> pd.DataFrame:
        """
        Обнаружение аномалий по расстоянию
        
        Args:
            distance_threshold: порог расстояния для аномалий
            
        Returns:
            DataFrame с аномальными треками по расстоянию
        """
        if self.processor.processed_data is None:
            self.processor.anonymize_data()
        
        metrics = self.processor.calculate_track_metrics()
        
        if distance_threshold is None:
            # Автоматический выбор порога на основе статистики
            distance_threshold = metrics['total_distance_km'].quantile(0.95)
        
        distance_anomalies = metrics[metrics['total_distance_km'] > distance_threshold].copy()
        distance_anomalies['anomaly_type'] = 'long_distance'
        distance_anomalies['severity'] = distance_anomalies['total_distance_km'] / distance_threshold
        
        return distance_anomalies
    
    def detect_pattern_anomalies(self, contamination: float = 0.1) -> pd.DataFrame:
        """
        Обнаружение аномалий с использованием машинного обучения
        
        Args:
            contamination: доля аномалий в данных
            
        Returns:
            DataFrame с аномальными треками по паттернам
        """
        if self.processor.processed_data is None:
            self.processor.anonymize_data()
        
        metrics = self.processor.calculate_track_metrics()
        
        # Подготавливаем признаки для ML
        features = ['total_distance_km', 'avg_speed_kmh', 'max_speed_kmh', 'duration_points']
        X = metrics[features].fillna(0)
        
        # Нормализуем данные
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Используем Isolation Forest для обнаружения аномалий
        iso_forest = IsolationForest(contamination=contamination, random_state=42)
        anomaly_labels = iso_forest.fit_predict(X_scaled)
        
        # Фильтруем аномалии
        pattern_anomalies = metrics[anomaly_labels == -1].copy()
        pattern_anomalies['anomaly_type'] = 'pattern_anomaly'
        pattern_anomalies['severity'] = iso_forest.decision_function(X_scaled[anomaly_labels == -1])
        
        return pattern_anomalies
    
    def detect_bottlenecks(self, grid_size: float = 0.001, min_trips: int = 5) -> pd.DataFrame:
        """
        Обнаружение узких мест (зоны с высокой концентрацией и низкой скоростью)
        
        Args:
            grid_size: размер ячейки сетки
            min_trips: минимальное количество поездок для анализа
            
        Returns:
            DataFrame с узкими местами
        """
        if self.processor.processed_data is None:
            self.processor.anonymize_data()
        
        heatmap_data = self.processor.get_heatmap_data(grid_size)
        
        # Фильтруем зоны с достаточной активностью
        active_zones = heatmap_data[heatmap_data['trip_count'] >= min_trips].copy()
        
        if len(active_zones) == 0:
            return pd.DataFrame()
        
        # Рассчитываем метрики для выявления узких мест
        active_zones['trip_density'] = active_zones['trip_count'] / (grid_size ** 2)
        active_zones['speed_efficiency'] = active_zones['avg_speed'] / active_zones['trip_count']
        
        # Определяем узкие места как зоны с высокой плотностью и низкой скоростью
        density_threshold = active_zones['trip_density'].quantile(0.8)
        speed_threshold = active_zones['avg_speed'].quantile(0.2)
        
        bottlenecks = active_zones[
            (active_zones['trip_density'] > density_threshold) & 
            (active_zones['avg_speed'] < speed_threshold)
        ].copy()
        
        bottlenecks['bottleneck_score'] = (
            bottlenecks['trip_density'] / density_threshold + 
            (speed_threshold - bottlenecks['avg_speed']) / speed_threshold
        ) / 2
        
        return bottlenecks.sort_values('bottleneck_score', ascending=False)
    
    def detect_route_clusters(self, eps: float = 0.002, min_samples: int = 5) -> Dict:
        """
        Обнаружение кластеров маршрутов
        
        Args:
            eps: радиус для DBSCAN
            min_samples: минимальное количество точек в кластере
            
        Returns:
            Словарь с информацией о кластерах
        """
        if self.processor.processed_data is None:
            self.processor.anonymize_data()
        
        # Группируем данные по трекам
        tracks = self.processor.processed_data.groupby('randomized_id')
        
        # Создаем признаки для кластеризации (средние координаты трека)
        track_features = []
        track_ids = []
        
        for track_id, track_data in tracks:
            if len(track_data) < 3:  # Пропускаем слишком короткие треки
                continue
                
            avg_lat = track_data['lat'].mean()
            avg_lng = track_data['lng'].mean()
            avg_speed = track_data['spd'].mean()
            distance = self.processor._haversine_distance(
                track_data['lat'].iloc[0], track_data['lng'].iloc[0],
                track_data['lat'].iloc[-1], track_data['lng'].iloc[-1]
            )
            
            track_features.append([avg_lat, avg_lng, avg_speed, distance])
            track_ids.append(track_id)
        
        if len(track_features) < min_samples:
            return {'clusters': [], 'outliers': track_ids}
        
        # Нормализуем признаки
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(track_features)
        
        # Применяем DBSCAN
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        cluster_labels = dbscan.fit_predict(features_scaled)
        
        # Группируем результаты
        clusters = {}
        outliers = []
        
        for i, (track_id, label) in enumerate(zip(track_ids, cluster_labels)):
            if label == -1:  # Выброс
                outliers.append(track_id)
            else:
                if label not in clusters:
                    clusters[label] = []
                clusters[label].append(track_id)
        
        return {
            'clusters': clusters,
            'outliers': outliers,
            'n_clusters': len(clusters),
            'n_outliers': len(outliers)
        }
    
    def detect_time_anomalies(self, time_window: int = 3600) -> pd.DataFrame:
        """
        Обнаружение временных аномалий (необычные паттерны во времени)
        
        Args:
            time_window: временное окно в секундах
            
        Returns:
            DataFrame с временными аномалиями
        """
        if self.processor.processed_data is None:
            self.processor.anonymize_data()
        
        # Создаем временные метки (предполагаем, что данные идут в хронологическом порядке)
        data_with_time = self.processor.processed_data.copy()
        data_with_time['timestamp'] = pd.date_range(
            start='2024-01-01 00:00:00', 
            periods=len(data_with_time), 
            freq='1S'
        )
        
        # Группируем по временным окнам
        data_with_time['time_window'] = data_with_time['timestamp'].dt.floor(f'{time_window}S')
        
        window_stats = data_with_time.groupby('time_window').agg({
            'randomized_id': 'nunique',
            'spd': ['mean', 'std', 'max'],
            'lat': 'nunique',
            'lng': 'nunique'
        }).reset_index()
        
        window_stats.columns = ['time_window', 'unique_tracks', 'avg_speed', 'speed_std', 'max_speed', 'unique_lats', 'unique_lngs']
        
        # Обнаруживаем аномальные временные окна
        # Аномалии: необычно высокая активность, необычные скорости
        activity_threshold = window_stats['unique_tracks'].quantile(0.95)
        speed_anomaly_threshold = window_stats['max_speed'].quantile(0.95)
        
        time_anomalies = window_stats[
            (window_stats['unique_tracks'] > activity_threshold) |
            (window_stats['max_speed'] > speed_anomaly_threshold)
        ].copy()
        
        time_anomalies['anomaly_type'] = 'time_anomaly'
        time_anomalies['severity'] = np.maximum(
            time_anomalies['unique_tracks'] / activity_threshold,
            time_anomalies['max_speed'] / speed_anomaly_threshold
        )
        
        return time_anomalies
    
    def get_comprehensive_anomalies(self) -> Dict:
        """
        Комплексный анализ всех типов аномалий
        
        Returns:
            Словарь с результатами анализа
        """
        print("Обнаружение аномалий по скорости...")
        speed_anomalies = self.detect_speed_anomalies()
        
        print("Обнаружение аномалий по расстоянию...")
        distance_anomalies = self.detect_distance_anomalies()
        
        print("Обнаружение паттерн-аномалий...")
        pattern_anomalies = self.detect_pattern_anomalies()
        
        print("Обнаружение узких мест...")
        bottlenecks = self.detect_bottlenecks()
        
        print("Анализ кластеров маршрутов...")
        route_clusters = self.detect_route_clusters()
        
        print("Обнаружение временных аномалий...")
        time_anomalies = self.detect_time_anomalies()
        
        # Объединяем все аномалии
        all_anomalies = []
        
        if len(speed_anomalies) > 0:
            all_anomalies.append(speed_anomalies)
        if len(distance_anomalies) > 0:
            all_anomalies.append(distance_anomalies)
        if len(pattern_anomalies) > 0:
            all_anomalies.append(pattern_anomalies)
        
        combined_anomalies = pd.concat(all_anomalies, ignore_index=True) if all_anomalies else pd.DataFrame()
        
        return {
            'speed_anomalies': speed_anomalies,
            'distance_anomalies': distance_anomalies,
            'pattern_anomalies': pattern_anomalies,
            'bottlenecks': bottlenecks,
            'route_clusters': route_clusters,
            'time_anomalies': time_anomalies,
            'combined_anomalies': combined_anomalies,
            'summary': {
                'total_speed_anomalies': len(speed_anomalies),
                'total_distance_anomalies': len(distance_anomalies),
                'total_pattern_anomalies': len(pattern_anomalies),
                'total_bottlenecks': len(bottlenecks),
                'n_route_clusters': route_clusters['n_clusters'],
                'n_route_outliers': route_clusters['n_outliers'],
                'total_time_anomalies': len(time_anomalies)
            }
        }
    
    def generate_anomaly_report(self) -> str:
        """
        Генерация отчета по аномалиям
        
        Returns:
            Текстовый отчет
        """
        results = self.get_comprehensive_anomalies()
        
        report = f"""
=== ОТЧЕТ ПО АНАЛИЗУ АНОМАЛИЙ В ГЕОТРЕКАХ ===

1. АНОМАЛИИ ПО СКОРОСТИ:
   - Обнаружено: {results['summary']['total_speed_anomalies']} треков
   - Средняя скорость аномальных треков: {results['speed_anomalies']['max_speed_kmh'].mean():.1f} км/ч

2. АНОМАЛИИ ПО РАССТОЯНИЮ:
   - Обнаружено: {results['summary']['total_distance_anomalies']} треков
   - Среднее расстояние аномальных треков: {results['distance_anomalies']['total_distance_km'].mean():.2f} км

3. ПАТТЕРН-АНОМАЛИИ (ML):
   - Обнаружено: {results['summary']['total_pattern_anomalies']} треков
   - Использован алгоритм Isolation Forest

4. УЗКИЕ МЕСТА:
   - Обнаружено: {results['summary']['total_bottlenecks']} зон
   - Средний балл узкого места: {results['bottlenecks']['bottleneck_score'].mean():.2f}

5. КЛАСТЕРЫ МАРШРУТОВ:
   - Количество кластеров: {results['summary']['n_route_clusters']}
   - Выбросы: {results['summary']['n_route_outliers']} треков

6. ВРЕМЕННЫЕ АНОМАЛИИ:
   - Обнаружено: {results['summary']['total_time_anomalies']} временных окон

=== РЕКОМЕНДАЦИИ ===
"""
        
        if results['summary']['total_speed_anomalies'] > 0:
            report += "- Проверить треки с высокой скоростью на предмет ошибок GPS\n"
        
        if results['summary']['total_bottlenecks'] > 0:
            report += "- Оптимизировать распределение водителей в узких местах\n"
        
        if results['summary']['n_route_outliers'] > 0:
            report += "- Исследовать необычные маршруты для улучшения планирования\n"
        
        if results['summary']['total_time_anomalies'] > 0:
            report += "- Проанализировать временные паттерны для оптимизации сервиса\n"
        
        return report

if __name__ == "__main__":
    # Демонстрация работы детектора аномалий
    from data_processor import GeoTrackProcessor
    
    processor = GeoTrackProcessor()
    processor.create_sample_data()
    processor.anonymize_data()
    
    detector = AnomalyDetector(processor)
    
    print("Запуск комплексного анализа аномалий...")
    results = detector.get_comprehensive_anomalies()
    
    print("\n" + "="*50)
    print(detector.generate_anomaly_report())
    print("="*50)
    
    # Сохраняем результаты
    if len(results['combined_anomalies']) > 0:
        results['combined_anomalies'].to_csv('anomalies.csv', index=False)
        print("Результаты сохранены в anomalies.csv")
    
    if len(results['bottlenecks']) > 0:
        results['bottlenecks'].to_csv('bottlenecks.csv', index=False)
        print("Узкие места сохранены в bottlenecks.csv")
