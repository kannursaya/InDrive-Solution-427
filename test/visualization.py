"""
Модуль для визуализации геотреков и создания тепловых карт
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium import plugins
from typing import List, Tuple, Dict
import warnings
warnings.filterwarnings('ignore')

class GeoTrackVisualizer:
    """Класс для создания визуализаций геотреков"""
    
    def __init__(self, data_processor):
        """
        Инициализация визуализатора
        
        Args:
            data_processor: экземпляр GeoTrackProcessor
        """
        self.processor = data_processor
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
    def create_heatmap(self, save_path: str = None) -> folium.Map:
        """
        Создание тепловой карты спроса
        
        Args:
            save_path: путь для сохранения карты
            
        Returns:
            Folium карта с тепловым слоем
        """
        if self.processor.processed_data is None:
            self.processor.anonymize_data()
        
        # Получаем данные для тепловой карты
        heatmap_data = self.processor.get_heatmap_data()
        
        # Создаем базовую карту
        center_lat = self.processor.processed_data['lat'].mean()
        center_lng = self.processor.processed_data['lng'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Добавляем тепловой слой
        heat_data = [[row['center_lat'], row['center_lng'], row['trip_count']] 
                    for _, row in heatmap_data.iterrows()]
        
        plugins.HeatMap(
            heat_data,
            name='Тепловая карта спроса',
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={0.4: 'blue', 0.6: 'cyan', 0.7: 'lime', 0.8: 'yellow', 1.0: 'red'}
        ).add_to(m)
        
        # Добавляем маркеры для точек с высокой активностью
        high_activity = heatmap_data[heatmap_data['trip_count'] > heatmap_data['trip_count'].quantile(0.8)]
        
        for _, row in high_activity.iterrows():
            folium.CircleMarker(
                [row['center_lat'], row['center_lng']],
                radius=8,
                popup=f"Поездок: {row['trip_count']}<br>Средняя скорость: {row['avg_speed']:.1f} км/ч",
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.7
            ).add_to(m)
        
        # Добавляем слой контроля
        folium.LayerControl().add_to(m)
        
        if save_path:
            m.save(save_path)
            print(f"Тепловая карта сохранена: {save_path}")
        
        return m
    
    def create_speed_heatmap(self, save_path: str = None) -> folium.Map:
        """
        Создание тепловой карты скорости
        
        Args:
            save_path: путь для сохранения карты
            
        Returns:
            Folium карта с тепловым слоем скорости
        """
        if self.processor.processed_data is None:
            self.processor.anonymize_data()
        
        heatmap_data = self.processor.get_heatmap_data()
        
        # Создаем базовую карту
        center_lat = self.processor.processed_data['lat'].mean()
        center_lng = self.processor.processed_data['lng'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Добавляем тепловой слой скорости
        speed_data = [[row['center_lat'], row['center_lng'], row['avg_speed']] 
                     for _, row in heatmap_data.iterrows()]
        
        plugins.HeatMap(
            speed_data,
            name='Тепловая карта скорости',
            min_opacity=0.2,
            max_zoom=18,
            radius=25,
            blur=15,
            gradient={0.0: 'green', 0.3: 'yellow', 0.6: 'orange', 1.0: 'red'}
        ).add_to(m)
        
        # Добавляем маркеры для зон с высокой скоростью
        high_speed = heatmap_data[heatmap_data['avg_speed'] > heatmap_data['avg_speed'].quantile(0.8)]
        
        for _, row in high_speed.iterrows():
            folium.CircleMarker(
                [row['center_lat'], row['center_lng']],
                radius=6,
                popup=f"Средняя скорость: {row['avg_speed']:.1f} км/ч<br>Поездок: {row['trip_count']}",
                color='orange',
                fill=True,
                fillColor='orange',
                fillOpacity=0.7
            ).add_to(m)
        
        folium.LayerControl().add_to(m)
        
        if save_path:
            m.save(save_path)
            print(f"Тепловая карта скорости сохранена: {save_path}")
        
        return m
    
    def plot_track_metrics(self, save_path: str = None) -> None:
        """
        Создание графиков метрик треков
        
        Args:
            save_path: путь для сохранения графиков
        """
        metrics = self.processor.calculate_track_metrics()
        
        # Настройка стиля
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Анализ метрик геотреков', fontsize=16, fontweight='bold')
        
        # Распределение расстояний
        axes[0, 0].hist(metrics['total_distance_km'], bins=20, alpha=0.7, color=self.colors[0])
        axes[0, 0].set_title('Распределение расстояний поездок')
        axes[0, 0].set_xlabel('Расстояние (км)')
        axes[0, 0].set_ylabel('Количество поездок')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Распределение скоростей
        axes[0, 1].hist(metrics['avg_speed_kmh'], bins=20, alpha=0.7, color=self.colors[1])
        axes[0, 1].set_title('Распределение средних скоростей')
        axes[0, 1].set_xlabel('Скорость (км/ч)')
        axes[0, 1].set_ylabel('Количество поездок')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Связь между расстоянием и скоростью
        scatter = axes[1, 0].scatter(metrics['total_distance_km'], metrics['avg_speed_kmh'], 
                                   c=metrics['point_count'], cmap='viridis', alpha=0.6)
        axes[1, 0].set_title('Связь между расстоянием и скоростью')
        axes[1, 0].set_xlabel('Расстояние (км)')
        axes[1, 0].set_ylabel('Средняя скорость (км/ч)')
        axes[1, 0].grid(True, alpha=0.3)
        plt.colorbar(scatter, ax=axes[1, 0], label='Количество точек')
        
        # Распределение длительности поездок
        axes[1, 1].hist(metrics['duration_points'], bins=20, alpha=0.7, color=self.colors[2])
        axes[1, 1].set_title('Распределение длительности поездок')
        axes[1, 1].set_xlabel('Количество точек трека')
        axes[1, 1].set_ylabel('Количество поездок')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Графики метрик сохранены: {save_path}")
        
        plt.show()
    
    def create_route_analysis(self, save_path: str = None) -> folium.Map:
        """
        Создание карты с анализом маршрутов
        
        Args:
            save_path: путь для сохранения карты
            
        Returns:
            Folium карта с маршрутами
        """
        if self.processor.processed_data is None:
            self.processor.anonymize_data()
        
        # Создаем базовую карту
        center_lat = self.processor.processed_data['lat'].mean()
        center_lng = self.processor.processed_data['lng'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Группируем данные по трекам
        tracks = self.processor.processed_data.groupby('randomized_id')
        
        # Ограничиваем количество треков для читаемости
        max_tracks = 20
        track_ids = list(tracks.groups.keys())[:max_tracks]
        
        for i, track_id in enumerate(track_ids):
            track_data = tracks.get_group(track_id)
            
            # Сортируем по индексу (предполагаем хронологический порядок)
            track_data = track_data.sort_index()
            
            # Создаем маршрут
            route_coords = [[row['lat'], row['lng']] for _, row in track_data.iterrows()]
            
            # Цвет маршрута зависит от средней скорости
            avg_speed = track_data['spd'].mean()
            if avg_speed < 5:
                color = 'blue'
            elif avg_speed < 15:
                color = 'green'
            else:
                color = 'red'
            
            # Добавляем линию маршрута
            folium.PolyLine(
                route_coords,
                color=color,
                weight=3,
                opacity=0.7,
                popup=f'Трек {track_id}<br>Средняя скорость: {avg_speed:.1f} км/ч'
            ).add_to(m)
            
            # Добавляем маркеры начала и конца
            start_coord = route_coords[0]
            end_coord = route_coords[-1]
            
            folium.Marker(
                start_coord,
                popup=f'Начало трека {track_id}',
                icon=folium.Icon(color='green', icon='play')
            ).add_to(m)
            
            folium.Marker(
                end_coord,
                popup=f'Конец трека {track_id}',
                icon=folium.Icon(color='red', icon='stop')
            ).add_to(m)
        
        # Добавляем легенду
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 200px; height: 90px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Цвета маршрутов:</b></p>
        <p><i class="fa fa-circle" style="color:blue"></i> Медленные (< 5 км/ч)</p>
        <p><i class="fa fa-circle" style="color:green"></i> Средние (5-15 км/ч)</p>
        <p><i class="fa fa-circle" style="color:red"></i> Быстрые (> 15 км/ч)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(save_path)
            print(f"Карта маршрутов сохранена: {save_path}")
        
        return m
    
    def create_anomaly_map(self, save_path: str = None) -> folium.Map:
        """
        Создание карты с аномальными треками
        
        Args:
            save_path: путь для сохранения карты
            
        Returns:
            Folium карта с аномалиями
        """
        anomalies = self.processor.detect_anomalies()
        
        if len(anomalies) == 0:
            print("Аномалии не обнаружены")
            return None
        
        # Создаем базовую карту
        center_lat = self.processor.processed_data['lat'].mean()
        center_lng = self.processor.processed_data['lng'].mean()
        
        m = folium.Map(
            location=[center_lat, center_lng],
            zoom_start=12,
            tiles='OpenStreetMap'
        )
        
        # Добавляем маркеры аномалий
        for _, anomaly in anomalies.iterrows():
            if anomaly['anomaly_type'] == 'high_speed':
                color = 'red'
                icon = 'exclamation-triangle'
            elif anomaly['anomaly_type'] == 'long_distance':
                color = 'orange'
                icon = 'road'
            else:
                color = 'purple'
                icon = 'warning'
            
            folium.Marker(
                [anomaly['start_lat'], anomaly['start_lng']],
                popup=f'''
                <b>Аномальный трек</b><br>
                Тип: {anomaly['anomaly_type']}<br>
                Макс. скорость: {anomaly['max_speed_kmh']:.1f} км/ч<br>
                Расстояние: {anomaly['total_distance_km']:.2f} км<br>
                Длительность: {anomaly['duration_points']} точек
                ''',
                icon=folium.Icon(color=color, icon=icon)
            ).add_to(m)
        
        # Добавляем легенду
        legend_html = '''
        <div style="position: fixed; 
                    bottom: 50px; left: 50px; width: 250px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p><b>Типы аномалий:</b></p>
        <p><i class="fa fa-exclamation-triangle" style="color:red"></i> Высокая скорость</p>
        <p><i class="fa fa-road" style="color:orange"></i> Длинное расстояние</p>
        <p><i class="fa fa-warning" style="color:purple"></i> Оба типа</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        if save_path:
            m.save(save_path)
            print(f"Карта аномалий сохранена: {save_path}")
        
        return m
    
    def create_dashboard_plots(self, save_path: str = None) -> None:
        """
        Создание графиков для дашборда
        
        Args:
            save_path: путь для сохранения графиков
        """
        metrics = self.processor.calculate_track_metrics()
        heatmap_data = self.processor.get_heatmap_data()
        
        # Создаем subplot с 4 графиками
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Распределение скоростей', 'Топ зон по активности', 
                          'Связь скорость-расстояние', 'Временные паттерны'),
            specs=[[{"type": "histogram"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "histogram"}]]
        )
        
        # График 1: Распределение скоростей
        fig.add_trace(
            go.Histogram(x=metrics['avg_speed_kmh'], name='Скорости', nbinsx=20),
            row=1, col=1
        )
        
        # График 2: Топ зон по активности
        top_zones = heatmap_data.nlargest(10, 'trip_count')
        fig.add_trace(
            go.Bar(x=top_zones['trip_count'], y=[f"Зона {i}" for i in range(len(top_zones))], 
                   orientation='h', name='Активность зон'),
            row=1, col=2
        )
        
        # График 3: Связь скорость-расстояние
        fig.add_trace(
            go.Scatter(x=metrics['total_distance_km'], y=metrics['avg_speed_kmh'],
                      mode='markers', name='Треки',
                      marker=dict(size=8, color=metrics['point_count'], 
                                colorscale='Viridis', showscale=True)),
            row=2, col=1
        )
        
        # График 4: Распределение длительности
        fig.add_trace(
            go.Histogram(x=metrics['duration_points'], name='Длительность', nbinsx=20),
            row=2, col=2
        )
        
        # Обновляем макет
        fig.update_layout(
            height=800,
            title_text="Аналитическая панель геотреков",
            title_x=0.5,
            showlegend=False
        )
        
        # Обновляем оси
        fig.update_xaxes(title_text="Скорость (км/ч)", row=1, col=1)
        fig.update_xaxes(title_text="Количество поездок", row=1, col=2)
        fig.update_xaxes(title_text="Расстояние (км)", row=2, col=1)
        fig.update_xaxes(title_text="Количество точек", row=2, col=2)
        
        fig.update_yaxes(title_text="Количество", row=1, col=1)
        fig.update_yaxes(title_text="Зоны", row=1, col=2)
        fig.update_yaxes(title_text="Скорость (км/ч)", row=2, col=1)
        fig.update_yaxes(title_text="Количество", row=2, col=2)
        
        if save_path:
            fig.write_html(save_path)
            print(f"Интерактивные графики сохранены: {save_path}")
        
        fig.show()

if __name__ == "__main__":
    # Демонстрация работы визуализатора
    from data_processor import GeoTrackProcessor
    
    processor = GeoTrackProcessor()
    processor.create_sample_data()
    processor.anonymize_data()
    
    visualizer = GeoTrackVisualizer(processor)
    
    # Создаем различные визуализации
    print("Создание тепловой карты спроса...")
    heatmap = visualizer.create_heatmap('heatmap_demand.html')
    
    print("Создание тепловой карты скорости...")
    speed_heatmap = visualizer.create_speed_heatmap('heatmap_speed.html')
    
    print("Создание карты маршрутов...")
    route_map = visualizer.create_route_analysis('routes.html')
    
    print("Создание карты аномалий...")
    anomaly_map = visualizer.create_anomaly_map('anomalies.html')
    
    print("Создание графиков метрик...")
    visualizer.plot_track_metrics('track_metrics.png')
    
    print("Создание интерактивных графиков...")
    visualizer.create_dashboard_plots('dashboard.html')
    
    print("Все визуализации созданы!")
