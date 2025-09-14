"""
Интерактивная аналитическая панель для анализа геотреков
"""

import dash
from dash import dcc, html, Input, Output, callback
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from data_processor import GeoTrackProcessor
from visualization import GeoTrackVisualizer
from anomaly_detector import AnomalyDetector
import dash_bootstrap_components as dbc

class GeoTrackDashboard:
    """Класс для создания интерактивной панели"""
    
    def __init__(self):
        """Инициализация панели"""
        self.app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
        self.processor = None
        self.visualizer = None
        self.detector = None
        self.setup_layout()
        self.setup_callbacks()
    
    def setup_layout(self):
        """Настройка макета панели"""
        self.app.layout = dbc.Container([
            # Заголовок
            dbc.Row([
                dbc.Col([
                    html.H1("Аналитическая панель геотреков inDrive", 
                           className="text-center mb-4"),
                    html.P("Анализ спроса, маршрутов и оптимизация распределения водителей",
                          className="text-center text-muted mb-4")
                ])
            ]),
            
            # Кнопки управления
            dbc.Row([
                dbc.Col([
                    dbc.Button("Загрузить тестовые данные", id="load-data-btn", 
                              color="primary", className="me-2"),
                    dbc.Button("Обновить анализ", id="refresh-btn", 
                              color="success", className="me-2"),
                    dbc.Button("Экспорт отчета", id="export-btn", 
                              color="info")
                ], className="mb-4")
            ]),
            
            # Индикаторы
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="total-tracks", className="card-title"),
                            html.P("Всего треков", className="card-text")
                        ])
                    ], color="primary", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="total-points", className="card-title"),
                            html.P("Всего точек", className="card-text")
                        ])
                    ], color="success", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="avg-speed", className="card-title"),
                            html.P("Средняя скорость (км/ч)", className="card-text")
                        ])
                    ], color="warning", outline=True)
                ], width=3),
                dbc.Col([
                    dbc.Card([
                        dbc.CardBody([
                            html.H4(id="anomalies-count", className="card-title"),
                            html.P("Аномалий", className="card-text")
                        ])
                    ], color="danger", outline=True)
                ], width=3)
            ], className="mb-4"),
            
            # Основные графики
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Тепловая карта спроса"),
                        dbc.CardBody([
                            dcc.Graph(id="heatmap-graph")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Распределение скоростей"),
                        dbc.CardBody([
                            dcc.Graph(id="speed-distribution")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Анализ маршрутов
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Анализ маршрутов"),
                        dbc.CardBody([
                            dcc.Graph(id="route-analysis")
                        ])
                    ])
                ], width=8),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Топ зоны активности"),
                        dbc.CardBody([
                            dcc.Graph(id="top-zones")
                        ])
                    ])
                ], width=4)
            ], className="mb-4"),
            
            # Аномалии и узкие места
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Обнаруженные аномалии"),
                        dbc.CardBody([
                            dcc.Graph(id="anomalies-graph")
                        ])
                    ])
                ], width=6),
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Узкие места"),
                        dbc.CardBody([
                            dcc.Graph(id="bottlenecks-graph")
                        ])
                    ])
                ], width=6)
            ], className="mb-4"),
            
            # Временной анализ
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Временные паттерны"),
                        dbc.CardBody([
                            dcc.Graph(id="time-patterns")
                        ])
                    ])
                ], width=12)
            ], className="mb-4"),
            
            # Рекомендации
            dbc.Row([
                dbc.Col([
                    dbc.Card([
                        dbc.CardHeader("Рекомендации по оптимизации"),
                        dbc.CardBody([
                            html.Div(id="recommendations")
                        ])
                    ])
                ], width=12)
            ])
        ], fluid=True)
    
    def setup_callbacks(self):
        """Настройка обратных вызовов"""
        
        @self.app.callback(
            [Output("total-tracks", "children"),
             Output("total-points", "children"),
             Output("avg-speed", "children"),
             Output("anomalies-count", "children")],
            [Input("load-data-btn", "n_clicks"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_indicators(load_clicks, refresh_clicks):
            """Обновление индикаторов"""
            if load_clicks is None and refresh_clicks is None:
                return "0", "0", "0.0", "0"
            
            if self.processor is None:
                self.processor = GeoTrackProcessor()
                self.processor.create_sample_data()
                self.processor.anonymize_data()
                self.visualizer = GeoTrackVisualizer(self.processor)
                self.detector = AnomalyDetector(self.processor)
            
            metrics = self.processor.calculate_track_metrics()
            anomalies = self.detector.get_comprehensive_anomalies()
            
            total_tracks = len(metrics)
            total_points = len(self.processor.processed_data)
            avg_speed = metrics['avg_speed_kmh'].mean()
            anomalies_count = len(anomalies['combined_anomalies'])
            
            return (f"{total_tracks}", f"{total_points}", 
                   f"{avg_speed:.1f}", f"{anomalies_count}")
        
        @self.app.callback(
            Output("heatmap-graph", "figure"),
            [Input("load-data-btn", "n_clicks"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_heatmap(load_clicks, refresh_clicks):
            """Обновление тепловой карты"""
            if load_clicks is None and refresh_clicks is None:
                return go.Figure()
            
            if self.processor is None:
                return go.Figure()
            
            heatmap_data = self.processor.get_heatmap_data()
            
            fig = go.Figure(go.Scattermapbox(
                lat=heatmap_data['center_lat'],
                lon=heatmap_data['center_lng'],
                mode='markers',
                marker=dict(
                    size=heatmap_data['trip_count'] * 2,
                    color=heatmap_data['trip_count'],
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Количество поездок")
                ),
                text=heatmap_data['trip_count'],
                hovertemplate="<b>Зона активности</b><br>" +
                             "Поездок: %{text}<br>" +
                             "Средняя скорость: " + heatmap_data['avg_speed'].astype(str) + " км/ч<br>" +
                             "<extra></extra>"
            ))
            
            fig.update_layout(
                mapbox=dict(
                    style="open-street-map",
                    center=dict(
                        lat=heatmap_data['center_lat'].mean(),
                        lon=heatmap_data['center_lng'].mean()
                    ),
                    zoom=11
                ),
                margin=dict(r=0, t=0, l=0, b=0),
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output("speed-distribution", "figure"),
            [Input("load-data-btn", "n_clicks"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_speed_distribution(load_clicks, refresh_clicks):
            """Обновление распределения скоростей"""
            if load_clicks is None and refresh_clicks is None:
                return go.Figure()
            
            if self.processor is None:
                return go.Figure()
            
            metrics = self.processor.calculate_track_metrics()
            
            fig = go.Figure(data=[
                go.Histogram(
                    x=metrics['avg_speed_kmh'],
                    nbinsx=20,
                    marker_color='lightblue',
                    name='Средняя скорость'
                )
            ])
            
            fig.update_layout(
                title="Распределение средних скоростей",
                xaxis_title="Скорость (км/ч)",
                yaxis_title="Количество треков",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output("route-analysis", "figure"),
            [Input("load-data-btn", "n_clicks"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_route_analysis(load_clicks, refresh_clicks):
            """Обновление анализа маршрутов"""
            if load_clicks is None and refresh_clicks is None:
                return go.Figure()
            
            if self.processor is None:
                return go.Figure()
            
            metrics = self.processor.calculate_track_metrics()
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=metrics['total_distance_km'],
                    y=metrics['avg_speed_kmh'],
                    mode='markers',
                    marker=dict(
                        size=metrics['point_count'],
                        color=metrics['max_speed_kmh'],
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Макс. скорость (км/ч)")
                    ),
                    text=metrics['track_id'],
                    hovertemplate="<b>Трек %{text}</b><br>" +
                                 "Расстояние: %{x:.2f} км<br>" +
                                 "Средняя скорость: %{y:.1f} км/ч<br>" +
                                 "Точек: " + metrics['point_count'].astype(str) + "<br>" +
                                 "<extra></extra>"
                )
            ])
            
            fig.update_layout(
                title="Связь между расстоянием и скоростью",
                xaxis_title="Расстояние (км)",
                yaxis_title="Средняя скорость (км/ч)",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output("top-zones", "figure"),
            [Input("load-data-btn", "n_clicks"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_top_zones(load_clicks, refresh_clicks):
            """Обновление топ зон"""
            if load_clicks is None and refresh_clicks is None:
                return go.Figure()
            
            if self.processor is None:
                return go.Figure()
            
            heatmap_data = self.processor.get_heatmap_data()
            top_zones = heatmap_data.nlargest(10, 'trip_count')
            
            fig = go.Figure(data=[
                go.Bar(
                    x=top_zones['trip_count'],
                    y=[f"Зона {i+1}" for i in range(len(top_zones))],
                    orientation='h',
                    marker_color='lightcoral'
                )
            ])
            
            fig.update_layout(
                title="Топ-10 зон по активности",
                xaxis_title="Количество поездок",
                yaxis_title="Зоны",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output("anomalies-graph", "figure"),
            [Input("load-data-btn", "n_clicks"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_anomalies(load_clicks, refresh_clicks):
            """Обновление графика аномалий"""
            if load_clicks is None and refresh_clicks is None:
                return go.Figure()
            
            if self.processor is None:
                return go.Figure()
            
            anomalies = self.detector.get_comprehensive_anomalies()
            
            if len(anomalies['combined_anomalies']) == 0:
                return go.Figure().add_annotation(
                    text="Аномалии не обнаружены",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            anomaly_counts = anomalies['combined_anomalies']['anomaly_type'].value_counts()
            
            fig = go.Figure(data=[
                go.Pie(
                    labels=anomaly_counts.index,
                    values=anomaly_counts.values,
                    hole=0.3
                )
            ])
            
            fig.update_layout(
                title="Распределение типов аномалий",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output("bottlenecks-graph", "figure"),
            [Input("load-data-btn", "n_clicks"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_bottlenecks(load_clicks, refresh_clicks):
            """Обновление графика узких мест"""
            if load_clicks is None and refresh_clicks is None:
                return go.Figure()
            
            if self.processor is None:
                return go.Figure()
            
            bottlenecks = self.detector.detect_bottlenecks()
            
            if len(bottlenecks) == 0:
                return go.Figure().add_annotation(
                    text="Узкие места не обнаружены",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=bottlenecks['center_lng'],
                    y=bottlenecks['center_lat'],
                    mode='markers',
                    marker=dict(
                        size=bottlenecks['bottleneck_score'] * 20,
                        color=bottlenecks['bottleneck_score'],
                        colorscale='Reds',
                        showscale=True,
                        colorbar=dict(title="Балл узкого места")
                    ),
                    text=bottlenecks['trip_count'],
                    hovertemplate="<b>Узкое место</b><br>" +
                                 "Поездок: %{text}<br>" +
                                 "Средняя скорость: " + bottlenecks['avg_speed'].astype(str) + " км/ч<br>" +
                                 "Балл: " + bottlenecks['bottleneck_score'].astype(str) + "<br>" +
                                 "<extra></extra>"
                )
            ])
            
            fig.update_layout(
                title="Обнаруженные узкие места",
                xaxis_title="Долгота",
                yaxis_title="Широта",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output("time-patterns", "figure"),
            [Input("load-data-btn", "n_clicks"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_time_patterns(load_clicks, refresh_clicks):
            """Обновление временных паттернов"""
            if load_clicks is None and refresh_clicks is None:
                return go.Figure()
            
            if self.processor is None:
                return go.Figure()
            
            # Создаем временные данные для демонстрации
            time_data = pd.DataFrame({
                'hour': range(24),
                'activity': np.random.poisson(10, 24) + np.sin(np.arange(24) * np.pi / 12) * 5
            })
            
            fig = go.Figure(data=[
                go.Scatter(
                    x=time_data['hour'],
                    y=time_data['activity'],
                    mode='lines+markers',
                    line=dict(color='blue', width=3),
                    marker=dict(size=8)
                )
            ])
            
            fig.update_layout(
                title="Активность по часам дня",
                xaxis_title="Час дня",
                yaxis_title="Количество поездок",
                height=400
            )
            
            return fig
        
        @self.app.callback(
            Output("recommendations", "children"),
            [Input("load-data-btn", "n_clicks"),
             Input("refresh-btn", "n_clicks")]
        )
        def update_recommendations(load_clicks, refresh_clicks):
            """Обновление рекомендаций"""
            if load_clicks is None and refresh_clicks is None:
                return "Загрузите данные для получения рекомендаций"
            
            if self.processor is None:
                return "Загрузите данные для получения рекомендаций"
            
            anomalies = self.detector.get_comprehensive_anomalies()
            bottlenecks = self.detector.detect_bottlenecks()
            
            recommendations = []
            
            if len(anomalies['speed_anomalies']) > 0:
                recommendations.append(
                    dbc.Alert(
                        f"⚠️ Обнаружено {len(anomalies['speed_anomalies'])} треков с высокой скоростью. "
                        "Рекомендуется проверить качество GPS-данных.",
                        color="warning"
                    )
                )
            
            if len(bottlenecks) > 0:
                recommendations.append(
                    dbc.Alert(
                        f"🚦 Найдено {len(bottlenecks)} узких мест. "
                        "Рекомендуется увеличить количество водителей в этих зонах.",
                        color="info"
                    )
                )
            
            if len(anomalies['route_clusters']['outliers']) > 0:
                recommendations.append(
                    dbc.Alert(
                        f"🛣️ Обнаружено {len(anomalies['route_clusters']['outliers'])} необычных маршрутов. "
                        "Рекомендуется проанализировать для улучшения планирования.",
                        color="secondary"
                    )
                )
            
            if not recommendations:
                recommendations.append(
                    dbc.Alert(
                        "✅ Аномалий не обнаружено. Система работает стабильно.",
                        color="success"
                    )
                )
            
            return recommendations
    
    def run(self, debug=True, port=8050):
        """Запуск панели"""
        print(f"Запуск аналитической панели на http://localhost:{port}")
        self.app.run_server(debug=debug, port=port)

if __name__ == "__main__":
    dashboard = GeoTrackDashboard()
    dashboard.run()
