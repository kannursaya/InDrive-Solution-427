"""
Главный файл для демонстрации решения анализа геотреков
"""

import pandas as pd
import numpy as np
from data_processor import GeoTrackProcessor
from visualization import GeoTrackVisualizer
from anomaly_detector import AnomalyDetector
from demand_predictor import DemandPredictor
import os

def main():
    """Основная функция демонстрации"""
    print("="*60)
    print("🚗 АНАЛИЗ ГЕОТРЕКОВ ПОЕЗДОК - ДЕМОНСТРАЦИЯ inDrive")
    print("="*60)
    
    # 1. Инициализация и загрузка данных
    print("\n1️⃣ ИНИЦИАЛИЗАЦИЯ И ЗАГРУЗКА ДАННЫХ")
    print("-" * 40)
    
    processor = GeoTrackProcessor()
    processor.create_sample_data()
    processor.anonymize_data()
    
    print(f"✅ Загружено {len(processor.processed_data)} точек геоданных")
    print(f"✅ Анонимизация данных завершена")
    
    # 2. Анализ метрик
    print("\n2️⃣ АНАЛИЗ МЕТРИК ТРЕКОВ")
    print("-" * 40)
    
    metrics = processor.calculate_track_metrics()
    print(f"📊 Проанализировано {len(metrics)} треков")
    print(f"📊 Среднее расстояние: {metrics['total_distance_km'].mean():.2f} км")
    print(f"📊 Средняя скорость: {metrics['avg_speed_kmh'].mean():.1f} км/ч")
    print(f"📊 Максимальная скорость: {metrics['max_speed_kmh'].max():.1f} км/ч")
    
    # 3. Создание визуализаций
    print("\n3️⃣ СОЗДАНИЕ ВИЗУАЛИЗАЦИЙ")
    print("-" * 40)
    
    visualizer = GeoTrackVisualizer(processor)
    
    # Создаем папку для результатов
    os.makedirs('output', exist_ok=True)
    
    print("🗺️ Создание тепловой карты спроса...")
    heatmap = visualizer.create_heatmap('output/heatmap_demand.html')
    
    print("🗺️ Создание тепловой карты скорости...")
    speed_heatmap = visualizer.create_speed_heatmap('output/heatmap_speed.html')
    
    print("🗺️ Создание карты маршрутов...")
    route_map = visualizer.create_route_analysis('output/routes.html')
    
    print("📊 Создание графиков метрик...")
    visualizer.plot_track_metrics('output/track_metrics.png')
    
    print("📊 Создание интерактивных графиков...")
    visualizer.create_dashboard_plots('output/dashboard.html')
    
    # 4. Обнаружение аномалий
    print("\n4️⃣ ОБНАРУЖЕНИЕ АНОМАЛИЙ И УЗКИХ МЕСТ")
    print("-" * 40)
    
    detector = AnomalyDetector(processor)
    anomaly_results = detector.get_comprehensive_anomalies()
    
    print(f"🚨 Аномалий по скорости: {anomaly_results['summary']['total_speed_anomalies']}")
    print(f"🚨 Аномалий по расстоянию: {anomaly_results['summary']['total_distance_anomalies']}")
    print(f"🚨 Паттерн-аномалий: {anomaly_results['summary']['total_pattern_anomalies']}")
    print(f"🚦 Узких мест: {anomaly_results['summary']['total_bottlenecks']}")
    print(f"🛣️ Кластеров маршрутов: {anomaly_results['summary']['n_route_clusters']}")
    print(f"🛣️ Выбросов маршрутов: {anomaly_results['summary']['n_route_outliers']}")
    
    # Создаем карту аномалий
    print("🗺️ Создание карты аномалий...")
    anomaly_map = visualizer.create_anomaly_map('output/anomalies.html')
    
    # 5. ML-модель для предсказания спроса
    print("\n5️⃣ МАШИННОЕ ОБУЧЕНИЕ - ПРЕДСКАЗАНИЕ СПРОСА")
    print("-" * 40)
    
    predictor = DemandPredictor(processor)
    
    print("🤖 Подготовка данных для обучения...")
    training_data = predictor.prepare_training_data()
    print(f"📊 Подготовлено {len(training_data)} записей для обучения")
    
    print("🤖 Обучение моделей...")
    ml_results = predictor.train_models()
    best_model = ml_results['best_model']
    print(f"🏆 Лучшая модель: {best_model}")
    
    # Тестируем предсказания
    print("\n🔮 Тестирование предсказаний:")
    test_predictions = [
        (51.095, 71.427, 8, 1),   # Утром в понедельник
        (51.095, 71.427, 14, 1),  # Днем в понедельник
        (51.095, 71.427, 20, 5),  # Вечером в пятницу
        (51.095, 71.427, 12, 6),  # Днем в субботу
    ]
    
    for lat, lng, hour, day in test_predictions:
        pred = predictor.predict_demand(lat, lng, hour, day)
        day_name = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс'][day]
        print(f"   📍 {lat:.3f}, {lng:.3f} | {hour:02d}:00 {day_name} | "
              f"Спрос: {pred['predicted_demand']:.1f} | Уверенность: {pred['confidence']:.2f}")
    
    # 6. Генерация отчета
    print("\n6️⃣ ГЕНЕРАЦИЯ ОТЧЕТА")
    print("-" * 40)
    
    report = detector.generate_anomaly_report()
    print(report)
    
    # Сохраняем отчет
    with open('output/anomaly_report.txt', 'w', encoding='utf-8') as f:
        f.write(report)
    
    # Сохраняем данные
    metrics.to_csv('output/track_metrics.csv', index=False)
    if len(anomaly_results['combined_anomalies']) > 0:
        anomaly_results['combined_anomalies'].to_csv('output/anomalies.csv', index=False)
    if len(anomaly_results['bottlenecks']) > 0:
        anomaly_results['bottlenecks'].to_csv('output/bottlenecks.csv', index=False)
    
    # Сохраняем модель
    predictor.save_model('output/demand_model.pkl')
    
    # 7. Рекомендации по применению
    print("\n7️⃣ РЕКОМЕНДАЦИИ ПО ПРИМЕНЕНИЮ В inDrive")
    print("-" * 40)
    
    recommendations = [
        "🎯 ОПТИМИЗАЦИЯ РАСПРЕДЕЛЕНИЯ ВОДИТЕЛЕЙ:",
        "   • Используйте тепловые карты для определения зон высокого спроса",
        "   • Увеличьте количество водителей в узких местах",
        "   • Планируйте позиционирование водителей на основе предсказаний спроса",
        "",
        "🚦 УЛУЧШЕНИЕ БЕЗОПАСНОСТИ:",
        "   • Мониторьте треки с аномально высокой скоростью",
        "   • Анализируйте необычные маршруты для выявления проблем",
        "   • Настройте алерты на подозрительную активность",
        "",
        "📊 АНАЛИТИКА И ПЛАНИРОВАНИЕ:",
        "   • Используйте временные паттерны для планирования ресурсов",
        "   • Анализируйте кластеры маршрутов для оптимизации зон обслуживания",
        "   • Применяйте ML-модели для прогнозирования спроса",
        "",
        "🔧 ТЕХНИЧЕСКАЯ ИНТЕГРАЦИЯ:",
        "   • Интегрируйте дашборд в существующую систему мониторинга",
        "   • Настройте автоматические уведомления о критических событиях",
        "   • Реализуйте API для доступа к аналитике в реальном времени"
    ]
    
    for rec in recommendations:
        print(rec)
    
    # 8. Итоги
    print("\n8️⃣ ИТОГИ ДЕМОНСТРАЦИИ")
    print("-" * 40)
    
    print("✅ Созданы файлы:")
    print("   📁 output/heatmap_demand.html - Тепловая карта спроса")
    print("   📁 output/heatmap_speed.html - Тепловая карта скорости")
    print("   📁 output/routes.html - Карта маршрутов")
    print("   📁 output/anomalies.html - Карта аномалий")
    print("   📁 output/dashboard.html - Интерактивная панель")
    print("   📁 output/track_metrics.png - Графики метрик")
    print("   📁 output/anomaly_report.txt - Отчет по аномалиям")
    print("   📁 output/demand_model.pkl - ML-модель")
    print("   📁 output/*.csv - Данные для анализа")
    
    print("\n🚀 Для запуска интерактивной панели выполните:")
    print("   python dashboard.py")
    print("   Затем откройте http://localhost:8050 в браузере")
    
    print("\n" + "="*60)
    print("🎉 ДЕМОНСТРАЦИЯ ЗАВЕРШЕНА УСПЕШНО!")
    print("="*60)

if __name__ == "__main__":
    main()
