#!/usr/bin/env python3
"""
Скрипт для быстрого запуска демонстрации анализа геотреков
"""

import subprocess
import sys
import os
import webbrowser
import time
from pathlib import Path

def check_dependencies():
    """Проверка установленных зависимостей"""
    print("🔍 Проверка зависимостей...")
    
    try:
        import pandas
        import numpy
        import matplotlib
        import seaborn
        import plotly
        import folium
        import sklearn
        import dash
        print("✅ Все основные зависимости установлены")
        return True
    except ImportError as e:
        print(f"❌ Отсутствует зависимость: {e}")
        print("📦 Установите зависимости командой: pip install -r requirements.txt")
        return False

def run_demo():
    """Запуск основной демонстрации"""
    print("\n🚀 Запуск демонстрации анализа геотреков...")
    print("=" * 50)
    
    try:
        # Импортируем и запускаем демо
        from main_demo import main
        main()
        return True
    except Exception as e:
        print(f"❌ Ошибка при запуске демонстрации: {e}")
        return False

def run_dashboard():
    """Запуск интерактивной панели"""
    print("\n🌐 Запуск интерактивной панели...")
    print("=" * 50)
    print("Панель будет доступна по адресу: http://localhost:8050")
    print("Нажмите Ctrl+C для остановки")
    
    try:
        from dashboard import GeoTrackDashboard
        dashboard = GeoTrackDashboard()
        dashboard.run(debug=False, port=8050)
    except KeyboardInterrupt:
        print("\n👋 Панель остановлена")
    except Exception as e:
        print(f"❌ Ошибка при запуске панели: {e}")

def open_jupyter():
    """Запуск Jupyter notebook"""
    print("\n📓 Запуск Jupyter notebook...")
    print("=" * 50)
    
    notebook_path = Path("analysis_demo.ipynb")
    if notebook_path.exists():
        try:
            subprocess.run([sys.executable, "-m", "jupyter", "notebook", str(notebook_path)])
        except Exception as e:
            print(f"❌ Ошибка при запуске Jupyter: {e}")
    else:
        print("❌ Файл analysis_demo.ipynb не найден")

def show_results():
    """Показ созданных результатов"""
    print("\n📁 Созданные файлы:")
    print("=" * 50)
    
    output_dir = Path("output")
    if output_dir.exists():
        files = list(output_dir.glob("*"))
        if files:
            for file in sorted(files):
                size = file.stat().st_size
                print(f"   📄 {file.name} ({size:,} байт)")
        else:
            print("   📁 Папка output пуста")
    else:
        print("   📁 Папка output не создана")
    
    # Проверяем основные файлы
    main_files = [
        "heatmap_demand.html",
        "heatmap_speed.html", 
        "routes.html",
        "anomalies.html",
        "dashboard.html",
        "track_metrics.png",
        "anomaly_report.txt",
        "demand_model.pkl"
    ]
    
    print("\n📊 Основные результаты:")
    for file in main_files:
        file_path = output_dir / file
        if file_path.exists():
            print(f"   ✅ {file}")
        else:
            print(f"   ❌ {file}")

def main_menu():
    """Главное меню"""
    while True:
        print("\n" + "=" * 60)
        print("🚗 АНАЛИЗ ГЕОТРЕКОВ ПОЕЗДОК - ГЛАВНОЕ МЕНЮ")
        print("=" * 60)
        print("1. 🔍 Проверить зависимости")
        print("2. 🚀 Запустить полную демонстрацию")
        print("3. 🌐 Запустить интерактивную панель")
        print("4. 📓 Открыть Jupyter notebook")
        print("5. 📁 Показать созданные файлы")
        print("6. ❌ Выход")
        print("=" * 60)
        
        choice = input("Выберите опцию (1-6): ").strip()
        
        if choice == "1":
            check_dependencies()
        elif choice == "2":
            if check_dependencies():
                run_demo()
        elif choice == "3":
            if check_dependencies():
                run_dashboard()
        elif choice == "4":
            if check_dependencies():
                open_jupyter()
        elif choice == "5":
            show_results()
        elif choice == "6":
            print("👋 До свидания!")
            break
        else:
            print("❌ Неверный выбор. Попробуйте снова.")

if __name__ == "__main__":
    print("🚗 Добро пожаловать в систему анализа геотреков inDrive!")
    print("Этот инструмент поможет вам проанализировать обезличенные геотреки")
    print("и создать ценность для вашего сервиса такси/каршеринга.")
    
    # Проверяем, есть ли аргументы командной строки
    if len(sys.argv) > 1:
        if sys.argv[1] == "demo":
            if check_dependencies():
                run_demo()
        elif sys.argv[1] == "dashboard":
            if check_dependencies():
                run_dashboard()
        elif sys.argv[1] == "jupyter":
            if check_dependencies():
                open_jupyter()
        else:
            print("❌ Неизвестная команда. Используйте: demo, dashboard, jupyter")
    else:
        main_menu()
