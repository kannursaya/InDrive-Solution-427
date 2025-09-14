"""
ML-модель для предсказания спроса на основе геотреков
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

class DemandPredictor:
    """Класс для предсказания спроса на основе геотреков"""
    
    def __init__(self, data_processor):
        """
        Инициализация предсказателя спроса
        
        Args:
            data_processor: экземпляр GeoTrackProcessor
        """
        self.processor = data_processor
        self.models = {}
        self.scalers = {}
        self.feature_importance = {}
        self.training_data = None
        self.grid_size = 0.001  # Размер ячейки сетки
        
    def prepare_training_data(self) -> pd.DataFrame:
        """
        Подготовка данных для обучения модели
        
        Returns:
            DataFrame с признаками для обучения
        """
        if self.processor.processed_data is None:
            self.processor.anonymize_data()
        
        # Создаем сетку для анализа
        heatmap_data = self.processor.get_heatmap_data(self.grid_size)
        
        # Добавляем временные признаки
        heatmap_data['hour'] = np.random.randint(0, 24, len(heatmap_data))  # Симуляция часа
        heatmap_data['day_of_week'] = np.random.randint(0, 7, len(heatmap_data))  # Симуляция дня недели
        heatmap_data['is_weekend'] = (heatmap_data['day_of_week'] >= 5).astype(int)
        
        # Добавляем географические признаки
        heatmap_data['distance_from_center'] = np.sqrt(
            (heatmap_data['center_lat'] - heatmap_data['center_lat'].mean())**2 +
            (heatmap_data['center_lng'] - heatmap_data['center_lng'].mean())**2
        )
        
        # Добавляем признаки плотности
        heatmap_data['area'] = self.grid_size ** 2
        heatmap_data['density'] = heatmap_data['trip_count'] / heatmap_data['area']
        
        # Добавляем признаки скорости
        heatmap_data['speed_variance'] = np.random.uniform(0, 10, len(heatmap_data))  # Симуляция
        
        # Целевая переменная - количество поездок
        self.training_data = heatmap_data.copy()
        
        return self.training_data
    
    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Создание признаков для модели
        
        Args:
            data: исходные данные
            
        Returns:
            DataFrame с признаками
        """
        features = data.copy()
        
        # Временные признаки
        features['hour_sin'] = np.sin(2 * np.pi * features['hour'] / 24)
        features['hour_cos'] = np.cos(2 * np.pi * features['hour'] / 24)
        features['day_sin'] = np.sin(2 * np.pi * features['day_of_week'] / 7)
        features['day_cos'] = np.cos(2 * np.pi * features['day_of_week'] / 7)
        
        # Географические признаки
        features['lat_norm'] = (features['center_lat'] - features['center_lat'].min()) / \
                              (features['center_lat'].max() - features['center_lat'].min())
        features['lng_norm'] = (features['center_lng'] - features['center_lng'].min()) / \
                              (features['center_lng'].max() - features['center_lng'].min())
        
        # Взаимодействия
        features['hour_density'] = features['hour'] * features['density']
        features['weekend_density'] = features['is_weekend'] * features['density']
        
        return features
    
    def train_models(self, test_size: float = 0.2) -> Dict:
        """
        Обучение моделей для предсказания спроса
        
        Args:
            test_size: доля тестовых данных
            
        Returns:
            Словарь с результатами обучения
        """
        if self.training_data is None:
            self.prepare_training_data()
        
        # Подготавливаем признаки
        features_data = self.create_features(self.training_data)
        
        # Выбираем признаки для обучения
        feature_columns = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'lat_norm', 'lng_norm', 'distance_from_center',
            'density', 'avg_speed', 'speed_variance',
            'is_weekend', 'hour_density', 'weekend_density'
        ]
        
        X = features_data[feature_columns]
        y = features_data['trip_count']
        
        # Разделяем на обучающую и тестовую выборки
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Нормализуем признаки
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        self.scalers['main'] = scaler
        
        # Обучаем различные модели
        models = {
            'RandomForest': RandomForestRegressor(n_estimators=100, random_state=42),
            'GradientBoosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'LinearRegression': LinearRegression()
        }
        
        results = {}
        
        for name, model in models.items():
            print(f"Обучение модели {name}...")
            
            # Обучаем модель
            if name == 'LinearRegression':
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
            else:
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
            
            # Оцениваем качество
            mse = mean_squared_error(y_test, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            # Кросс-валидация
            if name == 'LinearRegression':
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
            else:
                cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            results[name] = {
                'model': model,
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std()
            }
            
            # Важность признаков
            if hasattr(model, 'feature_importances_'):
                self.feature_importance[name] = dict(zip(feature_columns, model.feature_importances_))
            elif hasattr(model, 'coef_'):
                self.feature_importance[name] = dict(zip(feature_columns, model.coef_))
            
            print(f"{name} - R²: {r2:.3f}, RMSE: {rmse:.3f}, CV: {cv_scores.mean():.3f}±{cv_scores.std():.3f}")
        
        # Выбираем лучшую модель
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2'])
        self.models = {name: results[name]['model'] for name in results.keys()}
        
        print(f"\nЛучшая модель: {best_model_name} (R² = {results[best_model_name]['r2']:.3f})")
        
        return {
            'results': results,
            'best_model': best_model_name,
            'feature_importance': self.feature_importance,
            'test_data': (X_test, y_test)
        }
    
    def predict_demand(self, lat: float, lng: float, hour: int, 
                      day_of_week: int, model_name: str = None) -> Dict:
        """
        Предсказание спроса для заданной точки и времени
        
        Args:
            lat: широта
            lng: долгота
            hour: час дня (0-23)
            day_of_week: день недели (0-6)
            model_name: название модели (если None, используется лучшая)
            
        Returns:
            Словарь с предсказанием и метриками
        """
        if not self.models:
            raise ValueError("Модели не обучены. Сначала вызовите train_models()")
        
        if model_name is None:
            model_name = list(self.models.keys())[0]
        
        # Подготавливаем данные для предсказания
        pred_data = pd.DataFrame({
            'center_lat': [lat],
            'center_lng': [lng],
            'hour': [hour],
            'day_of_week': [day_of_week],
            'is_weekend': [1 if day_of_week >= 5 else 0],
            'avg_speed': [15.0],  # Средняя скорость по умолчанию
            'speed_variance': [5.0]  # Дисперсия скорости по умолчанию
        })
        
        # Добавляем дополнительные признаки
        pred_data['distance_from_center'] = np.sqrt(
            (lat - self.training_data['center_lat'].mean())**2 +
            (lng - self.training_data['center_lng'].mean())**2
        )
        pred_data['area'] = self.grid_size ** 2
        pred_data['density'] = 0.1  # Базовая плотность
        
        # Создаем признаки
        features_data = self.create_features(pred_data)
        
        # Выбираем признаки
        feature_columns = [
            'hour_sin', 'hour_cos', 'day_sin', 'day_cos',
            'lat_norm', 'lng_norm', 'distance_from_center',
            'density', 'avg_speed', 'speed_variance',
            'is_weekend', 'hour_density', 'weekend_density'
        ]
        
        X_pred = features_data[feature_columns]
        
        # Предсказание
        model = self.models[model_name]
        
        if model_name == 'LinearRegression':
            X_pred_scaled = self.scalers['main'].transform(X_pred)
            prediction = model.predict(X_pred_scaled)[0]
        else:
            prediction = model.predict(X_pred)[0]
        
        # Ограничиваем предсказание разумными значениями
        prediction = max(0, min(prediction, 100))  # От 0 до 100 поездок
        
        return {
            'predicted_demand': prediction,
            'model_used': model_name,
            'location': {'lat': lat, 'lng': lng},
            'time': {'hour': hour, 'day_of_week': day_of_week},
            'confidence': min(1.0, max(0.0, prediction / 50))  # Простая оценка уверенности
        }
    
    def predict_demand_grid(self, bounds: Dict, hour: int, day_of_week: int, 
                           grid_size: float = 0.002) -> pd.DataFrame:
        """
        Предсказание спроса для сетки точек
        
        Args:
            bounds: границы области {'min_lat', 'max_lat', 'min_lng', 'max_lng'}
            hour: час дня
            day_of_week: день недели
            grid_size: размер ячейки сетки
            
        Returns:
            DataFrame с предсказаниями для каждой ячейки
        """
        if not self.models:
            raise ValueError("Модели не обучены. Сначала вызовите train_models()")
        
        # Создаем сетку
        lat_range = np.arange(bounds['min_lat'], bounds['max_lat'], grid_size)
        lng_range = np.arange(bounds['min_lng'], bounds['max_lng'], grid_size)
        
        predictions = []
        
        for lat in lat_range:
            for lng in lng_range:
                pred = self.predict_demand(lat, lng, hour, day_of_week)
                predictions.append({
                    'lat': lat,
                    'lng': lng,
                    'predicted_demand': pred['predicted_demand'],
                    'confidence': pred['confidence']
                })
        
        return pd.DataFrame(predictions)
    
    def get_feature_importance(self, model_name: str = None) -> pd.DataFrame:
        """
        Получение важности признаков
        
        Args:
            model_name: название модели
            
        Returns:
            DataFrame с важностью признаков
        """
        if not self.feature_importance:
            raise ValueError("Модели не обучены или важность признаков недоступна")
        
        if model_name is None:
            model_name = list(self.feature_importance.keys())[0]
        
        importance_data = self.feature_importance[model_name]
        
        df = pd.DataFrame([
            {'feature': feature, 'importance': importance}
            for feature, importance in importance_data.items()
        ]).sort_values('importance', ascending=False)
        
        return df
    
    def save_model(self, filepath: str, model_name: str = None):
        """
        Сохранение обученной модели
        
        Args:
            filepath: путь для сохранения
            model_name: название модели для сохранения
        """
        if not self.models:
            raise ValueError("Модели не обучены")
        
        if model_name is None:
            model_name = list(self.models.keys())[0]
        
        model_data = {
            'model': self.models[model_name],
            'scaler': self.scalers.get('main'),
            'feature_importance': self.feature_importance.get(model_name, {}),
            'grid_size': self.grid_size,
            'training_data_stats': {
                'lat_mean': self.training_data['center_lat'].mean(),
                'lat_std': self.training_data['center_lat'].std(),
                'lng_mean': self.training_data['center_lng'].mean(),
                'lng_std': self.training_data['center_lng'].std()
            }
        }
        
        joblib.dump(model_data, filepath)
        print(f"Модель {model_name} сохранена в {filepath}")
    
    def load_model(self, filepath: str):
        """
        Загрузка обученной модели
        
        Args:
            filepath: путь к файлу модели
        """
        model_data = joblib.load(filepath)
        
        self.models = {'loaded': model_data['model']}
        self.scalers = {'main': model_data['scaler']}
        self.feature_importance = {'loaded': model_data['feature_importance']}
        self.grid_size = model_data['grid_size']
        
        print(f"Модель загружена из {filepath}")

if __name__ == "__main__":
    # Демонстрация работы предсказателя спроса
    from data_processor import GeoTrackProcessor
    
    processor = GeoTrackProcessor()
    processor.create_sample_data()
    processor.anonymize_data()
    
    predictor = DemandPredictor(processor)
    
    print("Подготовка данных для обучения...")
    training_data = predictor.prepare_training_data()
    print(f"Подготовлено {len(training_data)} записей для обучения")
    
    print("\nОбучение моделей...")
    results = predictor.train_models()
    
    print("\nТестирование предсказаний...")
    # Предсказание для конкретной точки
    pred = predictor.predict_demand(51.095, 71.427, 14, 1)  # 14:00, вторник
    print(f"Предсказанный спрос: {pred['predicted_demand']:.1f} поездок")
    print(f"Уверенность: {pred['confidence']:.2f}")
    
    # Важность признаков
    print("\nВажность признаков:")
    importance = predictor.get_feature_importance()
    print(importance.head(10))
    
    # Сохранение модели
    predictor.save_model('demand_model.pkl')
    print("\nМодель сохранена!")
