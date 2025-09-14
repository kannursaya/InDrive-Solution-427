# 📦 Инструкции по установке

## Системные требования

- Python 3.8 или выше
- 4GB RAM (рекомендуется 8GB)
- 1GB свободного места на диске
- Современный браузер (Chrome, Firefox, Safari)

## Установка Python

### macOS
```bash
# Через Homebrew (рекомендуется)
brew install python3

# Или скачайте с python.org
# https://www.python.org/downloads/
```

### Windows
1. Скачайте Python с https://www.python.org/downloads/
2. Установите с опцией "Add Python to PATH"

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install python3 python3-pip
```

## Установка зависимостей

### Способ 1: Через pip
```bash
pip install -r requirements.txt
```

### Способ 2: Через pip3
```bash
pip3 install -r requirements.txt
```

### Способ 3: Создание виртуального окружения (рекомендуется)
```bash
# Создание виртуального окружения
python3 -m venv geotrack_env

# Активация (macOS/Linux)
source geotrack_env/bin/activate

# Активация (Windows)
geotrack_env\Scripts\activate

# Установка зависимостей
pip install -r requirements.txt
```

## Проверка установки

```bash
python3 -c "import pandas, numpy, matplotlib, plotly, folium, sklearn, dash; print('✅ Все зависимости установлены')"
```

## Возможные проблемы и решения

### Ошибка "command not found: python"
```bash
# Используйте python3 вместо python
python3 main_demo.py
```

### Ошибка "ModuleNotFoundError"
```bash
# Установите зависимости
pip3 install -r requirements.txt
```

### Ошибка с правами доступа
```bash
# Используйте --user
pip3 install --user -r requirements.txt
```

### Проблемы с matplotlib на macOS
```bash
# Установите через conda
conda install matplotlib
```

## Альтернативная установка через conda

```bash
# Создание окружения
conda create -n geotrack python=3.9

# Активация
conda activate geotrack

# Установка пакетов
conda install pandas numpy matplotlib seaborn plotly scikit-learn
pip install folium dash
```

## Проверка работоспособности

После установки запустите:

```bash
python3 run_demo.py
```

Если все работает, вы увидите меню с опциями.

## Поддержка

При возникновении проблем:
1. Проверьте версию Python: `python3 --version`
2. Проверьте установленные пакеты: `pip3 list`
3. Создайте issue в репозитории

---
**Удачной установки!** 🚀
