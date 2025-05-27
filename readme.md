# Для Windows:
```bash
python -m venv myenv
myenv\Scripts\activate  # Активация
```

# Для Linux/Mac:
```bash
python3 -m venv myenv
source myenv/bin/activate  # Активация
```

# Для установки зависимостей
```bash
pip install -r requirements.txt
```

# Для записи зависимостей
```bash
pip freeze > requirements.txt
```

# Для просмотра логов
```bash
tensorboard --logdir=./logs/
```