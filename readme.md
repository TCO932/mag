# Для Windows:
```bash
python -m venv env
env\Scripts\activate
```

# Для Linux/Mac:
```bash
python3 -m venv env
source env/bin/activate
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