import os
import glob
from pick import pick

def select(latest, models_dir = "models"):
    os.makedirs(models_dir, exist_ok=True)

    # Получаем список .zip файлов (SB3 сохраняет модели в .zip)
    model_files = glob.glob(os.path.join(models_dir, "*.zip"))

    if not model_files:
        print("В папке 'models' нет сохранённых моделей!")
        exit()

    if latest: 
        selected_model = model_files.pop()
    else:
        # Выбор модели через интерактивное меню
        title = "Выберите модель для загрузки:"
        selected_model = pick(model_files, title, indicator="→")[0]

    print(f"Выбрана {selected_model}")

    return selected_model