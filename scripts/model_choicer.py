import os
import glob
import inquirer


def select(models_dir, latest=False):
    os.makedirs(models_dir, exist_ok=True)

    # Получаем список .zip файлов (SB3 сохраняет модели в .zip)
    model_files = glob.glob(os.path.join(models_dir, "*.zip"))

    if not model_files:
        print("В папке 'models' нет сохранённых моделей!")
        exit()

    model_files_sorted = sorted(
        model_files,
        key=lambda x: os.path.getmtime(x),
        reverse=True  # Новые файлы в начале
    )
    selected_model = model_files_sorted.pop(-1)

    if not latest: 
        # Выбор модели через интерактивное меню
        choice = inquirer.prompt([
            inquirer.List("menu", message="Выберите модель для загрузки", choices=model_files_sorted, default=selected_model)
        ])
        selected_model = choice["menu"]

    print(f"Выбрана {selected_model}")

    return selected_model