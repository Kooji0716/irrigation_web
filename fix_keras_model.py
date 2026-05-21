import json
import zipfile
from pathlib import Path
import shutil


BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "models"

input_model = MODEL_DIR / "best_irrigation_tf_model.keras"
output_model = MODEL_DIR / "best_irrigation_tf_model_fixed.keras"


UNSUPPORTED_KEYS = [
    "renorm",
    "renorm_clipping",
    "renorm_momentum",
    "quantization_config",
]


def remove_unsupported_keys(obj):
    """
    遞迴移除目前 Keras 版本不支援的舊版/新版模型參數。
    """
    if isinstance(obj, dict):
        for key in UNSUPPORTED_KEYS:
            obj.pop(key, None)

        for value in obj.values():
            remove_unsupported_keys(value)

    elif isinstance(obj, list):
        for item in obj:
            remove_unsupported_keys(item)


def fix_keras_file():
    if not input_model.exists():
        raise FileNotFoundError(f"找不到模型檔案：{input_model}")

    temp_dir = MODEL_DIR / "_keras_fix_temp"

    if temp_dir.exists():
        shutil.rmtree(temp_dir)

    temp_dir.mkdir()

    # .keras 是 zip 格式
    with zipfile.ZipFile(input_model, "r") as zip_ref:
        zip_ref.extractall(temp_dir)

    config_path = temp_dir / "config.json"

    if not config_path.exists():
        raise FileNotFoundError("模型檔內找不到 config.json")

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    remove_unsupported_keys(config)

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False)

    if output_model.exists():
        output_model.unlink()

    with zipfile.ZipFile(output_model, "w", zipfile.ZIP_DEFLATED) as zip_out:
        for file in temp_dir.rglob("*"):
            if file.is_file():
                zip_out.write(file, file.relative_to(temp_dir))

    shutil.rmtree(temp_dir)

    # 檢查 fixed 檔是否還有不支援參數
    with zipfile.ZipFile(output_model, "r") as z:
        fixed_config_text = z.read("config.json").decode("utf-8")

    still_exists = [key for key in UNSUPPORTED_KEYS if key in fixed_config_text]

    if still_exists:
        print("警告：fixed 模型中仍然存在以下參數：")
        print(still_exists)
    else:
        print("Keras 模型修復完成，已移除不支援參數。")

    print(f"輸入模型：{input_model}")
    print(f"輸出模型：{output_model}")


if __name__ == "__main__":
    fix_keras_file()