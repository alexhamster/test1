
import os

BASEDIR_PATH = os.path.dirname(__file__)
SRC_DIR = os.path.join(BASEDIR_PATH, "./tokenized")
DST_DIR = os.path.join(BASEDIR_PATH, "./decoded")

import os
import json
from pathlib import Path
from transformers import AutoTokenizer

# 🛈 Поменяй на свои пути

tokenizer = AutoTokenizer.from_pretrained(
    "google/gemma-3-4b-it",
    use_fast=False
)

def decode_gemma_file(path: Path):
    """Читает файл с JSON-строками и возвращает список декодированных текстов."""
    texts = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # пропускаем пустые строки
            obj = json.loads(line)
            input_ids = obj["input_ids"]
            text = tokenizer.decode(input_ids, skip_special_tokens=True)
            texts.append(text)
    return texts

def process_directory(src_dir: Path, dst_dir: Path):
    """
    Рекурсивно обходит src_dir, декодирует каждый файл
    и зеркально сохраняет в dst_dir.
    """
    for root, dirs, files in os.walk(src_dir):
        root_path = Path(root)
        for fname in files:
            src_file = root_path / fname

            # относительный путь от корня входной директории
            rel_path = src_file.relative_to(src_dir)
            # путь к выходному файлу (зеркальная структура каталогов)
            dst_file = (dst_dir / rel_path).with_suffix(".md")

            # создаём поддиректорию, если её ещё нет
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            # декодируем содержимое файла
            try:
                texts = decode_gemma_file(src_file)
            except Exception as e:
                print(f"Ошибка при обработке {src_file}: {e}")
                continue

            # сохраняем декодированный текст
            # здесь: один пример = один блок, разделённый пустой строкой
            with dst_file.open("w", encoding="utf-8") as out_f:
                for i, t in enumerate(texts):
                    if i > 0:
                        out_f.write("\n\n")  # пустая строка между примерами
                    out_f.write(t)

            print(f"Готово: {src_file} -> {dst_file}")

if __name__ == "__main__":
    process_directory(SRC_DIR, DST_DIR)
