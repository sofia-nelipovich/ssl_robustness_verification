import os

# корень проекта — текущая директория
ROOT = os.path.dirname(os.path.abspath(__file__))

IGNORE_DIRS = {
    ".git", ".idea", ".vscode", "__pycache__", ".pytest_cache",
    "venv", ".venv", "env", ".mypy_cache", ".DS_Store",
    "wandb", "lightning_logs", "logs", "checkpoints", "runs"
}
IGNORE_EXT = {
    ".pyc", ".pyo", ".pyd", ".so", ".dll",
    ".pt", ".pth", ".ckpt", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".tiff", ".bmp",
    ".zip", ".tar", ".gz", ".bz2"
}

def should_ignore_file(name):
    ext = os.path.splitext(name)[1].lower()
    return ext in IGNORE_EXT

def print_tree(root, file):
    for current_root, dirs, files in os.walk(root):
        # относительный путь от корня
        rel = os.path.relpath(current_root, root)
        if rel == ".":
            level = 0
            rel = ""
        else:
            level = rel.count(os.sep)

        # фильтруем директории
        dirs[:] = [d for d in dirs if d not in IGNORE_DIRS]

        indent = "    " * level
        dir_name = os.path.basename(current_root) if rel else os.path.basename(root)
        file.write(f"{indent}{dir_name}/\n")

        # файлы
        for name in sorted(files):
            if should_ignore_file(name):
                continue
            file.write(f"{indent}    {name}\n")

if __name__ == "__main__":
    out_path = os.path.join(ROOT, "project_tree.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        print_tree(ROOT, f)
    print(f"Структура проекта сохранена в {out_path}")
