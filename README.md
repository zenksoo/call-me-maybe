# Instructions

## Requirements
- `python` 3.12+
- `uv` Package Manager

## Install UV
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

## Install Project Dependencies
```bash
make install
```
This will:
- Create a virtual environment `.venv`
- Install all dependencies into it
- Install `flake8` and `mypy` (included in the dependencies list)

## Run the Project
```bash
make run
```

## Debug Mode
If you want to run the project in debug mode:
```bash
make debug
```

## Cleanup
When you run the project or `mypy`, some cache files will be created. To remove them:
```bash
make clean
```
