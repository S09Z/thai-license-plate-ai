# Project Structure Commands

mkdir -p thai-license-plate-ai/{app,detector,ocr,rag,models,datasets/{raw,processed,augmented},web,tests,notebooks,docs/{adr,benchmark,experiments,diagrams}}

cd thai-license-plate-ai

poetry init

poetry add fastapi uvicorn opencv-python ultralytics paddleocr torch torchvision albumentations chromadb sentence-transformers numpy pillow pydantic python-multipart gradio jupyter ipykernel

poetry add --group dev pytest ruff black mypy pre-commit

touch README.md CLAUDE.md .gitignore

tree .
