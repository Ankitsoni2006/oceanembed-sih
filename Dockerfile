FROM python:3.11-slim

WORKDIR /app

COPY backend/requirements.txt /app/backend/requirements.txt

# Install CPU-only PyTorch first, using the exact torch requirement line from
# backend/requirements.txt. The subsequent requirements install then sees torch
# as already satisfied and does not pull the default CUDA-enabled wheel.
RUN grep -iE '^torch([<>=!~ ;\[]|$)' /app/backend/requirements.txt > /tmp/torch-req.txt \
    && pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu -r /tmp/torch-req.txt \
    && pip install --no-cache-dir -r /app/backend/requirements.txt \
    && python -c "import torch; assert torch.version.cuda is None, 'CUDA torch installed'; print('torch', torch.__version__, 'CPU-only')" \
    && rm /tmp/torch-req.txt

COPY . .

ENV OMP_NUM_THREADS=1
ENV MKL_NUM_THREADS=1
ENV PYTHONUNBUFFERED=1

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
