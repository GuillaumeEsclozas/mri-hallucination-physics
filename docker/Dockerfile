FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV OMP_NUM_THREADS=8

ENTRYPOINT ["python"]
CMD ["scripts/train_ddp.py", "--help"]
