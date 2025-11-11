FROM python:3.11-slim

WORKDIR /app

ENV PIP_NO_CACHE_DIR=1

COPY requirements.txt /tmp/requirements.txt
RUN python -m venv /opt/venv \
    && . /opt/venv/bin/activate \
    && pip install --upgrade pip \
    && pip install -r /tmp/requirements.txt

ENV PATH="/opt/venv/bin:$PATH"

COPY . /app

CMD ["python", "end_to_end_demo.py"]
