FROM python:3-slim

WORKDIR /app
COPY requirements-api.txt requirements.txt
RUN pip install -r requirements.txt

RUN mkdir models
COPY models/hgb_regressor_0.5_500.pkl models/hgb_regressor_0.5_500.pkl
COPY models/one_hot_encoder.pkl models/one_hot_encoder.pkl
COPY src/service.py service.py

CMD ["uvicorn", "--host", "0.0.0.0", "service:app"]
