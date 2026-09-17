PY=python
VENV=.venv
CONFIG=configs/config.yaml

init:
	python3 -m venv $(VENV) && . $(VENV)/bin/activate && pip install -U pip && pip install -r requirements.txt

get-data:
	$(PY) src/get_data.py --config $(CONFIG)

train:
	$(PY) src/train.py --config $(CONFIG)

evaluate:
	$(PY) src/evaluate.py --config $(CONFIG)

test:
	pytest -q

run:
	uvicorn app:app --host 0.0.0.0 --port 8000 --reload

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db --port 5000

build:
	docker build -t credit-fraud-mlops:latest .

.PHONY: init get-data train evaluate test run mlflow-ui build
