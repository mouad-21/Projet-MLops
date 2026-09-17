FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8000

# Le serveur MLflow tourne sur la machine hote (`make mlflow-server`).
# On utilise l'IP privee de la passerelle Docker Desktop (et non le nom
# 'host.docker.internal') car le controle anti-DNS-rebinding de MLflow
# n'autorise par defaut que les IP privees RFC1918, pas les noms d'hote
# personnalises. Si cette IP differe sur ta machine : `docker run ... -e
# MLFLOW_TRACKING_URI=http://<ton-ip>:5000` pour la surcharger.
ENV MLFLOW_TRACKING_URI=http://192.168.65.254:5000

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
