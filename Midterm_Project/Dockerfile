FROM python:3.8.12-slim
 
RUN pip install pipenv
 
WORKDIR /app
COPY["requirements.txt", "./"]
 
RUN pip install -r requirements.txt
 
COPY["predict.py", "model.bin", "./"]
 
EXPOSE 9696
 
ENTRYPOINT["waitress", "--bind=0.0.0.0:9696", "predict:app"]