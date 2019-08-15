FROM python:3.6-slim

RUN pip install flask gunicorn numpy scikit-learn joblib flask_wtf pandas

WORKDIR /root

COPY . /root