FROM python:3.9.20

COPY ./requirements.txt ./requirements.txt

RUN pip install --upgrade pip && pip install --no-deps -r requirements.txt

COPY . .

EXPOSE 8501

WORKDIR /src

CMD ["python","predict.py"]