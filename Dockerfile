FROM python:3.9.5

WORKDIR /do_wyslania

COPY requirements.txt .

COPY RTA_model_pick.pkl .

COPY perceptron.py .

COPY model.py .

COPY apka.py .

RUN pip install -r requirements.txt

CMD ["python", "./apka.py"]
