FROM python:3.8
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install --root-user-action=ignore -r requirements.txt \
&& apt-get update \
&& apt-get install -y --no-install-recommends v4l-utils libgl1 \ 
&& apt-get clean \
&& rm -rf /var/lib/apt/lists/*
COPY ./app/ .
CMD ["python3", "app.py"]