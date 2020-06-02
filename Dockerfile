FROM python:3.8.3-slim

RUN apt-get update && apt-get install -y \
    python-opencv

RUN mkdir /api
WORKDIR /api
ADD src/requirements.txt /api/
RUN pip install -r requirements.txt
ADD src /api/

CMD ["python", "api.py"]