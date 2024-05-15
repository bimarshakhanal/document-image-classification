FROM python:3.10-slim

WORKDIR /doc-image-classification

COPY ./requirements.txt /app/requirements.txt

# copy requirements to container
COPY ./requirements.txt /code/requirements.txt

# install requirements in container
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

# copy source code to container
COPY ./app /doc-image-classification/app/

# copy trained model to docker image, can also train model inside docker
COPY ./resnet_all.pth /doc-image-classification/

RUN mkdir log

EXPOSE 8000
# command to start fastapi application 
CMD ["streamlit", "run", "app/app.py", "--server.port=8000", "--server.address=0.0.0.0"]