FROM python:3
WORKDIR /JCC
RUN mkdir /capstone
VOLUME ["/data"]
COPY requirements.txt ./
RUN apt-get update
RUN pip install -r requirements.txt
COPY ./ ./
EXPOSE 80
CMD ["python","./app.py"]
