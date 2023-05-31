FROM python:3.8
COPY . .
WORKDIR .
RUN apt-get update && apt-get install -y libglib2.0-0 libgl1-mesa-glx && rm -rf /var/lib/apt/lists/*
RUN /usr/local/bin/python -m pip install --upgrade pip
RUN pip install --no-cache-dir cmake
RUN pip install --no-cache-dir dlib==19.21.1 -vvv
RUN pip install --no-cache-dir -r requirements.txt
RUN pip cache purge
EXPOSE 5000
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
