FROM agrigorev/model-2024-hairstyle:v3

RUN pip install pillow
RUN pip install https://github.com/alexeygrigorev/tflite-aws-lambda/raw/main/tflite/tflite_runtime-2.14.0-cp310-cp310-linux_x86_64.whl?raw=true
RUN pip install numpy==1.23.5

COPY HW9.py .

CMD [ "HW9.handler" ]

#docker build . -t model_2024_hairstyle:v3 -f Dockerfile
#docker run -it --rm -p 8080:8080 agrigorev/model-2024-hairstyle:v3
