FROM public.ecr.aws/lambda/python:3.9

COPY ./requirements.txt ./requirements.txt

#COPY ./requirements.txt ${LAMBDA_TASK_ROOT}

RUN yum -y install libcurl-devel libssl-dev gcc openssl openssl-devel

#RUN yum load-transaction /tmp/yum_save_tx.2024-12-12.09-52.SPZ36M.yumtx

RUN pip install --upgrade pip && pip install --no-deps --default-timeout=100 -r requirements.txt

COPY . .

#COPY . ${LAMBDA_TASK_ROOT}

CMD [ "lambda_function.lambda_handler" ]

# FROM public.ecr.aws/lambda/python:3.13

# # Copy function code
# COPY lambda_function.py ${LAMBDA_TASK_ROOT}

# CMD [ "lambda_function.lambda_handler" ]