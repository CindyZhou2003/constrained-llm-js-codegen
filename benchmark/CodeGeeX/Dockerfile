FROM pytorch/pytorch:1.11.0-cuda11.3-cudnn8-runtime

RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak \
    && apt-get update && apt-get install -y curl npm git nano \
    && pip install fire zmq transformers tokenizers \
    && mkdir /workspace/download

RUN curl -o /workspace/download/go.tar.gz -SL https://go.dev/dl/go1.18.4.linux-amd64.tar.gz \
    && tar -zxf /workspace/download/go.tar.gz -C /usr/local && rm /workspace/download/go.tar.gz
ENV PATH=/bin:/usr/local/go/bin:$PATH

RUN curl -o /workspace/download/node.tar.gz -SL https://nodejs.org/download/release/v16.14.0/node-v16.14.0-linux-x64.tar.gz \
    && mkdir -p /usr/local/lib/nodejs && tar -zxf /workspace/download/node.tar.gz -C /usr/local/lib/nodejs && mv /usr/local/lib/nodejs/node-v16.14.0-linux-x64 /usr/local/lib/nodejs/node \
    && rm /workspace/download/node.tar.gz && npm install -g js-md5@0.7.3
ENV PATH=/usr/local/lib/nodejs/node/bin:$PATH
ENV NODE_PATH=/usr/local/lib/node_modules

RUN apt-get install -y build-essential && apt-get install -y g++ \
    && curl -o /workspace/download/boost_1_71_0.tar.gz -SL https://archives.boost.io/release/1.71.0/source/boost_1_71_0.tar.gz \
    && tar -zxf /workspace/download/boost_1_71_0.tar.gz && rm /workspace/download/boost_1_71_0.tar.gz && cd boost_1_71_0 \
    && ./bootstrap.sh --prefix=/usr/ && ./b2 && ./b2 install \
    && cd .. && rm -r boost_1_71_0
RUN curl -o /workspace/download/openssl.tar.gz -SL https://www.openssl.org/source/old/3.0/openssl-3.0.0.tar.gz \
    && tar -zxf /workspace/download/openssl.tar.gz && cd openssl-3.0.0 && ./Configure && make && make install \
    && rm /workspace/download/openssl.tar.gz && rm -r /workspace/openssl-3.0.0
ENV PATH=/usr/bin/openssl:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
    openjdk-17-jdk \
    && rm -rf /var/lib/apt/lists/*

COPY . /workspace/CodeGeeX

RUN pip install -e /workspace/CodeGeeX \
    && tar -zxf /workspace/CodeGeeX/codegeex/benchmark/humaneval-x/go/evaluation/vendor.tar.gz -C /workspace/CodeGeeX/codegeex/benchmark/humaneval-x/go/evaluation

WORKDIR /workspace/CodeGeeX
