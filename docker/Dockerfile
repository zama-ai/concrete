FROM ghcr.io/zama-ai/zamalang-compiler

RUN apt-get install --no-install-recommends -y python3.8 python3.8-venv python-is-python3 && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry && \
    echo "python3 -m venv /hdk/.docker_venv" >> /root/.bashrc && \
    echo "source /hdk/.docker_venv/bin/activate" >> /root/.bashrc && \
    echo "cd /hdk/ && make setup_env" >> /root/.bashrc

WORKDIR /hdk

ENTRYPOINT ["/bin/bash", "-l"]
