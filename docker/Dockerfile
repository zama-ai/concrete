FROM ghcr.io/zama-ai/zamalang-compiler

RUN apt-get install --no-install-recommends -y python3.8 python3.8-venv python-is-python3 git && \
    pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir poetry && \
    echo "source /hdk/.docker_venv/bin/activate" >> /root/.bashrc && \
    echo "if [[ \"\$?\" != \"0\" ]]; then" >> /root/.bashrc && \
    echo "    python3 -m venv /hdk/.docker_venv" >> /root/.bashrc && \
    echo "    source /hdk/.docker_venv/bin/activate" >> /root/.bashrc && \
    echo "    cd /hdk/ && make setup_env" >> /root/.bashrc && \
    echo "fi" >> /root/.bashrc && \
    echo "export LD_PRELOAD=/concrete/target/release/libconcrete_ffi.so" >> /root/.bashrc

WORKDIR /hdk

ENTRYPOINT ["/bin/bash", "-l"]
