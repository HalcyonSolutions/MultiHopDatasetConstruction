# Docker file for running some of the NLP benchmarks
FROM  pytorch/pytorch:2.4.0-cuda12.1-cudnn9-runtime

RUN apt update && apt install -y --no-install-recommends \
        nvim \
        pipx

# Fix above:
RUN pipx install poetry==1.8.3

COPY requirements.txt .

ENTRYPOINT ["/bin/bash"]
