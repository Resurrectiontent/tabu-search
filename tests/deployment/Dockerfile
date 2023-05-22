FROM python:3.10
LABEL maintainer="Resurrectiontent"

WORKDIR /tabu-trialing/
COPY tabu-search/ ./tabu-search
COPY sampo/ ./sampo

ENV PYTHONPATH="/tabu-trialing/tabu-search"
ENV PYTHONPATH="${PYTHONPATH}:/tabu-trialing/tabu-search/tabusearch"
ENV PYTHONPATH="${PYTHONPATH}:/tabu-trialing/tabu-search/tests"
ENV PYTHONPATH="${PYTHONPATH}:/tabu-trialing/sampo"

ENV PATH="${PATH}:${PYTHONPATH}"

#    && apk add libjpeg-dev zlib1g-dev \
RUN apt-get update \
    && pip install --upgrade pip setuptools wheel \
    && pip install -r /tabu-trialing/tabu-search/requirements.txt --use-pep517 \
    && pip install -r /tabu-trialing/sampo/requirements.txt --use-pep517 \
    && rm -rf /root/.cache/pip

#ENTRYPOINT ["python", "-c", "from tabusearch import TabuSearch; print('success')"]
ENTRYPOINT ["pytest", "tabu-search/tests/scheduling/checking_scheduling_tests.py"]
