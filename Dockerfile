# Required: use the fixed parent built from the matching source lock.
ARG MEWC_DETECT_BASE
FROM ${MEWC_DETECT_BASE}
WORKDIR /code
COPY requirements-runtime.txt .
RUN python -m pip install --no-cache-dir --no-deps --require-hashes -r requirements-runtime.txt
COPY src/ .
CMD ["python", "./mewc_exif.py"]
