# set base image (host OS)
FROM zaandahl/megadetector:latest

# Install the iptcinfo3 library
RUN pip install iptcinfo3

# set the working directory in the container
WORKDIR /code

# copy code
COPY src/ .

# run metadata_writer on start
CMD [ "python", "./mewc_exif.py" ]