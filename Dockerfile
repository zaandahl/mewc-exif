# set base image (host OS)
FROM zaandahl/megadetector:latest

# set the working directory in the container
WORKDIR /code

# copy code
COPY src/ .

# run metadata_writer on start
CMD [ "python", "./mewc_exif.py" ]