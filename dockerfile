FROM teploeodealko/parking11
EXPOSE 5001

# set a directory for the app
RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

# copy all the files to the container
COPY . /usr/src/app

# install dependencies
RUN pip install --no-cache-dir -r requirements.txt
RUN apt-get update && apt-get install -y libgl1-mesa-glx
# tell the port number the container should expose

# run the command
CMD [ "python", "./index.py" ]
