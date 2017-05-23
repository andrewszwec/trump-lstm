# RUN ME
# floyd run --mode serve
# Send POST
# curl -o taipei_output.jpg -F "file=@./images/taipei101.jpg" https://www.floydhub.com:8000/VtuBeYAksspRtMLmWcSd3K

"""
Flask Serving

This file is a sample flask app that can be used to test your model with an API.

"""
import os
from flask import Flask, send_file, request
from werkzeug.exceptions import BadRequest
from werkzeug.utils import secure_filename

from evaluate import trainModel

app = Flask(__name__)


@app.route('/<path:path>', methods=["POST"])
def runModel(path):

        trainModel()

        text_file = open("/output/Output.txt", "w")
        text_file.write("hello world from app.py. File path variable = "+path)
        text_file.close()

        return send_file(''/output/console_output.txt', mimetype='text/plain')


if __name__ == '__main__':
    app.run(host='0.0.0.0')
