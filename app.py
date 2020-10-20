import io
import base64
import dash_bootstrap_components as dbc
import flask
from PIL import Image
from flask import request, send_file, jsonify
import dash
from flask import send_file
from sqlalchemy import create_engine, ForeignKey
from flask_sqlalchemy import SQLAlchemy
import cv2
import os
import numpy as np
engine = create_engine('sqlite:///sqllite.db')
from Model.nnAPI import nnAPI
nnapi = nnAPI()

server = flask.Flask(__name__)
server.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///sqllite.db'
db = SQLAlchemy(server)

class Camera(db.Model):
    __tablename__ = "cameras"
    id = db.Column(db.Integer, primary_key=True)
    lon = db.Column(db.Float)
    lat = db.Column(db.Float)
    source = db.Column(db.String(500))

class ParkingBoxes(db.Model):
    __tablename__ = "parking_boxes"
    id = db.Column(db.Integer, primary_key=True)
    parkings = db.Column(db.String(500))
    camera_id = db.Column(db.Integer, ForeignKey('cameras.id'))

if not os.path.exists('sqllite.db'):
    db.create_all()

def distance(x1, y1, x2, y2):
    return ((x2 - x1)**2 + (y2 - y1)**2)**0.5


def find_nearest_camera(lat, lon):

    camera = Camera.query.all()

    x = 0
    y = 0
    nearest_park = np.Inf
    for i in camera:
        dist = distance(lon, lat, i.lon, i.lat)
        if dist < nearest_park:
            nearest_park = dist
            x = i.lat
            y = i.lon

    camera_info = Camera.query.filter_by(lat=x, lon=y).first()
    camera_id = camera_info.id
    camera_source = camera_info.source
    return x, y, camera_id, camera_source


# 'http://50.246.145.122/cgi-bin/faststream.jpg?stream=half&fps=15&rand=COUNTER'
@server.route('/api/coords', methods=['POST'])
def get_camera_coordinates():
    lon = request.get_json().get('lon', '')
    lat = request.get_json().get('lat', '')
    nearest_lat, nearest_lon, camera_id, camera_source = find_nearest_camera(float(lat), float(lon))
    print(camera_source)
    if lon and lat:
        video_capture = cv2.VideoCapture(camera_source)
        success, frame = video_capture.read()
        # frame = cv2.imread('img.jpg')
        predicted_img = nnapi.make_prediction(frame, camera_id)
        predicted_img = cv2.cvtColor(predicted_img, cv2.COLOR_BGR2RGB) # Convert to RGB

        retval, buffer = cv2.imencode('.jpg', predicted_img)
        jpg_as_text = base64.b64encode(buffer)
        response = {'lon': nearest_lon, 'lat': nearest_lat, 'imageBytes': str(jpg_as_text)}
        return response
    return "error"


@server.route('/')
def index():
    return 'Hello Flask app'


app = dash.Dash(
    __name__,
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    # these meta_tags ensure content is scaled correctly on different devices
    # see: https://www.w3schools.com/css/css_rwd_viewport.asp for more
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ], suppress_callback_exceptions=True
)

