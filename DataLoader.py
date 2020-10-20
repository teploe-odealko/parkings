import pandas as pd
import json
from datetime import datetime, timedelta
import time
import requests
import pickle
import glob
import numpy as np
from threading import Lock


from Model.dbconnection import DBconnection

current_milli_time = lambda: int(round(time.time() * 1000))


class SingletonMeta(type):
    _instances = {}

    _lock: Lock = Lock()

    def __call__(cls, *args, **kwargs):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]


class DataLoader(metaclass=SingletonMeta):
    def __init__(self):
        print("start dataloader init")
        self.dbconnection = DBconnection()
        self.cameras_table = self.dbconnection.get_cameras()



    def get_cameras_table(self):
        return self.cameras_table

    def get_rects_for_camera(self, camera_id):
        return self.dbconnection.get_rects_for_camera(camera_id)

    def add_new_cameras_to_db(self, new_cameras):
        self.dbconnection.add_new_cameras(new_cameras)

