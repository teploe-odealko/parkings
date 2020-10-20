import pandas as pd
from threading import Lock
from app import engine
from sqlalchemy.types import Integer, Text, Float
from app import db
from app import Camera

import os

class SingletonMeta(type):
    """
    Это потокобезопасная реализация класса Singleton.
    """

    _instances = {}

    _lock: Lock = Lock()
    """
    У нас теперь есть объект-блокировка для синхронизации потоков во время
    первого доступа к Одиночке.
    """

    def __call__(cls, *args, **kwargs):
        """
        Данная реализация не учитывает возможное изменение передаваемых
        аргументов в `__init__`.
        """
        # Теперь представьте, что программа была только-только запущена.
        # Объекта-одиночки ещё никто не создавал, поэтому несколько потоков
        # вполне могли одновременно пройти через предыдущее условие и достигнуть
        # блокировки. Самый быстрый поток поставит блокировку и двинется внутрь
        # секции, пока другие будут здесь его ожидать.
        with cls._lock:
            # Первый поток достигает этого условия и проходит внутрь, создавая
            # объект-одиночку. Как только этот поток покинет секцию и освободит
            # блокировку, следующий поток может снова установить блокировку и
            # зайти внутрь. Однако теперь экземпляр одиночки уже будет создан и
            # поток не сможет пройти через это условие, а значит новый объект не
            # будет создан.
            if cls not in cls._instances:
                instance = super().__call__(*args, **kwargs)
                cls._instances[cls] = instance
        return cls._instances[cls]
# Here you want to change your database, username & password according to your own values

class DBconnection(metaclass=SingletonMeta):

    def get_cameras(self):
        try:
            cameras_df = pd.read_sql_table(
                'cameras',
                con=engine
            )
            print(cameras_df)
        except ValueError:
            cameras_df = pd.DataFrame(columns=['id', 'lon', 'lat', 'source'])
            cameras_df.to_sql(
                'cameras',
                engine,
                if_exists='replace',
                index=False,
                dtype={
                    "id": Integer,
                    "lon": Float,
                    'lat': Float,
                    'source': Text
                }
            )
        # print(cameras_df)
        return cameras_df
    def get_rects_for_camera(self, camera_id):
        # exeute("""SELECT * FROM rect where camera_id={}""".format(camera_id))
        rects = [[(659.5276, 482.7602), (748.8723, 529.75305)],
                  [(947.66626, 279.93768), (965.4137, 294.1808)],
                  [(745.0082, 248.91858), (757.2035, 257.93304)],
                  [(133.7883, 282.50238), (177.917, 304.15292)],
                  [(472.16672, 322.5269), (515.29517, 345.88824)],
                  [(719.40247, 246.53339), (738.3664, 257.93546)],
                  [(264.6958, 291.24426), (317.86523, 310.84894)],
                  [(770.986, 242.51913), (785.36523, 253.7591)],
                  [(749.2883, 286.28345), (776.18616, 301.76523)],
                  [(390.86472, 484.6215), (483.3481, 535.8152)],
                  [(444.1245, 355.69254), (510.2761, 381.93747)],
                  [(1000.91864, 267.69052), (1026.4231, 279.84262)],
                  [(176.59976, 268.72406), (230.6368, 296.42017)],
                  [(735.3866, 384.4489), (779.8263, 404.699)],
                  [(211.23746, 340.33136), (384.16888, 494.83884)],
                  [(641.16986, 213.52638), (650.6248, 223.6352)],
                  [(1011.862, 288.9586), (1044.3118, 305.2235)],
                  [(689.4159, 384.07135), (715.97925, 407.7938)],
                  [(789.88586, 272.18997), (822.502, 289.53775)],
                  [(790.25214, 272.1839), (823.23236, 290.24084)]]
        return rects
    def add_new_cameras(self, new_cameras_df):
        for i, camera in new_cameras_df.iterrows():
            new_camera = Camera(lat = camera['lat'],
                              lon=camera['lon'],
                              source=camera['source'])
            db.session.add(new_camera)
        db.session.commit()
        # print(new_cameras_df)
        # new_cameras_df.to_sql(
        #     'cameras',
        #     engine,
        #     if_exists='replace',
        #     index=False,
        #     dtype={
        #         "lon": Float,
        #         'lat': Float,
        #         'source': Text
        #     }
        # )
