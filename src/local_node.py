import flwr as fl
import json
import numpy as np
import os
import pandas as pd
import sys
import time

from src.modules.utils import read_data, sample_data, preprocess_data_causality, DATASET_NAME
from src.modules.fl_client import FLClient
from src.modules.mqtt_client import MQTTClient


class LocalNode:
    """
    Local Node
    """
    def __init__(self, name, server_ip, server_port, mqtt_broker=None, path="../data"):
        self.name = name
        self.server_ip = server_ip
        self.server_port = server_port
        self.fl_client = None
        if mqtt_broker is not None:
            self.mqtt_client = MQTTClient(node=self, broker=mqtt_broker)
        else:
            self.mqtt_client = MQTTClient(node=self)
        self.mqtt_client.subscribe(f"{self.mqtt_client.topic_transmission}{self.name}")
        self.temp_file_path = f"../temp/temp_{self.name}.csv"
        self.result = None
        self.refresh_time = 1
        self.ma7_ate = []
        df, _ = read_data(f"{path}/{DATASET_NAME}_dataset/")
        df = sample_data(df, 100)
        df.to_csv(self.temp_file_path, header=True, index=False)
        self.nominal_attributes = {c: (True if t not in ['float64', 'int64'] else False) for (t, c) in zip(df.dtypes, df.columns)}
        df = preprocess_data_causality(df)
        self.nominal_attributes = [v for k, v in self.nominal_attributes.items() if k in df.columns]
        self.X = df.iloc[:, :-1].copy().to_numpy()
        self.y = df.iloc[:, -1].copy().to_numpy()
        self.columns = df.columns

    def start(self, doLoop=True):
        while self.mqtt_client.status != "running" and doLoop:
            time.sleep(1)
        time.sleep(2)
        self.run_loop(doLoop=doLoop)

    def run_loop(self, doLoop=True):
        i = 0
        while self.mqtt_client.status != "stop":
            if self.mqtt_client.status == "idle":
                time.sleep(1)
                continue
            print(i)
            self.run_fl()
            i += 1
            print(f"NODE {self.name}: {i}")
            if not doLoop:
                break
        sys.exit()

    def run_fl(self):
        self.fl_client = FLClient(self)
        df = pd.read_csv(self.temp_file_path, header=0, na_values='?')
        self.nominal_attributes = {c: (True if t not in ['float64', 'int64'] else False) for (t, c) in zip(df.dtypes, df.columns)}
        df = preprocess_data_causality(df)
        self.nominal_attributes = [v for k, v in self.nominal_attributes.items() if k in df.columns]
        self.X = df.iloc[:, :-1].copy().to_numpy()
        self.y = df.iloc[:, -1].copy().to_numpy()
        self.columns = df.columns
        while True:
            try:
                fl.client.start_numpy_client(
                    server_address=f'{self.server_ip}:{self.server_port}',
                    client=self.fl_client
                )
                self.send_result(self.result)
                break
            except Exception as e:
                print(e)
                time.sleep(1)
        time.sleep(self.refresh_time)

    def status(self):
        self.mqtt_client.publish(
            msg=json.dumps({"node": self.name, "status": "ready"}),
            topic_send=f"{self.mqtt_client.topic_status}{self.name}/"
        )

    def send_result(self, result):
        self.mqtt_client.publish(
            msg=json.dumps(result),
            topic_send=f"{self.mqtt_client.topic_result}{self.name}/"
        )


if __name__ == "__main__":
    name = os.environ.get("NODE_NAME")
    server_ip = os.environ.get("SERVER_IP")
    server_port = os.environ.get("SERVER_PORT")
    LocalNode(
        name=name,
        server_ip=server_ip,
        server_port=server_port,
    ).start(doLoop=True)



