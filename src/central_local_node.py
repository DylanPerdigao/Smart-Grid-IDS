import json
import flwr as fl
import numpy as np
import os
import sys
import time
import threading as th

from src.modules.fl_client import FLClient
from src.modules.fl_server import FLServer
from src.modules.mqtt_client import MQTTClient
from src.modules.fedavg_custom import FedAvgCustom


class CentralLocalNode:
    """
    Central Local Node
    """

    def __init__(self, name, ip, port, server_ip, server_port, mqtt_broker=None):
        self.name = name
        self.ip = ip
        self.port = port
        self.server_ip = server_ip
        self.server_port = server_port
        self.received_data = {}
        if mqtt_broker is not None:
            self.mqtt_client = MQTTClient(node=self, broker=mqtt_broker)
        else:
            self.mqtt_client = MQTTClient(node=self)
        self.X = np.array([])
        self.y = np.array([])
        self.columns = []
        self.nominal_attributes = []
        self.result = None
        self.refresh_time = 1
        self.ma7_ate = []
        self.fl_client = None
        self.fl_server = FLServer(self)

    def start(self, doLoop=True):
        """
        Start local node when recieves ``running`` status from MQTT broker.
        It will start a thread for the client and a thread for the server.
        :param doLoop: if True, the node will run forever
        :return:
        """
        while self.mqtt_client.status != "running" and doLoop:
            time.sleep(1)
        time.sleep(1)
        th_server = th.Thread(target=self.run_loop_server, args=(doLoop,))
        th_server.start()
        time.sleep(self.refresh_time)
        th_client = th.Thread(target=self.run_loop_client, args=(doLoop,))
        th_client.start()

    def run_loop_client(self, doLoop=True):
        i = 0
        while self.mqtt_client.status != "stop":
            if self.mqtt_client.status == "idle":
                time.sleep(1)
                continue
            self.run_fl_client()
            i += 1
            if not doLoop:
                break
        sys.exit()

    def run_loop_server(self, doLoop=True):
        i = 0
        while self.mqtt_client.status != "stop":
            fl.server.start_server(
                server_address=f'{self.ip}:{self.port}',
                strategy=FedAvgCustom(
                    evaluate_fn=self.fl_server.get_evaluate_fn(),
                )
            )
            i += 1
            if not doLoop or self.mqtt_client.status != "running":
                break
        sys.exit()

    def run_fl_client(self):
        while len(self.received_data) < 2:
            time.sleep(1)
        self.X = np.concatenate(([self.received_data[key]["x"] for key in self.received_data.keys()]))
        self.y = np.concatenate(([self.received_data[key]["y"] for key in self.received_data.keys()]))
        self.fl_client = FLClient(self)
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
    ip = os.environ.get("NODE_IP")
    port = os.environ.get("NODE_PORT")
    server_ip = os.environ.get("SERVER_IP")
    server_port = os.environ.get("SERVER_PORT")
    CentralLocalNode(
        name=name,
        ip=ip,
        port=port,
        server_ip=server_ip,
        server_port=server_port,
    ).start(doLoop=True)
