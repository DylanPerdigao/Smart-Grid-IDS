import json
import flwr as fl
import numpy as np
import os
import sys
import time

from src.modules.fl_client import FLClient
from src.modules.fl_server import FLServer
from src.modules.mqtt_client import MQTTClient
from src.modules.fedavg_custom import FedAvgCustom


class CentralNode:
    """
    Central Local Node
    """
    def __init__(self, name, ip, port, mqtt_broker=None):
        self.name = name
        self.ip = ip
        self.port = port
        if mqtt_broker is not None:
            self.mqtt_client = MQTTClient(node=self, broker=mqtt_broker)
        else:
            self.mqtt_client = MQTTClient(node=self)
        self.refresh_time = 1
        self.round = 0
        self.result = None
        self.received_data = {}
        self.X = np.array([])
        self.y = np.array([])
        self.columns = []
        self.nominal_attributes = []
        self.ma7_ate = []
        self.fl_client = None
        self.fl_server = FLServer(self)

    def start(self, doLoop=True):
        while self.mqtt_client.status != "running" and doLoop:
            time.sleep(1)
        self.run_loop(doLoop=doLoop)

    def run_loop(self, doLoop=True):
        while self.mqtt_client.status != "stop":
            if self.mqtt_client.status == "idle":
                time.sleep(1)
                continue
            fl.server.start_server(
                server_address=f'{self.ip}:{self.port}',
                strategy=FedAvgCustom(
                    evaluate_fn=self.fl_server.get_evaluate_fn(),
                )
            )
            self.fl_client = FLClient(self)
            self.X = np.concatenate(([self.received_data[key]["x"] for key in self.received_data.keys()]))
            self.y = np.concatenate(([self.received_data[key]["y"] for key in self.received_data.keys()]))
            params = self.fl_client.get_parameters(config={})
            self.fl_client.evaluate(params, config={})
            self.send_result(self.result)
            self.round += 1
            if not doLoop:
                break
        sys.exit()

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
        self.mqtt_client.publish(
            msg=json.dumps({"node": self.name, "round": str(self.round)}),
            topic_send=f"{self.mqtt_client.topic_round}{self.name}/"
        )


if __name__ == "__main__":
    name = os.environ.get("NODE_NAME")
    ip = os.environ.get("NODE_IP")
    port = os.environ.get("NODE_PORT")
    CentralNode(
        name=name,
        ip=ip,
        port=port,
    ).start(doLoop=True)
