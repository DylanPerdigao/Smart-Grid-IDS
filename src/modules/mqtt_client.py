import os
import random
import socket
import time
import pandas as pd
import json
from paho.mqtt import client as mqtt_client


class MQTTClient:
    """
    Client for MQTT
    """
    def __init__(self, node=None, broker='broker.hivemq.com', port=1883, topic_prefix='smartgrid_ids', username="", password="", buffer_size=100, buffer_path="../temp/"):
        self._parent = node
        """Parent node. Default is `None`"""
        self.client_id = f'htb--mqtt-{random.randint(0, 100)}-'+str(time.time())
        """MQTT Client ID."""
        self.broker = broker
        """MQTT Broker IP address. Default is `broker.hivemq.com`"""
        self.port = port
        """MQTT Broker port. Default is `1883`"""
        self.username = username
        """MQTT Broker username. Default is ``"""
        self.password = password
        """MQTT Broker password. Default is ``"""
        self.topic_prefix = topic_prefix
        """MQTT Broker topic prefix. Default is `smartgrid_ids`"""
        self.topic_command = f"{topic_prefix}/command/"
        """MQTT Broker topic for command. Default is `smartgrid_ids/command/`"""
        self.topic_status = f"{topic_prefix}/status/"
        """MQTT Broker topic for status. Default is `smartgrid_ids/status/`"""
        self.topic_result = f"{topic_prefix}/result/"
        """MQTT Broker topic for result. Default is `smartgrid_ids/result/`"""
        self.topic_round = f"{topic_prefix}/round/"
        """MQTT Broker topic for result. Default is `smartgrid_ids/round/`"""
        self.topic_transmission = f"{topic_prefix}/transmission/"
        """MQTT Broker topic for transmission. Default is `smartgrid_ids/transmission/`"""
        self.buffer_size = buffer_size
        """Buffer size for result. Default is `100`"""
        self.buffer_path = buffer_path
        """Buffer path for result. Default is `./temp/`"""
        self.status = "idle"
        """Status of the client. Default is `idle`"""
        self.last_received_data = None
        self.round = 0
        self.client = self.connect_mqtt()
        self.c.loop_start()

    def close(self):
        self.c.loop_stop()
        self.c.disconnect()

    def connect_mqtt(self) -> mqtt_client:
        def on_connect(client, userdata, flags, rc):
            if rc == 0:
                print("Connected to MQTT Broker!")
                self.subscribe(f"{self.topic_command}#")
            else:
                print("Failed to connect, return code "+str(rc))

        def on_disconnect(client, userdata, rc):
            if rc != 0:
                print("Unexpected disconnection.")

        
        self.c = mqtt_client.Client(self.client_id+str(os.getpid())+str(time.time()))
        self.c.username_pw_set(self.username,self.password)
        self.c.on_connect = on_connect
        self.c.on_disconnect = on_disconnect
        
        #client.tls_set(cert_reqs=ssl.CERT_NONE)
        #client.tls_insecure_set(True)

        self.c.connect(self.broker, self.port)

        #If latency is a concern
        self.c.socket().setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, True)
        return self.c

    #MQTT Subscribe
    def subscribe(self, topic_recv):
        print(f"Subscribe to {topic_recv}")
        # clear temp/mqtt.txt
        with open(f"{self.buffer_path}mqtt.txt", "w") as f:
            f.write("")
        def on_message(client, userdata, msg):
            message=msg.payload.decode().rstrip()
            if "status" in msg.topic:
                print(f"Received `{message}` from `{msg.topic}` topic")
                # TODO: Handle status
            elif "command" in msg.topic:
                print(f"Received `{message}` from `{msg.topic}` topic")
                json_msg = json.loads(message)
                if "command" in json_msg and json_msg["command"] == "start":
                    self.status = "running"
                if "command" in json_msg and json_msg["command"] == "stop":
                    self.status = "stop"
                if "command" in json_msg and json_msg["command"] == "idle":
                    self.status = "idle"
                # TODO: Handle command
            elif "result" in msg.topic:
                # count lines in temp/mqtt.txt
                with open(f"{self.buffer_path}mqtt.txt", "r") as f:
                    lines = f.readlines()
                    count = len(lines)
                # write to temp/mqtt.txt
                if count < self.buffer_size:
                    with open(f"{self.buffer_path}mqtt.txt", "a") as f:
                        f.write(f"{message}\n")
                else:
                    with open(f"{self.buffer_path}mqtt.txt", "w") as f:
                        f.writelines(lines[1:])
                    # append to last line
                    with open(f"{self.buffer_path}mqtt.txt", "a") as f:
                        f.write(f"{message}\n")
            elif "transmission" in msg.topic:
                self.last_received_data = message
                print(f"Received `{message}` from `{msg.topic}` topic")
                df_msg = pd.DataFrame(json.loads(message))
                # append in a temp csv file
                df_msg.to_csv(self._parent.temp_file_path, mode="a", header=False, index=False)
            elif "round" in msg.topic:
                print(f"Received `{message}` from `{msg.topic}` topic")
                json_msg = json.loads(message)
                self.round = int(json_msg["round"])
            print(f"Received `{message}` from `{msg.topic}` topic")

        self.c.subscribe(topic_recv,qos=0)
        self.c.on_message = on_message
        print(f"SUB: {topic_recv}")

    #MQTT Pub
    def publish(self,msg,topic_send):
        result = self.c.publish(topic_send, msg,qos=0)
        # result: [0, 1]
        status = result[0]
        if (status!=0):
            print(f"Failed to send message to topic {topic_send}")

#Initialization
if __name__ == "__main__":	
    client=MQTTClient()
    time.sleep(0.5)	
    command=""
    while (command!="STOP"):
        command=input("\n"+"command:")
        #client.subscribe("smartgrid_ids/#")
        client.publish(command,"smartgrid_ids/command/")
    client.close()
