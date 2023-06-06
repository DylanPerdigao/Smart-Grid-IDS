import numpy as np
import pandas as pd
import os
import sys
import time

from modules.mqtt_client import MQTTClient
from modules.utils import read_data, DATASET_NAME

def generate_data(nodes_names,frq=1,samples=1):
    df, _ = read_data(f"./data/{DATASET_NAME}_dataset/")
    client = MQTTClient(buffer_path="./temp/")
    while True:
        data = df.sample(n=samples)
        node_name = np.random.choice(nodes_names)
        time.sleep(frq)
        client.publish(data.to_json(orient='records'), f"smartgrid_ids/transmission/{node_name}")

def main():
    with open(f'./data/nodes.conf', 'r') as f:
        n = f.readlines()
    n = [line for line in n if 'none' in line.lower() and 'cn' not in line.lower()]
    n = [line.split()[2] for line in n]
    generate_data(n,frq=1)

if __name__ == '__main__':
    main()