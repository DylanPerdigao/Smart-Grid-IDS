# Smart-Grid-IDS

## Configuration

### Nodes configuration

The configuration is made on [´./data/nodes.conf´](./data/nodes.conf) file.

### Dataset

The dataset used is the [IEC61850-Security](./data/iec61850_security_dataset/) dataset.

## Running

To run the project, you need to run the following scripts regarding the number of nodes defined in the [´./data/nodes.conf´](./data/nodes.conf) file:

- Central Node: [´./src/central_node.py´](./src/central_node.py)
    - Set these enviroment variable as defined in [´./data/nodes.conf´](./data/nodes.conf):
        - "NODE_NAME"
        - "NODE_IP"
        - "NODE_PORT"

- Central Local Node: [´./src/central_local_node.py´](./src/central_local_node.py)
    - Set these enviroment variable as defined in [´./data/nodes.conf´](./data/nodes.conf):
        - "NODE_NAME"
        - "NODE_IP"
        - "NODE_PORT"
        - "SERVER_IP"
        - "SERVER_PORT"

- Local Node: [´./src/local_node.py´](./src/local_node.py)
    - Set these enviroment variable as defined in [´./data/nodes.conf´](./data/nodes.conf):
        - "NODE_NAME"
        - "SERVER_IP"
        - "SERVER_PORT"

The nodes will wait from the MQTT broker to send the status "running" to start the FL process.
For that it will necessary to run the GUI:

- GUI: [´./src/gui.py´](./src/gui.py)

And then, click on the "Start" button.

To feed the local nodes with data, run the following script:

- Data Generator: [´./src/data_generator.py´](./src/data_generator.py)
