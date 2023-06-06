import json

class FLServer:
    def __init__(self, node):
        self._parent = node

    def get_evaluate_fn(self, model=None):
        """
        Return an evaluation function for server-side evaluation.
        """
        def evaluate(server_round, parameters, config):
            self._parent.received_data.update(
                {
                    json.loads(parameters[0].tolist()): {#name
                        "x": json.loads(parameters[2].tolist()), #x
                        "y": json.loads(parameters[3].tolist()) #y
                    }
                }
            )
            self._parent.columns = json.loads(parameters[4].tolist())
            self._parent.nominal_attributes = json.loads(parameters[1].tolist())
            return float(0), {}
        return evaluate
