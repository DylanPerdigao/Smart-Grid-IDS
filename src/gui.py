import json
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import networkx as nx
import os
import sys
import tkinter as tk

from datetime import datetime
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from PIL import Image, ImageTk
from time import sleep, strftime, gmtime

PATH = os.path.dirname(os.path.abspath(__file__)) + "/../"
os.chdir(PATH)
sys.path.append(PATH)

from src.modules.mqtt_client import MQTTClient


class GUI(tk.Tk):
    def __init__(self, mqtt_broker=None, save_reports=True, save_plots=True):
        super().__init__()
        self.dir_path = f"{PATH}data/"
        self.title("Federated Visualization")
        self.width = 1100
        self.height = 800
        self.geometry(f"{self.width}x{self.height}")
        self.is_stopped = True
        self.time = 0
        self.save_reports = save_reports
        self.save_plots = save_plots
        self.prev_round = -1
        # plots
        # create a folder with the actual datetime
        now = datetime.now()
        self.dt_string = now.strftime("%Y%m%d_%H%M%S")
        if self.save_plots and not os.path.exists(f"{PATH}/results/round_plots/{self.dt_string}"):
            os.makedirs(f"{PATH}/results/round_plots/{self.dt_string}")
        if self.save_reports and not os.path.exists(f"{PATH}/results/reports/{self.dt_string}"):
            os.makedirs(f"{PATH}/results/reports/{self.dt_string}")
        if mqtt_broker is not None:
            self.mqtt_client = MQTTClient(broker=mqtt_broker, buffer_path=f"{PATH}/temp/")
        else:
            self.mqtt_client = MQTTClient(buffer_path=f"{PATH}/temp/")
        self.mqtt_client.subscribe(f"{self.mqtt_client.topic_result}#")
        self.mqtt_client.subscribe(f"{self.mqtt_client.topic_round}#")
        ########## init tree
        self.tree = self.read_config()
        ########## draw tree
        self.fig, self.ax = plt.subplots()
        self.update_tree()
        ########## create reports for each node
        if self.save_reports:
            for n in self.tree.nodes:
                #create csv file for reports
                with open(f"{PATH}/results/reports/{self.dt_string}/report_{self.tree.nodes[n]['name']}.csv", "w") as f:
                    f.write("time,simulation_time,round,safe,borderline,rare,outlier,ate,ma7_ate,accuracy,f1_score,precision,recall\n")
        ########## init elements
        self.group = tk.LabelFrame(self, padx=5, pady=5, width=200, height=self.height)
        self.group.pack(side=tk.RIGHT, anchor=tk.N, fill=tk.Y, padx=10, pady=10, expand=False)

        # group buttons at right side
        self.group_control = tk.LabelFrame(self.group, text="Control", padx=5, pady=5)
        self.group_control.pack(side=tk.TOP, anchor=tk.N, fill=tk.X, padx=10, pady=10)

        # add stacked buttons to the right side with the respective order start, stop, exit
        self.button_frame = tk.Frame(self.group_control)
        self.button_frame.pack(side=tk.BOTTOM, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.start_button = tk.Button(self.button_frame, text="Start", command=self.start_time)
        self.start_button.pack(side=tk.LEFT, anchor=tk.S, fill=tk.X, padx=10, pady=10)
        self.stop_button = tk.Button(self.button_frame, text="Pause", command=self.stop_time)
        self.stop_button.pack(side=tk.LEFT, anchor=tk.S, fill=tk.X, padx=10, pady=10)
        self.exit_button = tk.Button(self.button_frame, text="Exit", command=self.exit)
        self.exit_button.pack(side=tk.RIGHT, anchor=tk.S, fill=tk.X, padx=10, pady=10)

        # add stacked labels to the left side with the respective order time, time value, rounds, rounds value
        self.temporal_frame = tk.Frame(self.group_control)
        self.temporal_frame.pack(side=tk.TOP, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.time_value = tk.Label(self.temporal_frame, text="Time\n--:--:--")
        self.time_value.pack(side=tk.LEFT, anchor=tk.S, fill=tk.X, padx=10, pady=10)
        self.round_value = tk.Label(self.temporal_frame, text="Rounds\n0")
        self.round_value.pack(side=tk.RIGHT, anchor=tk.S, fill=tk.X, padx=10, pady=10)
        
        # add text box for logs
        self.group_log = tk.LabelFrame(self.group, text="Node", padx=5, pady=5)
        self.group_log.pack(side=tk.TOP, anchor=tk.N, fill=tk.Y, padx=10, pady=10)

        # add dropdown with list of nodes
        self.node_list = tk.StringVar(self.group_log)
        self.node_list.set("SELECT NODE")
        self.node_dropdown = tk.OptionMenu(self.group_log, self.node_list, *[self.tree.nodes[n]["name"] for n in self.tree.nodes])
        self.node_dropdown.pack(side=tk.TOP, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.node_list.trace_add("write", self.update_tree)
        
        self.dataset_info_frame = tk.Frame(self.group_log)
        self.dataset_info_frame.pack(side=tk.TOP, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.instances_value = tk.Label(self.dataset_info_frame, text="Instances\n--")
        self.instances_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.min_instances_value = tk.Label(self.dataset_info_frame, text="Min. Class Instances\n--")
        self.min_instances_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)
    
        # add labels for the number of instances of each class
        self.taxonomy_frame = tk.Frame(self.group_log)
        self.taxonomy_frame.pack(side=tk.TOP, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.safe_value = tk.Label(self.taxonomy_frame, text="Safe\n--")
        self.safe_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.borderline_value = tk.Label(self.taxonomy_frame, text="Borderline\n--")
        self.borderline_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.rare_value = tk.Label(self.taxonomy_frame, text="Rare\n--")
        self.rare_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.outlier_value = tk.Label(self.taxonomy_frame, text="Outlier\n--")
        self.outlier_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)

        # add info of the average treatment effect
        self.causality_frame = tk.Frame(self.group_log)
        self.causality_frame.pack(side=tk.TOP, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.ate_value = tk.Label(self.causality_frame, text="ATE\n--")
        self.ate_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.ma7_ate_value = tk.Label(self.causality_frame, text="ATE (MA7)\n--")
        self.ma7_ate_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)

        # add info of the average treatment effect
        self.classification_frame = tk.Frame(self.group_log)
        self.classification_frame.pack(side=tk.TOP, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.acc_value = tk.Label(self.classification_frame, text="Accuracy\n--")
        self.acc_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.f1_value = tk.Label(self.classification_frame, text="F1-Score\n--")
        self.f1_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.prec_value = tk.Label(self.classification_frame, text="Precsision\n--")
        self.prec_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        self.rec_value = tk.Label(self.classification_frame, text="Recall\n--")
        self.rec_value.pack(side=tk.LEFT, anchor=tk.N, fill=tk.X, padx=10, pady=10)
        # add canvas
        self.canvas = FigureCanvasTkAgg(self.fig, self)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)
        self.canvas.draw()

    def read_config(self):
        # read config file
        with open(f"{self.dir_path}/nodes.conf") as f:
            lines = f.readlines()
        # create tree
        tree = nx.Graph()
        for line in lines:
            # remove comments
            line = line.split("#")[0]
            # remove whitespaces
            line = line.strip()
            # skip empty lines
            if not line:
                continue
            # split line
            parent, child, name, src_addr, dst_addr = line.split()
            src_ip, src_port = src_addr.split(":") if "None" not in src_addr else ["None", "None"]
            dst_ip, dst_port = dst_addr.split(":") if "None" not in dst_addr else ["None", "None"]
            level = int(parent)+1
            # add node
            if level == 1:
                tree.add_node(int(child), 
                    name=name, 
                    taxonomy=None#[0, 0, 0, 100], 
                )
            elif level > 1:

                tree.add_node(int(child), 
                    name=name, 
                    taxonomy=None#[0, 0, 0, 100], 
                ) 
                tree.add_edge(int(parent), int(child))
            else:
                assert False, "Invalid level"
        return tree

    def exit(self):
        self.exit_fl()
        self.start_button.destroy()
        self.stop_button.destroy()
        self.time_value.destroy()
        self.exit_button.destroy()
        self.group.destroy()
        self.canvas.get_tk_widget().destroy()
        self.destroy()
        sys.exit()

    def update_tree(self, *args):
        taxonomy_name = ["safe", "borderline", "rare", "outlier"]
        colors = [f'#00FF00', f'#FFFF00', f'#FFA500', f'#FF0000']

        # init
        self.fig.clear()
        self.ax.clear()
        self.fig.add_axes(self.ax)
        plt.axis('off')

        # draw edges
        pos = nx.nx_agraph.graphviz_layout(self.tree, prog="dot", args="")
        nx.draw_networkx_edges(self.tree, pos,
                               ax=self.ax,
                               edge_color='darkblue',
                               width=3,
                               alpha=0.25,
                               )
        # draw legend
        if self.time == 0:
            self.fig.legend(["connection"], loc="upper left")
        # draw nodes
        trans = self.ax.transData.transform
        trans2 = self.fig.transFigure.inverted().transform
        pie_size = 0.2
        p2 = pie_size / 2.0

        for n in self.tree:
            # get values from tree
            values = self.tree.nodes[n]["taxonomy"]
            try:
                values_nums = self.tree.nodes[n]["taxonomy_nums"]
            except:
                values_nums = ["--", "--", "--", "--"]
            if values is not None:
                text_values = [f"{x:.1f}%" if x != 0 else None for x in values]
            name = self.tree.nodes[n]["name"]

            # compute coordinates
            xx, yy = trans(pos[n])  # figure coordinates
            xa, ya = trans2((xx, yy))  # axes coordinates

            a = plt.axes([xa - p2, ya - p2, pie_size, pie_size])

            # draw pie chart
            if values is None:
                a.pie([0,0,0,100], colors=['lightblue', 'lightblue', 'lightblue', 'lightblue'], startangle=90)
            else:
                a.pie(values,
                      labels=text_values,
                      radius=1,
                      colors=colors,
                      startangle=90,
                      wedgeprops=dict(width=1, edgecolor='w'),
                      )
                # check which node is selected in the dropdown menu
                if name == self.node_list.get():
                    # update values instances
                    self.instances_value.config(text=f"Instances\n{self.tree.nodes[n]['instances']}")
                    self.min_instances_value.config(text=f"Min. Class Instances\n{self.tree.nodes[n]['min_instances']}")
                    # update values in the log
                    self.safe_value.config(text=f"Safe\n{text_values[0]}\n({values_nums[0]})")
                    self.borderline_value.config(text=f"Borderline\n{text_values[1]}\n({values_nums[1]})")
                    self.rare_value.config(text=f"Rare\n{text_values[2]}\n({values_nums[2]})")
                    self.outlier_value.config(text=f"Outlier\n{text_values[3]}\n({values_nums[3]})")
                    # update values in the log
                    self.ate_value.config(text=f"ATE\n{self.tree.nodes[n]['ate']:.2f}")
                    self.ma7_ate_value.config(text=f"ATE (MA7)\n{self.tree.nodes[n]['ma7_ate']:.2f}")
                    # classification values
                    self.acc_value.config(text=f"Accuracy\n{self.tree.nodes[n]['accuracy']:.2f}")
                    self.f1_value.config(text=f"F1\n{self.tree.nodes[n]['macro_f1']:.2f}")
                    self.prec_value.config(text=f"Precision\n{self.tree.nodes[n]['macro_precision']:.2f}")
                    self.rec_value.config(text=f"Recall\n{self.tree.nodes[n]['macro_recall']:.2f}")
                    # log values
                    #self.log_text.insert(tk.INSERT, f"{name.split()[0]}: {text_values}\n")
                if self.save_reports and self.prev_round != self.mqtt_client.round:
                    #create csv file for reports
                    with open(f"{PATH}/results/reports/{self.dt_string}/report_{self.tree.nodes[n]['name']}.csv", "a") as f:
                        # get current time
                        now = datetime.now()
                        f.write(f"{now},{self.time},{str(self.mqtt_client.round)},{values[0]},{values[1]},{values[2]},{values[3]},"+
                                f"{self.tree.nodes[n]['ate']},{self.tree.nodes[n]['ma7_ate']},"+
                                f"{self.tree.nodes[n]['accuracy']},{self.tree.nodes[n]['macro_f1']},{self.tree.nodes[n]['macro_precision']},{self.tree.nodes[n]['macro_recall']}\n")
                # log values
                #self.log_text.insert(tk.INSERT, f"{name.split()[0]}: {text_values}\n")

            # draw circle in the middle
            circle = plt.Circle((0, 0), 0.7, color='white')
            p = plt.gcf()
            p.gca().add_artist(circle)

            # node name
            a.text(0.5, 0.5, name,
                   transform=a.transAxes,
                   va='center',
                   ha='center',
                   fontsize=8,
                   fontweight='bold',
                   )
        # update previous round
        if self.prev_round != self.mqtt_client.round:
            self.prev_round = self.mqtt_client.round
        # draw legend
        if self.time > 0:
            self.fig.legend(["connection"] + taxonomy_name, loc="upper left")
        if self.save_plots:
            plt.savefig(f"{self.dir_path}/round_plots/{self.dt_string}/round_{self.mqtt_client.round}.png", dpi=300, bbox_inches='tight', pad_inches=0.05)

    def stop_time(self):
        self.is_stopped = True
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        #self.log_text.insert(tk.INSERT, "Stoping FL\n")
        self.stop_fl()

    def start_time(self):
        self.is_stopped = False
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        #self.log_text.insert(tk.INSERT, "Starting FL\n")
        self.run()

    def read_results(self):
        print("Reading results")
        with open(f"{PATH}/temp/mqtt.txt", "r") as f:
            lines = f.readlines()
            reads = []
            for line in lines:
                line_dict = json.loads(line)
                node_name = line_dict["node"]
                if not line or node_name in reads:
                    continue
                taxonomy = [line_dict["S"],line_dict["B"],line_dict["R"],line_dict["O"]]
                taxonomy_nums = [line_dict["num_S"],line_dict["num_B"],line_dict["num_R"],line_dict["num_O"]]
                for n in self.tree:
                    if self.tree.nodes[n]["name"] == node_name:
                        self.tree.nodes[n]["instances"] = line_dict["instances"]
                        self.tree.nodes[n]["min_instances"] = line_dict["min_instances"]
                        self.tree.nodes[n]["taxonomy"] = taxonomy
                        self.tree.nodes[n]["taxonomy_nums"] = taxonomy_nums
                        self.tree.nodes[n]["ate"] = line_dict["ate"]
                        self.tree.nodes[n]["ma7_ate"] = line_dict["ma7_ate"]
                        self.tree.nodes[n]["accuracy"] = line_dict["accuracy"]
                        self.tree.nodes[n]["macro_f1"] = line_dict["macro_f1"]
                        self.tree.nodes[n]["macro_precision"] = line_dict["macro_precision"]
                        self.tree.nodes[n]["macro_recall"] = line_dict["macro_recall"]
                        self.tree.nodes[n]["weighted_f1"] = line_dict["weighted_f1"]
                        self.tree.nodes[n]["weighted_precision"] = line_dict["weighted_precision"]
                        self.tree.nodes[n]["weighted_recall"] = line_dict["weighted_recall"]
                        reads.append(node_name)
                        #print(reads)
                        break
        #self.log_text.insert(tk.INSERT, "Reading results\n")

    def begin_fl(self):
        print("Begin FL")
        self.mqtt_client.publish(
            msg=json.dumps({"node": "main" ,"command" : "start"}),
            topic_send=f"{self.mqtt_client.topic_command}",
        )
        #self.log_text.insert(tk.INSERT, "Begin FL\n")

    def exit_fl(self):
        self.mqtt_client.publish(
            msg=json.dumps({"node": "main" ,"command" : "stop"}),
            topic_send=f"{self.mqtt_client.topic_command}",
        )
        #self.log_text.insert(tk.INSERT, "Stop FL\n")

    def stop_fl(self):
        self.mqtt_client.publish(
            msg=json.dumps({"node": "main" ,"command" : "idle"}),
            topic_send=f"{self.mqtt_client.topic_command}",
        )
        #self.log_text.insert(tk.INSERT, "Pause FL\n")

    def run(self):
        if self.node_list.get() == "SELECT NODE":
            self.node_list.set(str(self.tree.nodes[1]["name"]))
        self.begin_fl()
        time_steps = 0.25
        while True:
            sleep(time_steps)
            if self.is_stopped:
                break
            self.time += time_steps
            if self.time % 1 == 0:
                self.time_value.config(text=f"Time\n{strftime('%H:%M:%S', gmtime(self.time))}")
                #self.log_text.insert(tk.INSERT, f"=========\nTIME: {strftime('%H:%M:%S', gmtime(self.time))}\n")
                ########## update figure
                self.read_results()
                self.update_tree()


            # update round from mqtt
            self.round_value.config(text=f"Rounds\n{str(self.mqtt_client.round)}")
            # re-drawing the figure
            self.canvas.draw()
            # to flush the GUI events
            self.canvas.flush_events()
            self.update()
                


def main():
    #gui = GUI(mqtt_broker="10.254.0.250", save_plots=True)
    gui = GUI(save_reports=True, save_plots=False)
    gui.mainloop()


if __name__ == '__main__':
    main()
