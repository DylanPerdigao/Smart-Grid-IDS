import pandas as pd
import pickle as pkl
import networkx as nx
import numpy as np
import os
import sys

from causalnex.discretiser import Discretiser
from collections import Counter
from dowhy import CausalModel
from imblearn.over_sampling import SMOTE
from scipy.io import arff
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder

DATASET_NAME = "iec61850_security"
IS_BASELINE_SCENARIO = False

def read_data(folder_path, max_rows=None, nominal=False, verbose=False):
    """
    Read all files in a folder and concat them into a single dataframe and return it
    :param folder_path: path to folder
    :param max_rows: max rows to read
    :param verbose: print info
    :return: tuple of dataframes (binary, multiclass)
    """
    df_list = []
    # read all files
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if "edge-iiot_dataset" in folder_path and "ML-EdgeIIoT-dataset.csv" not in file:
                continue
            if ("iot-23_dataset" in folder_path and "conn.log.labeled" not in file) or (
                    "iot-23_dataset" in folder_path and "conn.log.labeled" in file and "honeypot" in root.lower()):
                continue
            if "kdd99_dataset" in folder_path and "data.corrected" not in file:
                continue
            if "nsl-kdd_dataset" in folder_path and "+.arff" not in file:
                continue
            if "ton-iot_datasets" in folder_path and "Train_Test_Network.csv" not in file:
                continue
            if "unsw-nb15_dataset" in folder_path and "-set.csv" not in file:
                continue
            if "iscx-ids-2012_dataset" in folder_path and ".xml" not in file:
                continue
            if "iec61850_security_dataset" in folder_path and "conn.log" not in file:
                continue
            if "cse-cic-ids2018_dataset" in folder_path and ".csv" not in file:
                continue
            print("Reading file: ", file) if verbose else None
            if "iot-23_dataset" in folder_path:
                df = pd.read_csv(
                    os.path.join(root, file),
                    sep="\t",
                    na_values=["-", "(empty)"],
                    header=None,
                    names=["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service",
                           "duration", "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp",
                           "missed_bytes", "history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
                           "label"],
                    index_col=False,
                    comment="#",
                    low_memory=False,
                    nrows=max_rows
                )
            elif "iec61850_security_dataset" in folder_path:
                df = pd.read_csv(
                    os.path.join(root, file),
                    sep="\t",
                    na_values=["-", "(empty)"],
                    header=None,
                    names=["ts", "uid", "id.orig_h", "id.orig_p", "id.resp_h", "id.resp_p", "proto", "service",
                           "duration", "orig_bytes", "resp_bytes", "conn_state", "local_orig", "local_resp",
                           "missed_bytes", "history", "orig_pkts", "orig_ip_bytes", "resp_pkts", "resp_ip_bytes",
                           "tunnel_parents"],
                    index_col=False,
                    comment="#",
                    low_memory=False,
                    nrows=max_rows
                )
                # label the whole dataframe with the folder name
                if "attack" in root.lower():
                    if "CA" in root:
                        df["label"] = "CA"
                    elif "DM" in root:
                        df["label"] = "DM"
                    elif "DoS" in root:
                        df["label"] = "DoS"
                    elif "MS" in root:
                        df["label"] = "MS"
                elif "normal" in root.lower():
                    df["label"] = "Normal"
                elif "disturbance" in root.lower():
                    df["label"] = "Normal"
            elif "nsl-kdd_dataset" in folder_path:
                data = arff.loadarff(os.path.join(root, file))
                df = pd.DataFrame(data[0])
            elif "iscx-ids-2012_dataset" in folder_path:
                df = pd.read_xml(
                    os.path.join(root, file),
                    encoding="utf-8",
                )
            elif "edge-iiot_dataset" in folder_path:
                df = pd.read_csv(os.path.join(root, file), 
                    sep=",", 
                    index_col=False, 
                    low_memory=False,
                )
                # remove last 2 rows
                df = df[:-2]
            else:
                names_list = None
                if "kdd99_dataset" in folder_path:
                    # read file with names this is each lines except the first one
                    df_names = pd.read_csv(
                        os.path.join(root, "kddcup.names"),
                        skiprows=1,
                        sep=":",
                        header=None,
                    )
                    names_list = df_names[0].values.tolist()
                    names_list.append("label")
                df = pd.read_csv(
                    os.path.join(root, file),
                    nrows=max_rows,
                    names=names_list,
                    low_memory=False,
                    index_col=False
                )
            print("Shape: ", df.shape) if verbose else None
            df_list.append(df)
    # concat all files
    df = pd.concat(df_list, axis=0)
    # columns to lower and replace space with underscore
    print("Columns: ", df.columns) if verbose else None
    df.columns = [col.lower().strip().replace(" ", "_") for col in df.columns]
    print(df.head()) if verbose else None
    if "bot-iot_dataset" in folder_path:
        # ['Normal' 'Reconnaissance' 'DoS' 'DDoS' 'Theft']
        # stime as index
        df = df.set_index("stime")
        df = df.sort_index()
        # drop columns
        df_bin = df.iloc[:, :-3]
        df_bin["label"] = df["attack"]
        df_multi = df.iloc[:, :-3]
        df_multi["label"] = df["category"].apply(lambda x: str(0) if str(x).lower().strip() == "normal" else x)
        nominal_attributes = [False, True, False, True, False, True, True, True, True, False, False, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    elif "cic-ids-2017_dataset" in folder_path:
        #  ['BENIGN' 'Infiltration' 'Bot' 'PortScan' 'DDoS' 'FTP-Patator''DoS slowloris' 'DoS Slowhttptest' 'DoS Hulk' 'Web Attack � Brute Force' 'Web Attack � XSS' 'Web Attack � Sql Injection']
        # drop columns
        df_bin = df.iloc[:, :-1]
        df_multi = df.iloc[:, :-1]
        # change benign to 0 and other to 1
        df_bin["label"] = df["label"].apply(lambda x: 0 if str(x).lower().strip() == "benign" else 1)
        # remove special characters and unknown characters
        df_multi["label"] = df["label"].apply(lambda x: str(0) if str(x).lower().strip() == "benign" else x)
        nominal_attributes = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    elif "iscx-ids-2012_dataset" in folder_path:
        #  [normal 'Attack']
        # drop columns
        df_bin = df.drop(["tag"], axis=1)
        # change benign to 0 and other to 1
        df_bin["label"] = df["tag"].apply(lambda x: 0 if str(x).lower().strip() == "normal" else 1)
        df_multi = pd.DataFrame(["no_multiclass"], columns=["label"])
        nominal_attributes = [True, False, False, False, False, True, True, True, True, True, True, True, True, True, False, True, False, True, True, False, False]
    elif "cse-cic-ids-2018_dataset" in folder_path:
        # ['Benign' 'Bot' 'FTP-BruteForce' 'DoS attacks-GoldenEye''DoS attacks-Slowloris' 'DoS attacks-Hulk' 'DoS attacks-SlowHTTPTest' 'DDoS attacks-LOIC-HTTP' 'DDOS attack-HOIC' 'DDOS attack-LOIC-UDP''Brute Force -XSS' 'SQL Injection' 'Brute Force -Web' 'Label']
        # stime as index
        df = df.set_index("timestamp")
        df = df.sort_index()
        df.replace(np.inf,np.NaN,inplace=True)
        df.replace(np.inf,np.NaN,inplace=True)
        # drop columns
        df_bin = df.drop(["label"], axis=1)
        df_multi = df.drop(["label"], axis=1)
        df_bin["label"] = df["label"].apply(lambda x: 0 if str(x).lower().strip() == "benign" else 1)
        df_multi["label"] = df["label"].apply(lambda x: str(0) if str(x).lower().strip() == "benign" else x)
        nominal_attributes = [False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, True, True, False, True]
    elif "edge-iiot_dataset" in folder_path:
        #  ['Ransomware' 'Normal' 'Password' 'DDoS_HTTP' 'Port_Scanning' 'XSS' 'Backdoor' 'DDoS_TCP' 'Vulnerability_scanner' 'SQL_injection' 'Fingerprinting' 'Uploading' 'DDoS_ICMP' 'MITM' 'DDoS_UDP']
        # stime as index
        df = df.set_index("frame.time")
        df = df.sort_index()
        # drop columns
        df_bin = df.iloc[:, :-2]
        df_bin["label"] = df["attack_label"]
        df_multi = df.iloc[:, :-2]
        df_multi["label"] = df["attack_type"].apply(lambda x: str(0) if str(x).lower().strip() == "normal" else x)
        nominal_attributes = [True, True, True, True, True, True, False, False, False, True, True, False, True, True, True, True, True, True, True, False, False, False, True, True, True, True, False, False, True, False, True, True, False, True, True, False, True, False, True, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True, True]
    elif "iot-23_dataset" in folder_path:
        # ['PartOfAHorizontalPortScan' 'Benign' 'Okiru' 'C&CHeartBeat' 'C&C' 'C&CFileDownload' 'FileDownload' 'C&CHeartBeatFileDownload']
        # stime as index
        df = df.set_index("ts")
        df = df.sort_index()
        # drop columns uid
        df = df.drop(["uid"], axis=1)
        df_bin = df.iloc[:, :-1]
        df_bin["label"] = df["label"].apply(lambda x: 0 if "benign" in str(x).lower().strip() else 1)
        df_multi = df.iloc[:, :-1]
        df_multi["label"] = df["label"].apply(
            lambda x: x.replace("-", "").replace("Malicious", "").replace("(empty)", "").strip() if "malicious" in str(
                x).lower().strip() else str(0))
        nominal_attributes = [True, False, True, False, True, True, False, False, False, True, True, True, True, True, False, False, False, False, True]
    elif "kdd99_dataset" in folder_path:
        # [normal 'buffer_overflow' 'loadmodule' 'perl' 'neptune' 'smurf' 'guess_passwd''pod' 'teardrop' 'portsweep' 'ipsweep' 'land' 'ftp_write' 'back' 'imap''satan' 'phf' 'nmap' 'multihop' 'warezmaster' 'warezclient' 'spy''rootkit']
        df_bin = df.iloc[:, :-1]
        df_bin["label"] = df["label"].apply(lambda x: 0 if "normal" in str(x).lower().strip() else 1)
        df_multi = df.iloc[:, :-1]
        df_multi["label"] = df["label"].apply(lambda x: str(0) if "normal" in str(x).lower().strip() else str(x).replace(".", ""))
        nominal_attributes = [False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    elif "nsl-kdd_dataset" in folder_path:
        # [normal anomaly]
        df_bin = df.iloc[:, :-1]
        df_bin["label"] = df["class"].apply(lambda x: 0 if "normal" in str(x).lower().strip() else 1)
        df_multi = pd.DataFrame(["no_multiclass"], columns=["label"])
        nominal_attributes = [False, True, True, True, False, False, True, False, False, False, False, True, False, False, False, False, False, False, False, False, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    elif "unsw-nb15_dataset" in folder_path:
        # [Normal 'Backdoor' 'Analysis' 'Fuzzers' 'Shellcode' 'Reconnaissance' 'Exploits' 'DoS' 'Worms' 'Generic']
        # drop columns uid
        df = df.drop(["id"], axis=1)
        df_bin = df.iloc[:, :-2]
        df_bin["label"] = df["label"]
        df_multi = df.iloc[:, :-2]
        df_multi["label"] = df["attack_cat"].apply(lambda x: str(0) if "normal" in str(x).lower().strip() else x)
        nominal_attributes = [False, True, True, True, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False, False]
    elif "ton-iot_datasets" in folder_path:
        # ['PartOfAHorizontalPortScan' 'Benign' 'Okiru' 'C&CHeartBeat' 'C&C' 'C&CFileDownload' 'FileDownload' 'C&CHeartBeatFileDownload']
        # stime as index
        df = df.set_index("ts")
        df = df.sort_index()
        # drop columns uid
        df_bin = df.iloc[:, :-2]
        df_bin["label"] = df["label"]
        df_multi = df.iloc[:, :-2]
        df_multi["label"] = df["type"].apply(lambda x: str(0) if str(x).lower().strip() == "normal" else x)
        nominal_attributes = [True, False, True, False, True, True, False, False, False, True, False, False, False, False, False, True, False, False, False, True, True, True, True, True, True, True, True, True, True, True, True, True, True, False, False, False, True, True, True, True, True, True]
    elif "iec61850_security_dataset" in folder_path:
        # [normal 'MS' 'CA' 'DoS' 'DM']
        # stime as index
        df = df.set_index("ts")
        df = df.sort_index()
        # drop columns
        df = df.drop(["uid"], axis=1)
        df_bin = df.drop(["label"], axis=1)
        df_multi = df.drop(["label"], axis=1)
        df_bin["label"] = df["label"].apply(lambda x: 0 if str(x).lower().strip() == "normal" else 1)
        df_multi["label"] = df["label"].apply(lambda x: str(0) if str(x).lower().strip() == "normal" else x)
        nominal_attributes = [True, False, True, False, True, True, False, False, True, True, True, True, True, True, False, False, True, True, True]
    else:
        print("Invalid folder path")
        sys.exit()

    print("Labels: ", df_multi["label"].unique()) if verbose else None
    if nominal:
        return df_bin, df_multi, nominal_attributes
    return df_bin, df_multi

def smote_rebalance(df, percentage=0.3, seed=None, return_df=True, verbose=False):
    """
    Rebalance data using SMOTE
    :param df: dataframe
    :param percentage: percentage of minority class
    :return: rebalanced dataframe
    """
    df_copy = df.copy()
    # split data
    x = df_copy.drop(["label"], axis=1)
    y = df_copy["label"]
    print("Oversampling with SMOTE") if verbose else None
    # get majority class
    counter = Counter(y.to_numpy().tolist())
    majority_class_label = max(counter, key=counter.get)
    # dict with 30% more samples for each class except the majority class
    sampling_strategy = {k: int(v * (1+percentage)) for k, v in counter.items() if k != majority_class_label}  
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=seed)
    x_smote, y_smote = smote.fit_resample(x, y)
    # create dataframe
    if return_df:
        df_smote = pd.DataFrame(x_smote, columns=df_copy.columns[:-1])
        df_smote["label"] = y_smote
        print("Oversampling with SMOTE done") if verbose else None
        return df_smote
    return x_smote.to_numpy(), y_smote.to_numpy()

def preprocessing(df, verbose=False):
    # drop columns with all values equal
    df_copy = df.copy()
    # replace all empty values or "-" with NaN
    print("Replacing empty values and - with NaN") if verbose else None
    df_copy.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    df_copy.replace(r'^\-$', np.nan, regex=True, inplace=True)
    # label encoding
    if len(df["label"].unique()) > 2:
        print("Label encoding") if verbose else None
        le = LabelEncoder()
        df_copy["label"] = le.fit_transform(df_copy["label"])
        print(le.classes_)
    # ordinal encoder
    print("Ordinal encoding") if verbose else None
    oe = OrdinalEncoder() 
    # convert object dtypes to string
    df_copy[df_copy.select_dtypes(include=['object']).columns] = df_copy.select_dtypes(include=['object']).astype(str)
    # fit and transform
    df_copy[df_copy.select_dtypes(include=['object']).columns] = oe.fit_transform(df_copy.select_dtypes(include=['object']))
    # replace NaN with -1
    print("Replacing NaN with -1") if verbose else None
    df_copy.fillna(-1, inplace=True)
    # normalizing but not the label
    print("Normalizing") if verbose else None
    scaler = StandardScaler()
    df_copy.iloc[:, :-1] = scaler.fit_transform(df_copy.iloc[:, :-1])
    return df_copy

def sample_data(df, n, seed=None, verbose=False):
    """Sample data preserving class distribution.
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe containing data to be sampled.
    n : int
        Number of samples to be drawn.

    Returns
    -------
    df_sample : pandas.DataFrame
        Dataframe containing sampled data.
    """
    if n > len(df):
        print("n is greater than the number of samples")
        return df
    df_sample = pd.DataFrame()
    counter = Counter(df.iloc[:, -1])
    total = sum(counter.values())
    proportion = {key: value / total for key, value in counter.items()}
    print(proportion) if verbose else None
    for key, value in proportion.items():
        df_class = df[df.iloc[:, -1] == key]
        samples = round(n * value)
        if samples <= 10:
            print(f"Class {key} has less than 10 samples.") if verbose else None
            df_class_sample = df_class.sample(n=10, random_state=seed, replace=True)
        else:
            df_class_sample = df_class.sample(n=samples, random_state=seed)
        df_sample = pd.concat([df_sample, df_class_sample])
    df_sample = df_sample.sort_index()
    return df_sample

def preprocess_data_causality(df):
    print("Preprocessing data...")
    df = df.copy()
    # infinte values to NaN
    df.replace(np.inf, np.nan, inplace=True)
    df.replace(-np.inf, np.nan, inplace=True)
    # remove columns where 90% of the values are NaN
    df = df.dropna(thresh=0.1*len(df), axis=1)
    # The missing or null values of the remain-ing attributes by replacing them with the median of all validvalues of that attribute
    df = df.fillna(df.median(numeric_only=True))
    # remove columns where all values are the same
    #df = df.loc[:, (df != df.iloc[0]).any()]
    # add index column to df
    # df = df.reset_index()
    # rename last column to label
    if df.columns[-1] != "label":
        df = df.rename(columns={df.columns[-1]: "label"})
    print(df.columns)

    struct_data = df.copy()
    non_numeric_columns = list(struct_data.select_dtypes(exclude=[np.number]).columns)
    le = LabelEncoder()
    for col in non_numeric_columns:
        # col as string
        struct_data[col] = struct_data[col].astype(str)
        struct_data[col] = le.fit_transform(struct_data[col])
    # encode index column
    struct_data.index = le.fit_transform(struct_data.index)
    return struct_data

def get_dags(path, dataset_name):
    # search all pickles in the folder beginning with "model_dag"
    dag_paths = [f for f in os.listdir(path) if f.startswith(f"model_dag_{dataset_name}")]
    # select the dag
    # load the dag
    with open(f"{path}/{dag_paths[0]}", "rb") as f:
        dag = pkl.load(f)
    return dag

def convert_to_nx(sm):
    # convert to networkx graph
    graph = nx.DiGraph()
    for edge in sm.edges:
        graph.add_edge(edge[0], edge[1])
    outcomes = ["label"]
    treatments = set()
    for node in graph.nodes:
        print(f"Child node: {node} --> {list(graph.successors(node))}")
        if outcomes == list(graph.successors(node)):
            treatments.add(node)
    confounders = list(set(sm.nodes) - treatments - set(outcomes))
    return graph, outcomes, list(treatments), confounders

def compute_causality(df, dag_file_path, dot_file_path):
    """Compute causality between features and label."""
    dag_path, dag_file = os.path.split(dag_file_path)
    dataset_name = dag_file.split("_")[2]
    dag = get_dags(dag_path, dataset_name)
    graph, outcomes, treatments, confounders = convert_to_nx(dag)
    
    # get all columns names from the graph
    col = []
    for node in graph.nodes:
        col.append(node)
    df_c = df.loc[:, col]
    
    for i in range(len(col)-1):
        c = col[i]
        df_c[c] = Discretiser(method="uniform",num_buckets=2).fit(df_c[c].values).transform(df_c[c].values)
    
    nx.drawing.nx_pydot.write_dot(graph, dot_file_path)

    model=CausalModel(
        data = df_c,
        treatment=treatments,
        outcome=outcomes[0],
        graph=dot_file_path,
    )
    # Estimate the causal effect and compare it with Average Treatment Effect
    identified_estimand = model.identify_effect(
        proceed_when_unidentifiable=True
    )
    #print(identified_estimand)
        
    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression", 
        test_significance=True,
        confidence_intervals=True,
    )
    return estimate.value


def get_model_parameters(model):
    """Returns the paramters of a model."""
    params = [
        model.matrix,
        model.nominal_attributes,
    ]
    return params


def set_model_params(model, params):
    """Sets the parameters of a model."""
    model.matrix = params[0]
    model.nominal_attributes = params[1]
    return model


def set_initial_params(model):
    """Sets initial parameters as zeros Required since model params are
    uninitialized until model.fit is called.

    But server asks for initial parameters from clients at launch. Refer
    to sklearn.linear_model.LogisticRegression documentation for more
    information.
    """
    return model

def partition(X, y, num_partitions):
    """Split X and y into a number of partitions."""
    return list(
        zip(np.array_split(X, num_partitions), np.array_split(y, num_partitions))
    )
def partition_df(X, y, num_partitions):
    """Split X and y into a number of partitions."""
    # concatenate X and y
    df_sampled = pd.DataFrame()
    df = pd.concat([X, y], axis=1)
    num_classes = len(np.unique(y))
    i = 0
    while i == 0 or df_sampled.iloc[:, -1].nunique() < num_classes:
        df_sampled = df.sample(frac=(1/num_partitions))
        i += 1
    df_sampled = df_sampled.sort_index()
    X = df_sampled.iloc[:, :-1]
    y = df_sampled.iloc[:, -1]
    return X, y

if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    print(path)
    df_bin, df_multi = read_data(f"{path}/../../data/iec61850_security_dataset/")
    df_s = sample_data(df_multi, 100)
    print(df_s)
    print(df_s.iloc[:, -1].value_counts())