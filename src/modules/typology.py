import arff
import os
import subprocess
import pandas as pd
import numpy as np
import re

# create class for running jar file
class Typology:
    """
    Class for running Typology.jar
    """
    def __init__(self, path=os.path.dirname(os.path.abspath(__file__)), dataframe=None, input_file=None , remove_temp=True, temp_file="temp",verbose=False):
        self.path = path
        self.temp_file = f'{self.path}/{temp_file}.arff'
        self.typo_nums = pd.DataFrame(columns=["S", "B", "R", "O"])
        self.typo_percentage = pd.DataFrame(columns=["S", "B", "R", "O"])
        if dataframe is not None:
            self.dataframe_fit(dataframe, remove_temp=remove_temp, verbose=verbose)
        elif input_file is not None:
            self.file_fit(input_file, verbose=verbose)
        else:
            assert dataframe is not None or input_file is not None, "Please provide dataframe or input_file"

    def dataframe_fit(self, df, remove_temp=True, verbose=False):
        nominal_attributes = [True if t not in ['float64', 'int64'] else False for t in df.dtypes]
        # label dtype int, convert to float
        try:
            df["label"] = df["label"].astype(float)
        except Exception as e:
            print(e)
        # save df to arff file
        arff.dump(self.temp_file,df.values,relation='temp',names=df.columns)
        # read arff file
        with open(self.temp_file, 'r') as f: lines = f.readlines()
        # change attribute label to class
        classes = ",".join(df["label"].unique().astype(str))
        for i, line in enumerate(lines):
            if line.startswith('@attribute'):
                if 'label' in line.lower():
                    lines[i] = '@attribute class {' + classes + '}\n'
                else :
                    l = lines[i].split()
                    l[2] = "STRING" if nominal_attributes[i-1] else "NUMERIC"
                    lines[i] = " ".join(l) + "\n"

        # write new data to file 
        with open(self.temp_file, 'w') as f: f.writelines(lines)

        # run jar file
        df = self.file_fit(self.temp_file, verbose=verbose)

        # remove temp file
        os.remove(self.temp_file) if remove_temp else None
        return df
    
    def file_fit(self, input_file, verbose=False):
        sp = subprocess.Popen(
            ["/Library/Java/JavaVirtualMachines/jdk1.8.0_321.jdk/Contents/Home/bin/java", 
            "-Xmx10g", # MAX HEAP SIZE
            "-Dfile.encoding=UTF-8", 
            "-jar", f"{os.path.dirname(os.path.abspath(__file__))}/Typology.jar",
            "-I", f"{self.temp_file}",
            "-F", "pl.poznan.put.cs.idss.imbalanced.LabellingFilter"
            ], 
            stdout=subprocess.PIPE
        )
        string = sp.stdout.read().decode('utf-8')
        print(string) if verbose else None
        if string[0].startswith("java.io.FileNotFoundException"):
            assert string[0].startswith("java.io.FileNotFoundException"), f"File not found\n{string}"
        string = string.split()[::2]
        string_percentage = [float(i) for i in string[1:]]
        string_percentage = [i/sum(string_percentage)*100 for i in string_percentage]
        df = pd.DataFrame([string[1:]+string_percentage], columns=['S', 'B', 'R', 'O','S [%]','B [%]','R [%]','O [%]'], index=[string[0]])
        self.typo_nums = df.iloc[0, :4]
        self.typo_percentage = df.iloc[0, 4:]
        self.typo_percentage.rename({'S [%]':'S', 'B [%]':'B', 'R [%]':'R', 'O [%]':'O'}, inplace=True)
        return df
