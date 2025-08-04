import pandas as pd
import numpy as np
import os

def load_joint_data(folder):
    joint_dfs = []
    for i in range(1, 7):
        df = pd.read_csv(os.path.join(folder, f"link{i}.csv"))
        joint_dfs.append(df)
    return pd.concat(joint_dfs, axis=1)

def load_data(folder):
    joint_data = load_joint_data(folder)
    gripper = pd.read_csv(os.path.join(folder, "gripper_base.csv"))
    base = pd.read_csv(os.path.join(folder, "base.csv"))
    bottle = pd.read_csv(os.path.join(folder, "bottle.csv"))
    distance = pd.read_csv(os.path.join(folder, "distance.csv"))

    full_data = pd.concat([joint_data, gripper, base, bottle, distance], axis=1)
    full_data.dropna(inplace=True)
    
    X = full_data.iloc[:-1].values
    y = full_data.iloc[1:, :36].values  # 36 = 6 joints × 6D (x,y,z + rot)
    
    return X, y
