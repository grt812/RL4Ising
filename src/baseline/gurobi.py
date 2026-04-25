import networkx as nx
import numpy as np
import os
import math
import gurobipy as gp
import argparse
import csv
import re
from utils import read_nxgraph
from utils import float_to_binary
from utils import base64_encode


def save_to_file(model, file_name, time_limit, out_dir, print_terminal=True):
    obj_val = model.ObjVal
    obj_bnd = model.ObjBound
    duration = model.Runtime
    mip_gap = model.MIPGap
    best_solution = "".join([float_to_binary(x.X) for x in model.getVars()])
    best_encoded = base64_encode(best_solution)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    base_name = os.path.basename(file_name)
    with open(os.path.join(out_dir, base_name), "w") as f:
        f.write(f"Objective Value: {obj_val}\n")
        f.write(f"Objective Bound: {obj_bnd}\n")
        f.write(f"Duration: {duration}\n")
        f.write(f"Time Limit: {time_limit}\n")
        f.write(f"MIP Gap: {mip_gap}\n")
        f.write(f"Best Solution Encoded: {best_encoded}\n")
        f.write(f"Best Solution Raw: {best_solution}\n")

    if print_terminal:
        print(f"Objective Value: {obj_val}")
        print(f"Objective Bound: {obj_bnd}")
        print(f"Duration: {duration}")
        print(f"Time Limit: {time_limit}")
        print(f"MIP Gap: {mip_gap}")
        print(f"Best Solution Encoded: {best_encoded}")
        print(f"Best Solution Raw: {best_solution}\n")

    return obj_val


def gurobi_solve(graph, file_name, time_limit, out_dir):
    nodes = len(list(graph.nodes))
    J = nx.to_numpy_array(graph)

    model = gp.Model("maxcut_qubo")

    x = model.addVars(nodes, vtype=gp.GRB.BINARY)
    objective = gp.quicksum(
        -J[i, j] * (2 * x[i] - 1) * (2 * x[j] - 1)
        for i in range(nodes)
        for j in range(i + 1, nodes)
        if J[i, j] != 0.0
    )
    model.setObjective(objective, gp.GRB.MINIMIZE)
    model.setParam("TimeLimit", time_limit)
    model.setParam("MIPGap", 0.0)

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    base_name = os.path.basename(file_name)
    model.Params.LogFile = os.path.join(out_dir, f"{base_name.split('.')[0]}.log")

    model.optimize()

    if model.Status == gp.GRB.INTERRUPTED:
        return "INTERRUPTED"

    return save_to_file(model, file_name, time_limit, out_dir)


def single_shot_instance(file_name, time_limit, out_dir):
    graph = read_nxgraph(file_name)
    return gurobi_solve(graph, file_name, time_limit, out_dir)


def multiple_shot_instance(file_names, time_limit, out_dir):
    for file_name in file_names:
        single_shot_instance(file_name, time_limit, out_dir)


def directory_shot_instance(in_dir, time_limit, out_dir, output_csv):
    if not os.path.isdir(in_dir):
        print(f"\n[!] ERROR: The input directory '{in_dir}' does not exist.")
        return

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    csv_path = os.path.join(out_dir, output_csv)

    if os.path.exists(csv_path):
        valid_rows = []
        cleaned_something = False
        with open(csv_path, mode="r", newline="") as f:
            reader = csv.reader(f)
            header = next(reader, None)
            if header:
                valid_rows.append(header)
            
            for row in reader:
                if len(row) >= 2:
                    val = str(row[1]).strip()
                    abs_path = row[0]
                    file_name = os.path.basename(abs_path)
                    log_file_name = f"{file_name.split('.')[0]}.log"
                    log_path = os.path.join(out_dir, log_file_name)

                    is_valid = True
                    is_binary = all(c in '01' for c in val) and len(val) >= 8
                    
                    if "INTERRUPTED" in val or "ERROR" in val:
                        is_valid = False
                    elif is_binary:
                        if os.path.exists(log_path):
                            with open(log_path, "r", errors="ignore") as log
