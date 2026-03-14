import networkx as nx
import numpy as np
import os
import math
import gurobipy as gp
import argparse
import csv
from utils import read_nxgraph
from utils import float_to_binary
from utils import base64_encode


def save_to_file(model, file_path, time_limit, print_terminal=True):
    file_dir = f"results/gurobi/{'/'.join(file_path.split('/')[:-1])}"
    file_name = file_path.split("/")[-1]

    obj_val = model.ObjVal
    obj_bnd = model.ObjBound
    duration = model.Runtime
    mip_gap = model.MIPGap
    best_solution = "".join([float_to_binary(x.X) for x in model.getVars()])
    best_encoded = base64_encode(best_solution)

    if not os.path.exists(file_dir):
        os.makedirs(file_dir)

    with open(f"{file_dir}/{file_name}", "w") as f:
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

    return best_solution


def gurobi_solve(graph, file_path, time_limit=3600):
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

    file_dir = f"results/gurobi/{'/'.join(file_path.split('/')[:-1])}"
    file_name = file_path.split("/")[-1]
    if not os.path.exists(file_dir):
        os.makedirs(file_dir)
    model.Params.LogFile = f"{file_dir}/{file_name.split('.')[0]}.log"

    model.optimize()

    return save_to_file(model, file_path, time_limit)


def single_shot_instance(file_name, time_limit):
    graph = read_nxgraph(file_name)
    # TODO : change for file path of various lengths
    file_path = file_name[5:]

    # Return the result up the chain
    return gurobi_solve(graph, file_path, time_limit)


def multiple_shot_instance(file_names, time_limit):
    for file_name in file_names:
        single_shot_instance(file_name, time_limit)


def directory_shot_instance(
    folder_path, time_limit, output_csv="directory_results.csv"
):
    results = []

    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_name = os.path.join(root, file)
            abs_path = os.path.abspath(file_name)

            print(f"\n--- Processing: {file_name} ---")
            try:
                best_solution = single_shot_instance(file_name, time_limit)
                results.append([abs_path, best_solution])
            except Exception as e:
                # Error handling so one bad file doesn't crash the whole batch
                print(f"Skipping {file_name} due to error: {e}")
                results.append([abs_path, f"ERROR: {e}"])

    # Write everything to CSV
    with open(output_csv, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Absolute Path", "Best Solution Raw"])
        writer.writerows(results)

    print(f"\nBatch processing complete. Results saved to '{output_csv}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Gurobi MaxCut QUBO solver on graphs."
    )

    parser.add_argument(
        "paths",
        nargs="*",
        default=["../../src/data/1D/VNA/Chain/32/ising_chain_32_seed1.txt"],
        help="One or more file paths to process. Defaults to a specific test file if omitted.",
    )

    parser.add_argument(
        "-t",
        "--time_limit",
        type=int,
        default=3600,
        help="Time limit for the solver in seconds (default: 3600).",
    )

    parser.add_argument(
        "-d",
        "--directory",
        type=str,
        help="Process all files in this directory and output results to a CSV.",
    )

    parser.add_argument(
        "--csv_out",
        type=str,
        default="gurobi_batch_results.csv",
        help="Name of the output CSV file when using the directory flag (default: batch_results.csv).",
    )

    args = parser.parse_args()

    if args.directory:
        directory_shot_instance(args.directory, args.time_limit, args.csv_out)
    elif len(args.paths) == 1:
        single_shot_instance(args.paths[0], args.time_limit)
    elif len(args.paths) > 1:
        # Multiple specific files
        multiple_shot_instance(args.paths, args.time_limit)
    else:
        print("No paths or directories were provided.")
