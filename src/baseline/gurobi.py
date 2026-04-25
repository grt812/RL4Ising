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

    # --- CSV SCRUBBING PHASE ---
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
                    val = str(row[1])
                    abs_path = row[0]
                    file_name = os.path.basename(abs_path)
                    log_file_name = f"{file_name.split('.')[0]}.log"
                    log_path = os.path.join(out_dir, log_file_name)

                    is_valid = True

                    if "INTERRUPTED" in val or "ERROR" in val:
                        is_valid = False
                    elif os.path.exists(log_path):
                        with open(log_path, "r", errors="ignore") as log_f:
                            if "Solve interrupted" in log_f.read():
                                is_valid = False

                    if is_valid:
                        valid_rows.append(row)
                    else:
                        cleaned_something = True

        if cleaned_something:
            print(
                f"\n[*] Scrubbing existing CSV: Removed interrupted or incomplete runs."
            )
            with open(csv_path, mode="w", newline="") as f:
                writer = csv.writer(f)
                writer.writerows(valid_rows)
    # ---------------------------

    file_exists = os.path.exists(csv_path)

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        if not file_exists:
            writer.writerow(["Absolute Path", "Objective Value"])

        for root, dirs, files in os.walk(in_dir):
            for file in files:
                if not file.endswith(".txt"):
                    continue

                if file == output_csv:
                    continue

                file_name = os.path.join(root, file)
                abs_path = os.path.abspath(file_name)

                log_file_name = f"{file.split('.')[0]}.log"
                expected_log_path = os.path.join(out_dir, log_file_name)

                if os.path.exists(expected_log_path):
                    with open(expected_log_path, "r", errors="ignore") as log_f:
                        log_content = log_f.read()

                    if "Solve interrupted" in log_content:
                        print(
                            f"Removing interrupted log for {file_name} and restarting..."
                        )
                        os.remove(expected_log_path)
                        txt_out_path = os.path.join(out_dir, file)
                        if os.path.exists(txt_out_path):
                            os.remove(txt_out_path)
                    else:
                        print(f"Skipping already processed file: {file_name}")
                        continue

                print(f"\nProcessing: {file_name}")
                try:
                    obj_val = single_shot_instance(file_name, time_limit, out_dir)

                    if obj_val == "INTERRUPTED":
                        print("\n[!] Run stopped by user. Halting batch safely.")
                        break

                    writer.writerow([abs_path, obj_val])
                except KeyboardInterrupt:
                    print("\n[!] Run stopped by user. Halting batch safely.")
                    break
                except Exception as e:
                    print(f"Failed {file_name} due to error: {e}")
                    writer.writerow([abs_path, f"ERROR: {e}"])

                f.flush()

    print(f"\nBatch processing complete. Results saved to '{csv_path}'.")


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
        "-i",
        "--in_dir",
        type=str,
        help="Input directory. Process all .txt files in this folder.",
    )

    parser.add_argument(
        "-o",
        "--out_dir",
        type=str,
        default="results/gurobi/newest",
        help="Output directory for logs, individual text files, and the CSV.",
    )

    parser.add_argument(
        "--csv_out",
        type=str,
        default="gurobi_batch_results.csv",
        help="Name of the output CSV file (saved inside out_dir).",
    )

    args = parser.parse_args()

    if args.in_dir:
        directory_shot_instance(
            args.in_dir, args.time_limit, args.out_dir, args.csv_out
        )
    elif len(args.paths) == 1:
        single_shot_instance(args.paths[0], args.time_limit, args.out_dir)
    elif len(args.paths) > 1:
        multiple_shot_instance(args.paths, args.time_limit, args.out_dir)
    else:
        print("No paths or directories were provided.")
