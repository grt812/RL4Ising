import networkx as nx
import os
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
    csv_data = {}

    if os.path.exists(csv_path):
        with open(csv_path, mode="r", newline="") as f:
            reader = csv.reader(f)
            next(reader, None)
            for row in reader:
                if len(row) >= 2:
                    csv_data[row[0]] = str(row[1]).strip()

    synced_something = False

    for root, dirs, files in os.walk(in_dir):
        for file in files:
            if not file.endswith(".txt") or file == output_csv:
                continue

            abs_path = os.path.abspath(os.path.join(root, file))
            log_file_name = f"{file.split('.')[0]}.log"
            log_path = os.path.join(out_dir, log_file_name)

            if abs_path in csv_data:
                val = csv_data[abs_path]
                is_binary = all(c in "01" for c in val) and len(val) >= 8

                # Skip license limit
                if "LICENSE_LIMIT" in val or "size-limited" in val.lower():
                    csv_data[abs_path] = "LICENSE_LIMIT"

                elif "INTERRUPTED" in val or "ERROR" in val:
                    del csv_data[abs_path]
                    synced_something = True

                elif is_binary:
                    if os.path.exists(log_path):
                        with open(log_path, "r", errors="ignore") as log_f:
                            log_content = log_f.read()
                            if (
                                "size-limited" in log_content.lower()
                                or "too large" in log_content.lower()
                            ):
                                csv_data[abs_path] = "LICENSE_LIMIT"
                                synced_something = True
                            elif "Solve interrupted" in log_content:
                                del csv_data[abs_path]
                                synced_something = True
                            else:
                                match = re.search(
                                    r"Objective Value:\s*([-\d\.]+)", log_content
                                )
                                if match:
                                    csv_data[abs_path] = match.group(1)
                                    synced_something = True
                                    print(
                                        f"[*] Repaired binary entry for {file} from log."
                                    )
                                else:
                                    del csv_data[abs_path]
                                    synced_something = True
                    else:
                        del csv_data[abs_path]
                        synced_something = True
                else:
                    if os.path.exists(log_path):
                        with open(log_path, "r", errors="ignore") as log_f:
                            if "Solve interrupted" in log_f.read():
                                del csv_data[abs_path]
                                synced_something = True

            else:
                if os.path.exists(log_path):
                    with open(log_path, "r", errors="ignore") as log_f:
                        log_content = log_f.read()

                        if (
                            "size-limited" in log_content.lower()
                            or "too large" in log_content.lower()
                        ):
                            csv_data[abs_path] = "LICENSE_LIMIT"
                            synced_something = True
                            print(
                                f"[*] Marked size-limited license error for {file}. Will not retry."
                            )
                        elif "Solve interrupted" in log_content:
                            print(
                                f"[*] Removing orphaned interrupted log for {file}..."
                            )
                            os.remove(log_path)
                            txt_out_path = os.path.join(out_dir, file)
                            if os.path.exists(txt_out_path):
                                os.remove(txt_out_path)
                        else:
                            match = re.search(
                                r"Objective Value:\s*([-\d\.]+)", log_content
                            )
                            if match:
                                csv_data[abs_path] = match.group(1)
                                synced_something = True
                                print(
                                    f"[*] Recovered missing objective {csv_data[abs_path]} for {file} from log."
                                )

    if synced_something or not os.path.exists(csv_path):
        if synced_something:
            print("[*] Synchronization complete: CSV is now up-to-date with log files.")
        with open(csv_path, mode="w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Absolute Path", "Objective Value"])
            for p, v in csv_data.items():
                writer.writerow([p, v])

    with open(csv_path, mode="a", newline="") as f:
        writer = csv.writer(f)

        for root, dirs, files in os.walk(in_dir):
            for file in files:
                if not file.endswith(".txt") or file == output_csv:
                    continue

                abs_path = os.path.abspath(os.path.join(root, file))

                if abs_path in csv_data:
                    if csv_data[abs_path] != "LICENSE_LIMIT":
                        print(f"Skipping already completed file: {file}")
                    continue

                print(f"\nProcessing: {file}")
                try:
                    obj_val = single_shot_instance(
                        file_name=os.path.join(root, file),
                        time_limit=time_limit,
                        out_dir=out_dir,
                    )

                    if obj_val == "INTERRUPTED":
                        print("\n[!] Run stopped by user. Halting batch safely.")
                        return

                    writer.writerow([abs_path, obj_val])
                    csv_data[abs_path] = obj_val
                except KeyboardInterrupt:
                    print("\n[!] Run stopped by user. Halting batch safely.")
                    return
                except Exception as e:
                    error_str = str(e).lower()
                    if "size-limited" in error_str or "too large" in error_str:
                        print(f"Skipped {file} permanently due to license size limit.")
                        writer.writerow([abs_path, "LICENSE_LIMIT"])
                        csv_data[abs_path] = "LICENSE_LIMIT"
                    else:
                        print(f"Failed {file} due to error: {e}")
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
