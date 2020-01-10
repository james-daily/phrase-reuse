import argparse
import glob
import subprocess

import pandas as pd


def spawn_process(i, num_chunks):
    print(f"starting analysis for chunk {i} of {num_chunks}")

    cmd = [
        "python", "-u", "analyze.py",
        "--chunks", str(num_chunks), "--chunk_num", str(i)
    ]

    cmd = " ".join(cmd)

    procs = []

    # start subprocesses
    with open(f"analysis_{str(i).zfill(2)}_of_{num_chunks}.out", "w") as f:
        print('executing "{}"'.format(cmd))
        return subprocess.Popen(cmd, stdout=f, stderr=f, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("num_processes", type=int)
    args = parser.parse_args()

    procs = [spawn_process(i, args.num_processes) for i in range(0, args.num_processes)]

    for p in procs:
        print(f"waiting on {p.args}")
        p.wait()
        print(f"{p.args} complete with return code {p.returncode}")

    # combine output
    filenames = sorted(glob.glob("data/antecedent_counts_*.csv"))
    df = pd.concat([pd.read_csv(f) for f in filenames])

    df.to_csv("data/antecedent_counts.csv", index=False)


if __name__ == '__main__':
    main()
