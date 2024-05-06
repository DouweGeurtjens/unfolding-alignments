import os
from settings import *
import json
import matplotlib.pyplot as plt
import re


class ModelParameters:

    def __init__(self, b: int, d: int, n: int | None) -> None:
        self.breadth = b
        self.depth = d
        self.nestedness = n

    def __hash__(self) -> int:
        if self.nestedness is not None:
            return (self.breadth, self.depth, self.nestedness)
        else:
            return (self.breadth, self.depth)


def get_parameters_from_filename(filename: str):
    split = filename.split("_")
    b = int(re.search(r"\d+", split[0]).group())
    d = int(re.search(r"\d+", split[1]).group())
    nested = re.match(r"[n]{1}\d+", split[2])
    if nested is not None:
        n = int(re.search(re.compile(r"\d+"), split[2]).group())
    else:
        n = None
    return ModelParameters(b, d, n)


# The scaling for co-sets - time should be the same regardless of model structure
def compare_co_t_plots():
    dirs = [
        CONCURRENT_RESULT_DIR,
        CONCURRENT_CONCURRENT_NESTED_RESULT_DIR,
        CONCURRENT_EXCLUSIVE_NESTED_RESULT_DIR,
        EXCLUSIVE_RESULT_DIR,
        EXCLUSIVE_CONCURRENT_NESTED_RESULT_DIR,
        EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR,
    ]
    fig_total = plt.figure()
    fig_total.suptitle("total")
    ax_total = fig_total.add_subplot()
    ax_total.set_xlabel("co")
    ax_total.set_ylabel("t")
    for dir_name in dirs:
        dir = os.listdir(dir_name)

        fig_co_t = plt.figure()
        fig_co_t.suptitle(dir_name)
        ax_co_t = fig_co_t.add_subplot()
        ax_co_t.set_xlabel("co")
        ax_co_t.set_ylabel("t")
        for f in dir:
            if f.endswith(".prof"):
                continue
            params = get_parameters_from_filename(f)
            with open(f"{dir_name}/{f}") as rf:
                contents = json.load(rf)
            avg_co = sum(contents["unf_co"]) / len(contents["unf_co"])
            avg_t = sum(contents["unf_elapsed_time"]) / len(
                contents["unf_elapsed_time"])
            ax_co_t.scatter(avg_co, avg_t)
            ax_total.scatter(avg_co, avg_t)
    plt.show()


def plot_basic(b=None, d=None):
    dir_name = CONCURRENT_RESULT_DIR
    res_dir = os.listdir(dir_name)

    fig_co_t = plt.figure()
    ax_co_t = fig_co_t.add_subplot()
    ax_co_t.set_xlabel("co")
    ax_co_t.set_ylabel("t")

    fig_q = plt.figure()
    fig_v = plt.figure()
    fig_t = plt.figure()
    if b is None and d is None:
        ax_q = fig_q.add_subplot(projection='3d')
        ax_q.set_xlabel("breadth")
        ax_q.set_ylabel("depth")
        ax_q.set_zlabel("q")

        ax_v = fig_v.add_subplot(projection='3d')
        ax_v.set_xlabel("breadth")
        ax_v.set_ylabel("depth")
        ax_v.set_zlabel("v")

        ax_t = fig_t.add_subplot(projection='3d')
        ax_t.set_xlabel("breadth")
        ax_t.set_ylabel("depth")
        ax_t.set_zlabel("t")
    if b is not None:
        ax_q = fig_q.add_subplot()
        ax_q.set_xlabel("depth")
        ax_q.set_ylabel("q")

        ax_v = fig_v.add_subplot()
        ax_v.set_xlabel("depth")
        ax_v.set_ylabel("v")

        ax_t = fig_t.add_subplot()
        ax_t.set_xlabel("depth")
        ax_t.set_ylabel("t")
    if d is not None:
        ax_q = fig_q.add_subplot()
        ax_q.set_xlabel("breadth")
        ax_q.set_ylabel("q")

        ax_v = fig_v.add_subplot()
        ax_v.set_xlabel("breadth")
        ax_v.set_ylabel("v")

        ax_t = fig_t.add_subplot()
        ax_t.set_xlabel("breadth")
        ax_t.set_ylabel("t")

    for f in res_dir:
        if f.endswith(".prof"):
            continue
        params = get_parameters_from_filename(f)
        with open(f"{dir_name}/{f}") as rf:
            contents = json.load(rf)
        avg_co = sum(contents["unf_co"]) / len(contents["unf_co"])
        avg_q = sum(contents["unf_q"]) / len(contents["unf_q"])
        avg_v = sum(contents["unf_v"]) / len(contents["unf_v"])
        avg_t = sum(contents["unf_elapsed_time"]) / len(
            contents["unf_elapsed_time"])
        ax_co_t.scatter(avg_co, avg_t)
        if b is None and d is None:
            ax_q.scatter(params.breadth, params.depth, avg_q)
            ax_v.scatter(params.breadth, params.depth, avg_v)
            ax_t.scatter(params.breadth, params.depth, avg_t)
        if b is not None:
            if params.breadth == b:
                ax_q.scatter(params.depth, avg_q)
                ax_v.scatter(params.depth, avg_v)
                ax_t.scatter(params.depth, avg_t)
        if d is not None:
            if params.depth == d:
                ax_q.scatter(params.breadth, avg_q)
                ax_v.scatter(params.breadth, avg_v)
                ax_t.scatter(params.breadth, avg_t)
    plt.show()


if __name__ == "__main__":
    # plot_basic()
    compare_co_t_plots()
