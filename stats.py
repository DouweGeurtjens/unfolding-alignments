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


def preliminary(filepath):
    with open(filepath, "r") as f:
        res = json.load(f)

    table_data = {}

    fig_c_v = plt.figure()
    fig_c_v.suptitle("unfolding")
    ax_c_v = fig_c_v.add_subplot()
    ax_c_v.set_xlabel("cost")
    ax_c_v.set_ylabel("visisted states")

    fig_c_t = plt.figure()
    fig_c_t.suptitle("unfolding")
    ax_c_t = fig_c_t.add_subplot()
    ax_c_t.set_xlabel("cost")
    ax_c_t.set_ylabel("elapsed time (s)")

    fig_q_t = plt.figure()
    fig_q_t.suptitle("unfolding")
    ax_q_t = fig_q_t.add_subplot()
    ax_q_t.set_xlabel("queued states")
    ax_q_t.set_ylabel("elapsed time (s)")

    fig_v_t = plt.figure()
    fig_v_t.suptitle("unfolding")
    ax_v_t = fig_v_t.add_subplot()
    ax_v_t.set_xlabel("visited states")
    ax_v_t.set_ylabel("elapsed time (s)")

    fig_unf_time_box = plt.figure()
    fig_unf_time_box.suptitle("unfolding (no outliers)")
    unf_time_box = fig_unf_time_box.add_subplot()
    unf_time_box.set_ylabel("elapsed time (s)")

    fig_unf_time_fliers = plt.figure()
    fig_unf_time_fliers.suptitle("unfolding (only outliers)")
    unf_time_fliers = fig_unf_time_fliers.add_subplot()
    unf_time_fliers.set_ylabel("elapsed time (s)")

    fig_dijkstra_time_box = plt.figure()
    fig_dijkstra_time_box.suptitle("dijkstra (no outliers)")
    dijkstra_time_box = fig_dijkstra_time_box.add_subplot()
    dijkstra_time_box.set_ylabel("elapsed time (s)")

    fig_dijkstra_time_fliers = plt.figure()
    fig_dijkstra_time_fliers.suptitle("dijkstra (only outliers)")
    dijkstra_time_fliers = fig_dijkstra_time_fliers.add_subplot()
    dijkstra_time_fliers.set_ylabel("elapsed time (s)")

    fig_astar_time_box = plt.figure()
    fig_astar_time_box.suptitle("astar (no outliers)")
    astar_time_box = fig_astar_time_box.add_subplot()
    astar_time_box.set_ylabel("elapsed time (s)")

    fig_astar_time_fliers = plt.figure()
    fig_astar_time_fliers.suptitle("astar (only outliers)")
    astar_time_fliers = fig_astar_time_fliers.add_subplot()
    astar_time_fliers.set_ylabel("elapsed time (s)")

    unf_elapsed_time = [x["unf_elapsed_time"] for x in res]
    unf_q = [x["unf_q"] for x in res]
    unf_v = [x["unf_v"] for x in res]
    unf_cost = [x["unf_cost"] for x in res]

    dijkstra_elapsed_time = [x["dijkstra_elapsed_time"] for x in res]
    dijkstra_q = [x["dijkstra_q"] for x in res]
    dijkstra_v = [x["dijkstra_v"] for x in res]
    dijkstra_cost = [x["dijkstra_cost"] for x in res]

    astar_elapsed_time = [x["astar_elapsed_time"] for x in res]
    astar_q = [x["astar_q"] for x in res]
    astar_v = [x["astar_v"] for x in res]
    astar_cost = [x["astar_cost"] for x in res]

    unf_time_box.boxplot(unf_elapsed_time, showfliers=False)
    unf_time_fliers.boxplot(unf_elapsed_time,
                            meanline=False,
                            showbox=False,
                            showcaps=False,
                            showmeans=False)
    dijkstra_time_box.boxplot(dijkstra_elapsed_time, showfliers=False)
    dijkstra_time_fliers.boxplot(dijkstra_elapsed_time,
                                 meanline=False,
                                 showbox=False,
                                 showcaps=False,
                                 showmeans=False)
    astar_time_box.boxplot(astar_elapsed_time, showfliers=False)
    astar_time_fliers.boxplot(astar_elapsed_time,
                              meanline=False,
                              showbox=False,
                              showcaps=False,
                              showmeans=False)

    ax_q_t.scatter(unf_q, unf_elapsed_time)
    ax_v_t.scatter(unf_v, unf_elapsed_time)
    ax_c_v.scatter(unf_cost, unf_v)
    ax_c_t.scatter(unf_cost, unf_elapsed_time)

    table_data["avg_q_difference_unf_dijkstra"] = sum([(
        (j - i) / i) * 100 for i, j in zip(unf_q, dijkstra_q)]) / len(unf_q)
    table_data["avg_q_difference_unf_astar"] = sum([(
        (j - i) / i) * 100 for i, j in zip(unf_q, astar_q)]) / len(unf_q)

    table_data["avg_v_difference_unf_dijkstra"] = sum([(
        (j - i) / i) * 100 for i, j in zip(unf_v, dijkstra_v)]) / len(unf_v)
    table_data["avg_v_difference_unf_astar"] = sum([(
        (j - i) / i) * 100 for i, j in zip(unf_v, astar_v)]) / len(unf_v)
    print(table_data)
    plt.show()


if __name__ == "__main__":
    # plot_basic()
    # compare_co_t_plots()
    preliminary("preliminary_sepsis_0_noise.json")
    preliminary("preliminary_sepsis_05_noise.json")
