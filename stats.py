import os
from settings import *
import json
import matplotlib.pyplot as plt
import re
import pm4py


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
    nested = None
    if len(split) > 2:
        nested = re.match(r"[n]{1}\d+", split[2])
    if nested is not None:
        n = int(re.search(re.compile(r"\d+"), split[2]).group())
    else:
        n = None
    return ModelParameters(b, d, n)


def v_transitions_plots():
    dir_pairs = [(CONCURRENT_RESULT_DIR, CONCURRENT_MODEL_DIR),
                 (CONCURRENT_CONCURRENT_NESTED_RESULT_DIR,
                  CONCURRENT_CONCURRENT_NESTED_MODEL_DIR),
                 (EXCLUSIVE_RESULT_DIR, EXCLUSIVE_MODEL_DIR),
                 (EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR,
                  EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR),
                 (LOOP_RESULT_DIR, LOOP_MODEL_DIR)]
    colors = ["pink", "red", "turquoise", "blue", "green"]
    color_index = 0
    fig_no_deviations = plt.figure()
    fig_no_deviations.suptitle("no deviations")
    ax_no_deviations = fig_no_deviations.add_subplot()
    ax_no_deviations.set_xlabel("transitions")
    ax_no_deviations.set_ylabel("visited states")

    fig_start_deviations = plt.figure()
    fig_start_deviations.suptitle("start deviations")
    ax_start_deviations = fig_start_deviations.add_subplot()
    ax_start_deviations.set_xlabel("transitions")
    ax_start_deviations.set_ylabel("visited states")

    fig_halfway_deviations = plt.figure()
    fig_halfway_deviations.suptitle("halfway deviations")
    ax_halfway_deviations = fig_halfway_deviations.add_subplot()
    ax_halfway_deviations.set_xlabel("transitions")
    ax_halfway_deviations.set_ylabel("visited states")

    fig_end_deviations = plt.figure()
    fig_end_deviations.suptitle("end deviations")
    ax_end_deviations = fig_end_deviations.add_subplot()
    ax_end_deviations.set_xlabel("transitions")
    ax_end_deviations.set_ylabel("visited states")
    for res_dir, model_dir in dir_pairs:
        color = colors[color_index]
        color_index += 1
        res_files = os.listdir(res_dir)
        model_files = os.listdir(model_dir)
        for res_file in res_files:
            if res_file.endswith(".prof"):
                continue

            # Get model file that matches this result
            for model_file in model_files:
                if model_file.endswith(".xes"):
                    continue
                if res_file.split(".")[0] == model_file.split(".")[0]:
                    pt = pm4py.read_ptml(f"{model_dir}/{model_file}")
                    model_net, model_im, model_fm = pm4py.convert_to_petri_net(
                        pt)
                    transitions = len(model_net.transitions)

            with open(f"{res_dir}/{res_file}") as rf:
                contents = json.load(rf)
            v_no_deviations = [
                x["v"] for x in contents["no_deviation"]
                # if x["elapsed_time"] != -1
            ]
            v_start_deviations = [
                x["v"] for x in contents["trace_deviation_start"]
                # if x["elapsed_time"] != -1
            ]
            v_halfway_deviations = [
                x["v"] for x in contents["trace_deviation_halfway"]
                # if x["elapsed_time"] != -1
            ]
            v_end_deviations = [
                x["v"] for x in contents["trace_deviation_end"]
                # if x["elapsed_time"] != -1
            ]
            try:
                avg_v_no_deviations = sum(v_no_deviations) / len(
                    v_no_deviations)
                ax_no_deviations.scatter(transitions,
                                         avg_v_no_deviations,
                                         c=color)
            except:
                pass
            try:
                avg_v_start_deviations = sum(v_start_deviations) / len(
                    v_start_deviations)
                ax_start_deviations.scatter(transitions,
                                            avg_v_start_deviations,
                                            c=color)
            except:
                pass
            try:
                avg_v_halfway_deviations = sum(v_halfway_deviations) / len(
                    v_halfway_deviations)
                ax_halfway_deviations.scatter(transitions,
                                              avg_v_halfway_deviations,
                                              c=color)
            except:
                pass
            try:
                avg_v_end_deviations = sum(v_end_deviations) / len(
                    v_end_deviations)
                ax_end_deviations.scatter(transitions,
                                          avg_v_end_deviations,
                                          c=color)
            except:
                pass
    plt.show()


def v_trace_length_plots():
    dir_pairs = [(CONCURRENT_RESULT_DIR, CONCURRENT_MODEL_DIR),
                 (CONCURRENT_CONCURRENT_NESTED_RESULT_DIR,
                  CONCURRENT_CONCURRENT_NESTED_MODEL_DIR),
                 (EXCLUSIVE_RESULT_DIR, EXCLUSIVE_MODEL_DIR),
                 (EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR,
                  EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR),
                 (LOOP_RESULT_DIR, LOOP_MODEL_DIR)]
    colors = ["pink", "red", "turquoise", "blue", "green"]
    color_index = 0
    fig_no_deviations = plt.figure()
    fig_no_deviations.suptitle("no deviations")
    ax_no_deviations = fig_no_deviations.add_subplot()
    ax_no_deviations.set_xlabel("trace length")
    ax_no_deviations.set_ylabel("visited states")

    fig_start_deviations = plt.figure()
    fig_start_deviations.suptitle("start deviations")
    ax_start_deviations = fig_start_deviations.add_subplot()
    ax_start_deviations.set_xlabel("trace length")
    ax_start_deviations.set_ylabel("visited states")

    fig_halfway_deviations = plt.figure()
    fig_halfway_deviations.suptitle("halfway deviations")
    ax_halfway_deviations = fig_halfway_deviations.add_subplot()
    ax_halfway_deviations.set_xlabel("trace length")
    ax_halfway_deviations.set_ylabel("visited states")

    fig_end_deviations = plt.figure()
    fig_end_deviations.suptitle("end deviations")
    ax_end_deviations = fig_end_deviations.add_subplot()
    ax_end_deviations.set_xlabel("trace length")
    ax_end_deviations.set_ylabel("visited states")
    for res_dir, model_dir in dir_pairs:
        color = colors[color_index]
        color_index += 1
        res_files = os.listdir(res_dir)
        model_files = os.listdir(model_dir)
        for res_file in res_files:
            if res_file.endswith(".prof"):
                continue

            with open(f"{res_dir}/{res_file}") as rf:
                contents = json.load(rf)
            tls = set([x["trace_length"] for x in contents["no_deviation"]])
            for tl in tls:
                v_no_deviations = [
                    x["elapsed_time"] for x in contents["no_deviation"]
                    if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
                    # if x["elapsed_time"] != -1
                ]
                v_start_deviations = [
                    x["elapsed_time"]
                    for x in contents["trace_deviation_start"]
                    if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
                    # if x["elapsed_time"] != -1
                ]
                v_halfway_deviations = [
                    x["elapsed_time"]
                    for x in contents["trace_deviation_halfway"]
                    if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
                    # if x["elapsed_time"] != -1
                ]
                v_end_deviations = [
                    x["elapsed_time"] for x in contents["trace_deviation_end"]
                    if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
                    # if x["elapsed_time"] != -1
                ]
                try:
                    avg_v_no_deviations = sum(v_no_deviations) / len(
                        v_no_deviations)
                    ax_no_deviations.scatter(tl, avg_v_no_deviations, c=color)
                except:
                    pass
                try:
                    avg_v_start_deviations = sum(v_start_deviations) / len(
                        v_start_deviations)
                    ax_start_deviations.scatter(tl,
                                                avg_v_start_deviations,
                                                c=color)
                except:
                    pass
                try:
                    avg_v_halfway_deviations = sum(v_halfway_deviations) / len(
                        v_halfway_deviations)
                    ax_halfway_deviations.scatter(tl,
                                                  avg_v_halfway_deviations,
                                                  c=color)
                except:
                    pass
                try:
                    avg_v_end_deviations = sum(v_end_deviations) / len(
                        v_end_deviations)
                    ax_end_deviations.scatter(tl,
                                              avg_v_end_deviations,
                                              c=color)
                except:
                    pass
    plt.show()


# The scaling for visisted - time should be the same regardless of model structure
def compare_v_t_plots():
    dirs = [
        CONCURRENT_RESULT_DIR,
        CONCURRENT_CONCURRENT_NESTED_RESULT_DIR,
        EXCLUSIVE_RESULT_DIR,
        EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR,
        LOOP_RESULT_DIR,
    ]
    fig_total = plt.figure()
    fig_total.suptitle("total")
    ax_total = fig_total.add_subplot()
    ax_total.set_xlabel("v")
    ax_total.set_ylabel("t")
    for dir_name in dirs:
        dir = os.listdir(dir_name)

        fig_v_t = plt.figure()
        fig_v_t.suptitle(dir_name)
        ax_v_t = fig_v_t.add_subplot()
        ax_v_t.set_xlabel("v")
        ax_v_t.set_ylabel("t")
        for f in dir:
            if f.endswith(".prof"):
                continue
            params = get_parameters_from_filename(f)
            with open(f"{dir_name}/{f}") as rf:
                contents = json.load(rf)
            subcontents = contents["no_deviation"]
            v = [x["v"] for x in subcontents]
            t = [x["elapsed_time"] for x in subcontents]
            avg_v = sum(v) / len(v)
            avg_t = sum(t) / len(t)
            ax_v_t.scatter(avg_v, avg_t)
            ax_total.scatter(avg_v, avg_t)
    plt.show()


def plot_basic(b=None, d=None):
    dir_name = LOOP_RESULT_DIR
    res_dir = os.listdir(dir_name)

    fig_v_t = plt.figure()
    ax_v_t = fig_v_t.add_subplot()
    ax_v_t.set_xlabel("v")
    ax_v_t.set_ylabel("t")

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
        subcontents = contents["no_deviation"]
        q = [x["q"] for x in subcontents]
        v = [x["v"] for x in subcontents]
        t = [x["elapsed_time"] for x in subcontents]
        avg_q = sum(q) / len(q)
        avg_v = sum(v) / len(v)
        avg_t = sum(t) / len(t)
        ax_v_t.scatter(avg_v, avg_t)
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


def plot_2d(x, y, title):
    fig = plt.figure()
    fig.suptitle(title)

    ax = fig.add_subplot()
    ax.set_xlabel(x)
    ax.set_ylabel(y)

    return fig, ax


def plot_3d(x, y, z, title):
    fig = plt.figure()
    fig.suptitle(title)

    ax = fig.add_subplot(projection='3d')
    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)

    return fig, ax


def stage_one_figures(dir_names_color_title_pairs, subfolder_name):
    breadth = "breadth"
    depth = "depth"
    v = "visited states"
    t = "time (s)"
    tl = "trace length"

    # visited time plots
    fig_vt_no_deviations, ax_vt_no_deviations = plot_2d(v, t, "no deviations")
    fig_vt_start_deviations, ax_vt_start_deviations = plot_2d(
        v, t, "start deviations")
    fig_vt_halfway_deviations, ax_vt_halfway_deviations = plot_2d(
        v, t, "halfway deviations")
    fig_vt_end_deviations, ax_vt_end_deviations = plot_2d(
        v, t, "end deviations")

    # trace_length visited plots
    fig_tl_v_no_deviations, ax_tl_v_no_deviations = plot_2d(
        tl, v, "no deviations")
    fig_tl_v_start_deviations, ax_tl_v_start_deviations = plot_2d(
        tl, v, "start deviations")
    fig_tl_v_halfway_deviations, ax_tl_v_halfway_deviations = plot_2d(
        tl, v, "halfway deviations")
    fig_tl_v_end_deviations, ax_tl_v_end_deviations = plot_2d(
        tl, v, "end deviations")

    for res_dir_name, model_dir_name, color, title in dir_names_color_title_pairs:
        res_files = os.listdir(res_dir_name)
        model_files = os.listdir(model_dir_name)

        # 3d plots
        fig_v_no, ax_v_no_deviations = plot_3d(breadth, depth, v,
                                               f"{title} no deviations")
        fig_t_no, ax_t_no_deviations = plot_3d(breadth, depth, t,
                                               f"{title} no deviations")

        fig_v_start, ax_v_start_deviations = plot_3d(
            breadth, depth, v, f"{title} start deviations")
        fig_t_start, ax_t_start_deviations = plot_3d(
            breadth, depth, t, f"{title} start deviations")

        fig_v_halfway, ax_v_halfway_deviations = plot_3d(
            breadth, depth, v, f"{title} halfway deviations")
        fig_t_halfway, ax_t_halfway_deviations = plot_3d(
            breadth, depth, t, f"{title} halfway deviations")

        fig_v_end, ax_v_end_deviations = plot_3d(breadth, depth, v,
                                                 f"{title} end deviations")
        fig_t_end, ax_t_end_deviations = plot_3d(breadth, depth, t,
                                                 f"{title} end deviations")

        for res_file in res_files:
            if res_file.endswith(".prof"):
                continue

            params = get_parameters_from_filename(res_file)

            # Get model file that matches this result
            for model_file in model_files:
                if model_file.endswith(".xes"):
                    continue
                if res_file.split(".")[0] == model_file.split(".")[0]:
                    pt = pm4py.read_ptml(f"{model_dir_name}/{model_file}")
                    model_net, model_im, model_fm = pm4py.convert_to_petri_net(
                        pt)
                    transitions = len(model_net.transitions)

            with open(f"{res_dir_name}/{res_file}") as rf:
                contents = json.load(rf)

            v_no_deviations = [
                x["v"] for x in contents["no_deviation"]
                if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
            ]
            v_start_deviations = [
                x["v"] for x in contents["trace_deviation_start"]
                if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
            ]
            v_halfway_deviations = [
                x["v"] for x in contents["trace_deviation_halfway"]
                if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
            ]
            v_end_deviations = [
                x["v"] for x in contents["trace_deviation_end"]
                if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
            ]

            t_no_deviations = [
                x["elapsed_time"] for x in contents["no_deviation"]
                if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
            ]
            t_start_deviations = [
                x["elapsed_time"] for x in contents["trace_deviation_start"]
                if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
            ]
            t_halfway_deviations = [
                x["elapsed_time"] for x in contents["trace_deviation_halfway"]
                if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
            ]
            t_end_deviations = [
                x["elapsed_time"] for x in contents["trace_deviation_end"]
                if (x["elapsed_time"] != -1 and x["elapsed_time"] < 100)
            ]
            try:
                avg_v_no_deviations = sum(v_no_deviations) / len(
                    v_no_deviations)
                avg_t_no_deviations = sum(t_no_deviations) / len(
                    t_no_deviations)

                ax_vt_no_deviations.scatter(avg_v_no_deviations,
                                            avg_t_no_deviations,
                                            c=color)

                ax_v_no_deviations.scatter(params.breadth,
                                           params.depth,
                                           avg_v_no_deviations,
                                           c=color)
                ax_t_no_deviations.scatter(params.breadth,
                                           params.depth,
                                           avg_t_no_deviations,
                                           c=color)
            except ZeroDivisionError:
                pass

            try:
                avg_v_start_deviations = sum(v_start_deviations) / len(
                    v_start_deviations)
                avg_t_start_deviations = sum(t_start_deviations) / len(
                    t_start_deviations)

                ax_vt_start_deviations.scatter(avg_v_start_deviations,
                                               avg_t_start_deviations,
                                               c=color)

                ax_v_start_deviations.scatter(params.breadth,
                                              params.depth,
                                              avg_v_start_deviations,
                                              c=color)
                ax_t_start_deviations.scatter(params.breadth,
                                              params.depth,
                                              avg_t_start_deviations,
                                              c=color)
            except ZeroDivisionError:
                pass

            try:
                avg_v_halfway_deviations = sum(v_halfway_deviations) / len(
                    v_halfway_deviations)
                avg_t_halfway_deviations = sum(t_halfway_deviations) / len(
                    t_halfway_deviations)

                ax_vt_halfway_deviations.scatter(avg_v_halfway_deviations,
                                                 avg_t_halfway_deviations,
                                                 c=color)

                ax_v_halfway_deviations.scatter(params.breadth,
                                                params.depth,
                                                avg_v_halfway_deviations,
                                                c=color)
                ax_t_halfway_deviations.scatter(params.breadth,
                                                params.depth,
                                                avg_t_halfway_deviations,
                                                c=color)
            except ZeroDivisionError:
                pass

            try:
                avg_v_end_deviations = sum(v_end_deviations) / len(
                    v_end_deviations)
                avg_t_end_deviations = sum(t_end_deviations) / len(
                    t_end_deviations)

                ax_vt_end_deviations.scatter(avg_v_end_deviations,
                                             avg_t_end_deviations,
                                             c=color)

                ax_v_end_deviations.scatter(params.breadth,
                                            params.depth,
                                            avg_v_end_deviations,
                                            c=color)
                ax_t_end_deviations.scatter(params.breadth,
                                            params.depth,
                                            avg_t_end_deviations,
                                            c=color)
            except ZeroDivisionError:
                pass

            tls = set([x["trace_length"] for x in contents["no_deviation"]])
            for tl in tls:
                v_no_deviations_tls = [
                    x["v"] for x in contents["no_deviation"]
                    if (x["trace_length"] == tl and x["elapsed_time"] != -1
                        and x["elapsed_time"] < 100)
                ]
                v_start_deviations_tls = [
                    x["v"] for x in contents["trace_deviation_start"]
                    if (x["trace_length"] == tl and x["elapsed_time"] != -1
                        and x["elapsed_time"] < 100)
                ]
                v_halfway_deviations_tls = [
                    x["v"] for x in contents["trace_deviation_halfway"]
                    if (x["trace_length"] == tl and x["elapsed_time"] != -1
                        and x["elapsed_time"] < 100)
                ]
                v_end_deviations_tls = [
                    x["v"] for x in contents["trace_deviation_end"]
                    if (x["trace_length"] == tl and x["elapsed_time"] != -1
                        and x["elapsed_time"] < 100)
                ]
                try:
                    avg_v_no_deviations_tls = sum(v_no_deviations_tls) / len(
                        v_no_deviations_tls)
                    ax_tl_v_no_deviations.scatter(tl,
                                                  avg_v_no_deviations_tls,
                                                  c=color)
                except ZeroDivisionError:
                    pass
                try:
                    avg_v_start_deviations_tls = sum(
                        v_start_deviations_tls) / len(v_start_deviations_tls)
                    ax_tl_v_start_deviations.scatter(
                        tl, avg_v_start_deviations_tls, c=color)
                except ZeroDivisionError:
                    pass
                try:
                    avg_v_halfway_deviations_tls = sum(
                        v_halfway_deviations_tls) / len(
                            v_halfway_deviations_tls)
                    ax_tl_v_halfway_deviations.scatter(
                        tl, avg_v_halfway_deviations_tls, c=color)
                except ZeroDivisionError:
                    pass
                try:
                    avg_v_end_deviations_tls = sum(v_end_deviations_tls) / len(
                        v_end_deviations_tls)
                    ax_tl_v_end_deviations.scatter(tl,
                                                   avg_v_end_deviations_tls,
                                                   c=color)
                except ZeroDivisionError:
                    pass
        fig_v_no.savefig(f"figures/stage_one/{subfolder_name}/{title}_3d_v_no")
        fig_t_no.savefig(f"figures/stage_one/{subfolder_name}/{title}_3d_t_no")
        fig_v_start.savefig(
            f"figures/stage_one/{subfolder_name}/{title}_3d_v_start")
        fig_t_start.savefig(
            f"figures/stage_one/{subfolder_name}/{title}_3d_t_start")
        fig_v_halfway.savefig(
            f"figures/stage_one/{subfolder_name}/{title}_3d_v_halfway")
        fig_t_halfway.savefig(
            f"figures/stage_one/{subfolder_name}/{title}_3d_t_halfway")
        fig_v_end.savefig(
            f"figures/stage_one/{subfolder_name}/{title}_3d_v_end")
        fig_t_end.savefig(
            f"figures/stage_one/{subfolder_name}/{title}_3d_t_end")

    fig_vt_no_deviations.savefig(f"figures/stage_one/{subfolder_name}/v_t_no")
    fig_vt_start_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/v_t_start")
    fig_vt_halfway_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/v_t_halfway")
    fig_vt_end_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/v_t_end")

    fig_tl_v_no_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/tl_v_no")
    fig_tl_v_start_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/tl_v_start")
    fig_tl_v_halfway_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/tl_v_halfway")
    fig_tl_v_end_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/tl_v_end")
    # plt.show()


def stage_two_figures(res_file_path_color_title_pairs, subfolder_name):
    v = "visited states"
    c = "count"
    t = "time (s)"
    tl = "trace length"
    unf = "unfolding"
    astar = "astar"
    dijkstra = "dijkstra"

    fig_unf_v_t, ax_unf_v_t = plot_2d(v, t, unf)
    fig_unf_tl_v, ax_unf_tl_v = plot_2d(v, "trace length", unf)
    for res_file_path, color, title in res_file_path_color_title_pairs:
        with open(res_file_path) as rf:
            contents = json.load(rf)

        fig_unf_t_bar, ax_unf_t_bar = plot_2d(t, c, f"{title} {unf}")
        fig_astar_t_bar, ax_astar_t_bar = plot_2d(t, c, f"{title} {astar}")
        fig_dijkstra_t_bar, ax_dijkstra_t_bar = plot_2d(
            t, c, f"{title} {dijkstra}")

        unf_elapsed_time_negative = [x["unf_elapsed_time"] for x in contents]
        unf_tl = [
            x["trace_length"] for x in contents if x["unf_elapsed_time"] != -1
        ]

        unf_elapsed_time = [
            x["unf_elapsed_time"] for x in contents
            if x["unf_elapsed_time"] != -1
        ]
        unf_v = [x["unf_v"] for x in contents if x["unf_elapsed_time"] != -1]

        astar_elapsed_time = [
            x["astar_elapsed_time"] for x in contents
            # if x["astar_elapsed_time"] != -1
        ]
        dijkstra_elapsed_time = [
            x["dijkstra_elapsed_time"] for x in contents
            # if x["dijkstra_elapsed_time"] != -1
        ]

        n, _, patches = ax_unf_t_bar.hist(unf_elapsed_time_negative, 102,
                                          (-1, 100))
        if n[0] > 0:
            patches[0].set_facecolor("red")
        n, _, patches = ax_astar_t_bar.hist(astar_elapsed_time, 102, (-1, 100))
        if n[0] > 0:
            patches[0].set_facecolor("red")
        n, _, patches = ax_dijkstra_t_bar.hist(dijkstra_elapsed_time, 102,
                                               (-1, 100))
        if n[0] > 0:
            patches[0].set_facecolor("red")

        fig_unf_t_bar.savefig(f"figures/stage_two/{title}_{unf}_t_bar")
        fig_astar_t_bar.savefig(f"figures/stage_two/{title}_{astar}_t_bar")
        fig_dijkstra_t_bar.savefig(
            f"figures/stage_two/{title}_{dijkstra}_t_bar")

        ax_unf_v_t.scatter(unf_v, unf_elapsed_time, c=color)
        ax_unf_tl_v.scatter(unf_tl, unf_v, c=color)

    fig_unf_v_t.savefig(f"figures/stage_two/{unf}_v_t")
    fig_unf_tl_v.savefig(f"figures/stage_two/{unf}_tl_v")

    # plt.show()


if __name__ == "__main__":
    # # No heuristics stage one
    # dir_name_color_title_pairs_no_heuristic = [
    #     (CONCURRENT_RESULT_DIR_NO_HEURISTIC, CONCURRENT_MODEL_DIR, "pink",
    #      "C"),
    #     (CONCURRENT_CONCURRENT_NESTED_RESULT_DIR_NO_HEURISTIC,
    #      CONCURRENT_CONCURRENT_NESTED_MODEL_DIR, "red", "CN"),
    #     (EXCLUSIVE_RESULT_DIR_NO_HEURISTIC, EXCLUSIVE_MODEL_DIR, "turquoise",
    #      "E"),
    #     (EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR_NO_HEURISTIC,
    #      EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR, "blue", "EN"),
    #     (LOOP_RESULT_DIR_NO_HEURISTIC, LOOP_MODEL_DIR, "green", "L")
    # ]
    # stage_one_figures(dir_name_color_title_pairs_no_heuristic, "no_heuristic")

    # # Heuristics stage one
    # dir_name_color_title_pairs_heuristic = [
    #     (CONCURRENT_RESULT_DIR, CONCURRENT_MODEL_DIR, "pink", "C"),
    #     (CONCURRENT_CONCURRENT_NESTED_RESULT_DIR,
    #      CONCURRENT_CONCURRENT_NESTED_MODEL_DIR, "red", "CN"),
    #     (EXCLUSIVE_RESULT_DIR, EXCLUSIVE_MODEL_DIR, "turquoise", "E"),
    #     (EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR,
    #      EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR, "blue", "EN"),
    #     (LOOP_RESULT_DIR, LOOP_MODEL_DIR, "green", "L")
    # ]
    # stage_one_figures(dir_name_color_title_pairs_heuristic, "heuristic")

    res_file_path_color_title_pairs = [
        ("results/sepsis/sepsis_05.json", "red", "SP"),
        ("results/bpic17/bpic17_02.json", "green", "BPIC17"),
        ("results/bpic19/bpic19_02.json", "blue", "BPIC19"),
        ("results/inthelarge/prAm6.json", "pink", "ITL prAm6"),
        ("results/inthelarge/prBm6.json", "purple", "ITL prBm6"),
        ("results/inthelarge/prCm6.json", "magenta", "ITL prCm6"),
        ("results/inthelarge/prDm6.json", "turquoise", "ITL prDm6"),
        ("results/inthelarge/prEm6.json", "cyan", "ITL prEm6"),
        ("results/inthelarge/prFm6.json", "yellow", "ITL prFm6"),
        ("results/inthelarge/prGm6.json", "orange", "ITL prGm6"),
    ]
    stage_two_figures(res_file_path_color_title_pairs, None)
    # stage_two_figures("results/inthelarge/prFm6.json")
    # v_trace_length_plots()
