import os

import sklearn.decomposition
import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.tests
import statsmodels.api as sm
from settings import *
import json
import matplotlib.pyplot as plt
import re
import pm4py
import petrinet as ptn
import sklearn
import pandas as pd
from scipy import stats
import numpy as np

from sklearn.base import BaseEstimator, RegressorMixin


class SMWrapper(BaseEstimator, RegressorMixin):
    """ A universal sklearn-style wrapper for statsmodels regressors """

    def __init__(self, model_class, fit_intercept=True):
        self.model_class = model_class
        self.fit_intercept = fit_intercept

    def fit(self, X, y):
        if self.fit_intercept:
            X = sm.add_constant(X)
        self.model_ = self.model_class(y, X)
        self.results_ = self.model_.fit()
        return self

    def predict(self, X):
        if self.fit_intercept:
            X = sm.add_constant(X)
        return self.results_.predict(X)


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
    sp = "synchronous product transitions"

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

    # sync_prod visisted plots
    fig_sp_v_no_deviations, ax_sp_v_no_deviations = plot_2d(
        sp, v, "no deviations")
    fig_sp_v_start_deviations, ax_sp_v_start_deviations = plot_2d(
        sp, v, "start deviations")
    fig_sp_v_halfway_deviations, ax_sp_v_halfway_deviations = plot_2d(
        sp, v, "halfway deviations")
    fig_sp_v_end_deviations, ax_sp_v_end_deviations = plot_2d(
        sp, v, "end deviations")

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
            # Use avg cause loop
            tls = [x["trace_length"] for x in contents["no_deviation"]]
            avg_trace_length = sum(tls) / len(tls)

            sync_prod_size_no_deviation = transitions + (2 * avg_trace_length)
            sync_prod_size_with_deviation = sync_prod_size_no_deviation - 2

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
                ax_tl_v_no_deviations.scatter(avg_trace_length,
                                              avg_v_no_deviations,
                                              c=color)
                ax_sp_v_no_deviations.scatter(sync_prod_size_no_deviation,
                                              avg_v_no_deviations,
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
                ax_tl_v_start_deviations.scatter(avg_trace_length,
                                                 avg_v_start_deviations,
                                                 c=color)
                ax_sp_v_start_deviations.scatter(sync_prod_size_with_deviation,
                                                 avg_v_start_deviations,
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
                ax_tl_v_halfway_deviations.scatter(avg_trace_length,
                                                   avg_v_halfway_deviations,
                                                   c=color)
                ax_sp_v_halfway_deviations.scatter(
                    sync_prod_size_with_deviation,
                    avg_v_halfway_deviations,
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
                ax_tl_v_end_deviations.scatter(avg_trace_length,
                                               avg_v_end_deviations,
                                               c=color)
                ax_sp_v_end_deviations.scatter(sync_prod_size_with_deviation,
                                               avg_v_end_deviations,
                                               c=color)
            except ZeroDivisionError:
                pass
        # fig_v_no.savefig(f"figures/stage_one/{subfolder_name}/{title}_3d_v_no")
        # fig_t_no.savefig(f"figures/stage_one/{subfolder_name}/{title}_3d_t_no")
        # fig_v_start.savefig(
        #     f"figures/stage_one/{subfolder_name}/{title}_3d_v_start")
        # fig_t_start.savefig(
        #     f"figures/stage_one/{subfolder_name}/{title}_3d_t_start")
        # fig_v_halfway.savefig(
        #     f"figures/stage_one/{subfolder_name}/{title}_3d_v_halfway")
        # fig_t_halfway.savefig(
        #     f"figures/stage_one/{subfolder_name}/{title}_3d_t_halfway")
        # fig_v_end.savefig(
        #     f"figures/stage_one/{subfolder_name}/{title}_3d_v_end")
        # fig_t_end.savefig(
        #     f"figures/stage_one/{subfolder_name}/{title}_3d_t_end")

    # fig_vt_no_deviations.savefig(f"figures/stage_one/{subfolder_name}/v_t_no")
    # fig_vt_start_deviations.savefig(
    #     f"figures/stage_one/{subfolder_name}/v_t_start")
    # fig_vt_halfway_deviations.savefig(
    #     f"figures/stage_one/{subfolder_name}/v_t_halfway")
    # fig_vt_end_deviations.savefig(
    #     f"figures/stage_one/{subfolder_name}/v_t_end")

    fig_tl_v_no_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/tl_v_no")
    fig_tl_v_start_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/tl_v_start")
    fig_tl_v_halfway_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/tl_v_halfway")
    fig_tl_v_end_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/tl_v_end")

    fig_sp_v_no_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/sp_v_no")
    fig_sp_v_start_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/sp_v_start")
    fig_sp_v_halfway_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/sp_v_halfway")
    fig_sp_v_end_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/sp_v_end")
    # plt.show()


def stage_two_figures():
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
    v = "visited states"
    c = "count"
    t = "time (s)"
    tlog = "time (log2(s))"
    tl = "trace length"
    sp = "synchronous product transitions"
    unf = "unfolding"
    astar = "astar"
    dijkstra = "dijkstra"

    fig_unf_v_t, ax_unf_v_t = plot_2d(v, t, unf)
    fig_unf_tl_v, ax_unf_tl_v = plot_2d(tl, v, unf)
    fig_unf_sp_v, ax_unf_sp_v = plot_2d(sp, v, unf)
    fig_unf_sp_t, ax_unf_sp_t = plot_2d(sp, t, unf)
    fig_unf_v_tlog, ax_unf_v_tlog = plot_2d(v, tlog, unf)
    fig_unf_sp_tlog, ax_unf_sp_tlog = plot_2d(sp, tlog, unf)

    for res_file_path, color, title in res_file_path_color_title_pairs:
        with open(res_file_path) as rf:
            contents = json.load(rf)

        fig_unf_t_bar, ax_unf_t_bar = plot_2d(t, c, f"{title} {unf}")
        fig_astar_t_bar, ax_astar_t_bar = plot_2d(t, c, f"{title} {astar}")
        fig_dijkstra_t_bar, ax_dijkstra_t_bar = plot_2d(
            t, c, f"{title} {dijkstra}")

        unf_elapsed_time_negative = []
        unf_elapsed_time = []
        astar_elapsed_time = []
        dijkstra_elapsed_time = []
        sync_net_transitions = []
        trace_length = []
        unf_v = []

        for entry in contents:
            unf_elapsed_time_negative.append(entry["unf_elapsed_time"])
            if entry["unf_elapsed_time"] != -1:
                unf_elapsed_time.append(entry["unf_elapsed_time"])
                astar_elapsed_time.append(entry["astar_elapsed_time"])
                dijkstra_elapsed_time.append(entry["dijkstra_elapsed_time"])
                sync_net_transitions.append(entry["sync_net_transitions"])
                trace_length.append(entry["trace_length"])
                unf_v.append(entry["unf_v"])

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
        ax_unf_tl_v.scatter(trace_length, unf_v, c=color)
        ax_unf_sp_v.scatter(sync_net_transitions, unf_v, c=color)
        ax_unf_sp_t.scatter(sync_net_transitions, unf_elapsed_time, c=color)

        unf_elapsed_time_log = np.emath.logn(2, unf_elapsed_time)
        ax_unf_sp_tlog.scatter(sync_net_transitions,
                               unf_elapsed_time_log,
                               c=color)
        ax_unf_v_tlog.scatter(unf_v, unf_elapsed_time_log, c=color)

    fig_unf_v_t.savefig(f"figures/stage_two/{unf}_v_t")
    fig_unf_tl_v.savefig(f"figures/stage_two/{unf}_tl_v")
    fig_unf_sp_v.savefig(f"figures/stage_two/{unf}_sp_v")
    fig_unf_sp_t.savefig(f"figures/stage_two/{unf}_sp_t")
    fig_unf_sp_tlog.savefig(f"figures/stage_two/{unf}_sp_tlog")
    fig_unf_v_tlog.savefig(f"figures/stage_two/{unf}_v_tlog")
    # plt.show()


def color_plot(x, y, title):
    dir_pairs = [
        ("results/sepsis/sepsis_05.json",
         "data/sepsis/Sepsis Cases - Event Log.xes"),
        ("results/bpic17/bpic17_02.json",
         "data/bpic17/BPI Challenge 2017.xes"),
        ("results/bpic19/bpic19_02.json",
         "data/bpic19/BPI_Challenge_2019.xes"),
        ("results/inthelarge/prAm6.json", "data/inthelarge/prAm6.tpn"),
        ("results/inthelarge/prBm6.json", "data/inthelarge/prBm6.tpn"),
        ("results/inthelarge/prCm6.json", "data/inthelarge/prCm6.tpn"),
        ("results/inthelarge/prDm6.json", "data/inthelarge/prDm6.tpn"),
        ("results/inthelarge/prEm6.json", "data/inthelarge/prEm6.tpn"),
        ("results/inthelarge/prFm6.json", "data/inthelarge/prFm6.tpn"),
    ]
    colors = [
        "red",
        "green",
        "blue",
        "pink",
        "purple",
        "magenta",
        "turquoise",
        "cyan",
        "yellow",
        "orange",
    ]
    color_index = 0
    for res_file, model_file in dir_pairs:
        fig, ax = plot_2d(x, y, res_file)
        # color = colors[color_index]
        # color_index += 1
        with open(res_file) as rf:
            contents = json.load(rf)
        for t in contents:
            if t["unf_elapsed_time"] != -1 and t["astar_elapsed_time"] != -1:
                alpha = (len(t["exclusive_transitions"]) /
                         t["trace_length"]) * 100
                color = [1, 0, 0, alpha]
                ax.scatter(alpha, t["unf_elapsed_time"], c="red")
    # fig.savefig(f"figures/stage_two/{title}_sp_v")
    plt.show()


def v_sync_prod_plots(x, y, title):
    files = [
        "results/sepsis/sepsis_05.json",
        "results/bpic17/bpic17_02.json",
        "results/bpic19/bpic19_02.json",
        "results/inthelarge/prAm6.json",
        "results/inthelarge/prBm6.json",
        "results/inthelarge/prCm6.json",
        "results/inthelarge/prDm6.json",
        "results/inthelarge/prEm6.json",
        "results/inthelarge/prFm6.json",
        "results/inthelarge/prGm6.json",
    ]
    colors = [
        "red",
        "green",
        "blue",
        "pink",
        "purple",
        "magenta",
        "turquoise",
        "cyan",
        "yellow",
        "orange",
    ]
    color_index = 0
    fig, ax = plot_2d(x, y, title)

    for res_file in files:
        color = colors[color_index]
        color_index += 1
        with open(res_file) as rf:
            contents = json.load(rf)
        for t in contents:
            if t["unf_elapsed_time"] != -1:
                ax.scatter(t["sync_net_transitions"], t["unf_v"], c=color)
    # fig.savefig(f"figures/stage_two/{title}_sp_v")
    plt.show()


def table():
    dir_pairs = [
        ("results/sepsis/sepsis_05.json",
         "data/sepsis/Sepsis Cases - Event Log.xes"),
        ("results/bpic17/bpic17_02.json",
         "data/bpic17/BPI Challenge 2017.xes"),
        ("results/bpic19/bpic19_02.json",
         "data/bpic19/BPI_Challenge_2019.xes"),
        ("results/inthelarge/prAm6.json", "data/inthelarge/prAm6.tpn"),
        ("results/inthelarge/prBm6.json", "data/inthelarge/prBm6.tpn"),
        ("results/inthelarge/prCm6.json", "data/inthelarge/prCm6.tpn"),
        ("results/inthelarge/prDm6.json", "data/inthelarge/prDm6.tpn"),
        ("results/inthelarge/prEm6.json", "data/inthelarge/prEm6.tpn"),
        ("results/inthelarge/prFm6.json", "data/inthelarge/prFm6.tpn"),
        ("results/inthelarge/prGm6.json", "data/inthelarge/prGm6.tpn")
    ]
    for res_file, _ in dir_pairs:
        with open(res_file, "r") as f:
            res = json.load(f)

        unf_t = [
            x["unf_elapsed_time"] for x in res if x["unf_elapsed_time"] != -1
        ]
        unf_q = [x["unf_q"] for x in res if x["unf_elapsed_time"] != -1]
        unf_v = [x["unf_v"] for x in res if x["unf_elapsed_time"] != -1]

        dijkstra_t = [
            x["dijkstra_elapsed_time"] for x in res
            if x["dijkstra_elapsed_time"] != -1
        ]
        dijkstra_q = [
            x["dijkstra_q"] for x in res if x["dijkstra_elapsed_time"] != -1
        ]
        dijkstra_v = [
            x["dijkstra_v"] for x in res if x["dijkstra_elapsed_time"] != -1
        ]

        astar_t = [
            x["astar_elapsed_time"] for x in res
            if x["astar_elapsed_time"] != -1
        ]
        astar_q = [x["astar_q"] for x in res if x["astar_elapsed_time"] != -1]
        astar_v = [x["astar_v"] for x in res if x["astar_elapsed_time"] != -1]

        table_data = {}
        try:
            table_data["avg_unf_t"] = sum(unf_t) / len(unf_t)
        except:
            pass
        try:
            table_data["avg_dijkstra_t"] = sum(dijkstra_t) / len(dijkstra_t)
        except:
            pass
        try:
            table_data["avg_astar_t"] = sum(astar_t) / len(astar_t)
        except:
            pass
        try:
            table_data["avg_q_difference_unf_dijkstra"] = sum([
                ((j - i) / i) * 100 for i, j in zip(unf_q, dijkstra_q)
            ]) / len(unf_q)
        except:
            pass
        try:
            table_data["avg_q_difference_unf_astar"] = sum([
                ((j - i) / i) * 100 for i, j in zip(unf_q, astar_q)
            ]) / len(unf_q)
        except:
            pass
        try:
            table_data["avg_v_difference_unf_dijkstra"] = sum([
                ((j - i) / i) * 100 for i, j in zip(unf_v, dijkstra_v)
            ]) / len(unf_v)
        except:
            pass
        try:
            table_data["avg_v_difference_unf_astar"] = sum([
                ((j - i) / i) * 100 for i, j in zip(unf_v, astar_v)
            ]) / len(unf_v)
        except:
            pass
        print(res_file)
        print(table_data)
        print(".................................................")


def view_models():
    dir_pairs = [
        ("results/inthelarge/prAm6.json", "data/inthelarge/prAm6.xes"),
        ("results/inthelarge/prBm6.json", "data/inthelarge/prBm6.xes"),
        ("results/inthelarge/prCm6.json", "data/inthelarge/prCm6.xes"),
        ("results/inthelarge/prDm6.json", "data/inthelarge/prDm6.xes"),
        ("results/inthelarge/prEm6.json", "data/inthelarge/prEm6.xes"),
        ("results/inthelarge/prFm6.json", "data/inthelarge/prFm6.xes"),
        ("results/inthelarge/prGm6.json", "data/inthelarge/prGm6.xes")
    ]

    for res_file, model_file in dir_pairs:
        if model_file.endswith(".tpn"):
            model_net, model_im, model_fm = ptn.import_from_tpn(model_file)
        else:
            xes_df = pm4py.read_xes(model_file)
            noise_threshold = 0.5 if "sepsis" in model_file else 0.2
            model_net, model_im, model_fm = pm4py.discover_petri_net_inductive(
                xes_df, noise_threshold=noise_threshold)
        print(model_file)
        pm4py.view_petri_net(model_net, model_im, model_fm)


def test_plot():
    dir_pairs = [
        ("results/sepsis/sepsis_05.json",
         "data/sepsis/Sepsis Cases - Event Log.xes", 1.2307692307692308,
         1.391304347826087),
        ("results/bpic17/bpic17_02.json", "data/bpic17/BPI Challenge 2017.xes",
         1.056338028169014, 1.6666666666666667),
        ("results/bpic19/bpic19_02.json", "data/bpic19/BPI_Challenge_2019.xes",
         1.1123595505617978, 1.8679245283018868),
        ("results/inthelarge/prAm6.json", "data/inthelarge/prAm6.tpn",
         1.159779614325069, 1.2132564841498559),
        ("results/inthelarge/prBm6.json", "data/inthelarge/prBm6.tpn",
         1.1798107255520505, 1.1798107255520505),
        ("results/inthelarge/prCm6.json", "data/inthelarge/prCm6.tpn",
         1.1798107255520505, 1.1798107255520505),
        ("results/inthelarge/prDm6.json", "data/inthelarge/prDm6.tpn",
         1.324009324009324, 1.0737240075614367),
        ("results/inthelarge/prEm6.json", "data/inthelarge/prEm6.tpn",
         1.1781818181818182, 1.1696750902527075),
        ("results/inthelarge/prFm6.json", "data/inthelarge/prFm6.tpn",
         1.2842809364548495, 1.0607734806629834),
    ]
    colorss = [
        "red",
        "green",
        "blue",
        "pink",
        "purple",
        "magenta",
        "turquoise",
        "cyan",
        "yellow",
        "orange",
    ]
    color_index = 0
    fig, ax = plot_2d("sp", "v", "")
    # fig, ax1 = plot_2d("v", "t", "unf")
    # fig, ax2 = plot_2d("q", "t", "unf")
    # fig, ax3 = plot_2d("tl", "t", "unf")
    # fig, ax4 = plot_2d("sp", "t", "unf")
    # fig, ax5 = plot_2d("c", "t", "unf")
    # fig, ax6 = plot_2d("f", "t", "unf")
    max_sp = 0
    max_v = 0
    max_tl = 0
    alphas = []
    sizes = []
    times = []
    x = []
    y = []
    colors = []
    # for res_file,model_file in dir_pairs:
    #     with open(res_file) as rf:
    #         contents = json.load(rf)
    #     for t in contents:
    #         if t["sync_net_transitions"] > max_sp:
    #             max_sp = t["sync_net_transitions"]
    #         if t["unf_v"] > max_v:
    #             max_v = t["unf_v"]
    #         if t["trace_length"] > max_tl:
    #             max_tl = t["trace_length"]

    for res_file, model_file, conc_factor, exc_factor in dir_pairs:
        if model_file.endswith(".tpn"):
            model_net, model_im, model_fm = ptn.import_from_tpn(model_file)
        else:
            xes_df = pm4py.read_xes(model_file)
            noise_threshold = 0.5 if "sepsis" in model_file else 0.2
            model_net, model_im, model_fm = pm4py.discover_petri_net_inductive(
                xes_df, noise_threshold=noise_threshold)
        avg_t_outgoing_arc = sum(
            [len(t.out_arcs)
             for t in model_net.transitions]) / len(model_net.transitions)
        avg_p_outgoing_arc = sum([len(p.out_arcs) for p in model_net.places
                                  ]) / len(model_net.places)
        avg_t_incoming_arc = sum(
            [len(t.in_arcs)
             for t in model_net.transitions]) / len(model_net.transitions)
        print(
            f"{model_file}, {avg_t_outgoing_arc}, {avg_p_outgoing_arc}, {avg_t_incoming_arc}"
        )
        subsize = []
        subx = []
        suby = []
        subalpha = []
        # color="red"
        color = colorss[color_index]
        color_index += 1
        with open(res_file) as rf:
            contents = json.load(rf)
        for t in contents:
            if t["unf_elapsed_time"] != -1 and t["astar_fitness"] > 0:
                subsize.append(t["astar_fitness"])
                subalpha.append(t["astar_fitness"])
                times.append(t["unf_elapsed_time"])
                subx.append(t["sync_net_transitions"])
                suby.append(t["unf_elapsed_time"])
                colors.append(color)
                # ax.scatter(t["trace_length"],t["unf_elapsed_time"],c=color,s=size)
                # ax1.scatter(t["unf_v"], t["unf_elapsed_time"], c=color)
                # ax2.scatter(t["unf_q"], t["unf_elapsed_time"], c=color)
                # ax3.scatter(t["trace_length"], t["unf_elapsed_time"], c=color)
                # ax4.scatter(t["sync_net_transitions"], t["unf_elapsed_time"], c=color)
                # ax5.scatter(t["unf_cost"], t["unf_elapsed_time"], c=color)
                # ax6.scatter(t["astar_fitness"], t["unf_elapsed_time"], c=color)

        subsize = sklearn.preprocessing.minmax_scale(subsize, (1, 50))
        # subx = sklearn.preprocessing.scale(subx)
        # suby = sklearn.preprocessing.scale(suby)
        # subalpha = sklearn.preprocessing.minmax_scale(subalpha)
        x.extend(subx)
        y.extend(suby)
        sizes.extend(subsize)
        alphas.extend(subalpha)
    # sizes = [x * 10 for x in sizes]
    # alphas=[x if x<= 1 else 1 for x in alphas]
    # times = [x*100 for x in times]
    # times = np.emath.logn(2, times)
    # y = np.log(y)
    # colors=[[1,0,0,x] for x in alphas]
    ax.scatter(x, y, c=colors, s=sizes)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    # plt.show()


def regress_a_priori():
    dir_pairs = [
        ("results/sepsis/sepsis_05.json",
         "data/sepsis/Sepsis Cases - Event Log.xes", 1.2307692307692308,
         1.391304347826087),
        ("results/bpic17/bpic17_02.json", "data/bpic17/BPI Challenge 2017.xes",
         1.056338028169014, 1.6666666666666667),
        ("results/bpic19/bpic19_02.json", "data/bpic19/BPI_Challenge_2019.xes",
         1.1123595505617978, 1.8679245283018868),
        ("results/inthelarge/prAm6.json", "data/inthelarge/prAm6.tpn",
         1.159779614325069, 1.2132564841498559),
        ("results/inthelarge/prBm6.json", "data/inthelarge/prBm6.tpn",
         1.1798107255520505, 1.1798107255520505),
        ("results/inthelarge/prCm6.json", "data/inthelarge/prCm6.tpn",
         1.1798107255520505, 1.1798107255520505),
        ("results/inthelarge/prDm6.json", "data/inthelarge/prDm6.tpn",
         1.324009324009324, 1.0737240075614367),
        ("results/inthelarge/prEm6.json", "data/inthelarge/prEm6.tpn",
         1.1781818181818182, 1.1696750902527075),
        ("results/inthelarge/prFm6.json", "data/inthelarge/prFm6.tpn",
         1.2842809364548495, 1.0607734806629834),
    ]
    x = []
    y = []
    for res_file, model_file, conc_factor, exc_factor in dir_pairs:
        # x = []
        # y = []
        # print(res_file)
        with open(res_file) as rf:
            contents = json.load(rf)

        for t in contents:
            if t["unf_elapsed_time"] != -1:
                sync_prod_size = t["sync_net_transitions"]
                tl = t["trace_length"]

                indep = [
                    sync_prod_size,
                    exc_factor,
                    conc_factor,
                ]

                x.append(indep)
                y.append(np.log(t["unf_elapsed_time"]))

    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    # x, y, random_state=2)

    m = SMWrapper(sm.OLS)

    poly = sklearn.preprocessing.PolynomialFeatures(interaction_only=True)

    # Scale each feature
    # Use regular scale instead of robust scale because we don't have "true" outliers because our outliers aren't by random chance
    scaler = sklearn.preprocessing.StandardScaler()

    # Pipeline to not get dataleaks
    pipeline = sklearn.pipeline.Pipeline([
        ("scaler", scaler),
        # ("poly", poly),
        ("model", m),
    ])

    pipeline.fit(x, y)
    score = pipeline.score(x, y)
    print(score)
    print(pipeline["model"].results_.summary())
    # model = pipeline.fit(x, y)

    # Residuals plot
    y_pred = pipeline.predict(x)
    fig, axs = plt.subplots(ncols=2, figsize=(8, 4))
    sklearn.metrics.PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="actual_vs_predicted",
        subsample=100,
        ax=axs[0],
        random_state=0,
    )
    axs[0].set_title("Actual vs. Predicted values")
    sklearn.metrics.PredictionErrorDisplay.from_predictions(
        y,
        y_pred=y_pred,
        kind="residual_vs_predicted",
        subsample=100,
        ax=axs[1],
        random_state=0,
    )
    axs[1].set_title("Residuals vs. Predicted Values")
    fig.suptitle("Plotting cross-validated predictions")
    plt.tight_layout()
    plt.show()


def regress():
    dir_pairs = [
        ("results/sepsis/sepsis_05.json",
         "data/sepsis/Sepsis Cases - Event Log.xes"),
        ("results/bpic17/bpic17_02.json",
         "data/bpic17/BPI Challenge 2017.xes"),
        ("results/bpic19/bpic19_02.json",
         "data/bpic19/BPI_Challenge_2019.xes"),
        ("results/inthelarge/prAm6.json", "data/inthelarge/prAm6.tpn"),
        ("results/inthelarge/prBm6.json", "data/inthelarge/prBm6.tpn"),
        ("results/inthelarge/prCm6.json", "data/inthelarge/prCm6.tpn"),
        ("results/inthelarge/prDm6.json", "data/inthelarge/prDm6.tpn"),
        ("results/inthelarge/prEm6.json", "data/inthelarge/prEm6.tpn"),
        ("results/inthelarge/prFm6.json", "data/inthelarge/prFm6.tpn"),
    ]
    x = []
    y = []
    for res_file, model_file in dir_pairs:
        # print(res_file)
        with open(res_file) as rf:
            contents = json.load(rf)

        for t in contents:
            if t["unf_elapsed_time"] != -1 and t["astar_fitness"] != -1:
                trace_length = t["trace_length"]
                sync_prod_size = t["sync_net_transitions"]
                fitness = t["astar_fitness"]
                cost = t["unf_cost"]
                visited_states = t["unf_v"]
                queued_states = t["unf_q"]
                # Applying recursive feature elimination we end up with tl, v, cost and fitness.
                # We remove queued_states because it's the smallest (negative) coefficient
                # We fit again, then remove sync_prod_size because it's again the smallest (negative)
                # We leave it at this, since now all coefficients are positive
                # Further feature removal drops coeffient of determination substantially so we don't remove any further features
                indep = [
                    trace_length,
                    visited_states,
                    sync_prod_size,
                    cost,
                    queued_states,
                    fitness,
                ]
                x.append(indep)
                y.append(t["unf_elapsed_time"])

    # Spearman R to show multicolinearity is very present
    print(stats.spearmanr(x))
    # 5 folds cause that sounds reasonable and is the standard
    # shuffle because we don't want any folds to be 1 dataset only
    cv = sklearn.model_selection.KFold(shuffle=True, random_state=1)

    # polylolly?
    poly = sklearn.preprocessing.PolynomialFeatures(2)
    # Use Ridge because multicolinearity between many of the metrics
    m = sklearn.linear_model.RidgeCV(scoring="r2", cv=cv)
    # Scale each feature
    # Use regular scale instead of robust scale because we don't have "true" outliers because our outliers aren't by random chance
    scaler = sklearn.preprocessing.StandardScaler()

    # Pipeline to not get dataleaks
    pipeline = sklearn.pipeline.Pipeline([
        ("scaler", scaler),
        ("poly", poly),
        ("model", m),
    ])

    # scores = sklearn.model_selection.cross_val_score(pipeline, x, y, cv=cv)
    # brief interpretation of results as follows:
    # coefficient for trace length is highest, but the range of
    scores = sklearn.model_selection.cross_validate(pipeline,
                                                    x,
                                                    y,
                                                    cv=cv,
                                                    scoring=["r2"],
                                                    return_estimator=True)
    print(scores["test_r2"])
    for p in scores["estimator"]:
        pass
        # print(f"coefficients: {p["model"].coef_}")


def fix_sp_count():
    pass
    # dir_pairs = [
    #     ("results/sepsis/sepsis_05.json",
    #      "data/sepsis/Sepsis Cases - Event Log.xes"),
    #     ("results/bpic17/bpic17_02.json",
    #      "data/bpic17/BPI Challenge 2017.xes"),
    #     ("results/bpic19/bpic19_02.json",
    #      "data/bpic19/BPI_Challenge_2019.xes"),
    #     ("results/inthelarge/prAm6.json", "data/inthelarge/prAm6.tpn"),
    #     ("results/inthelarge/prBm6.json", "data/inthelarge/prBm6.tpn"),
    #     ("results/inthelarge/prCm6.json", "data/inthelarge/prCm6.tpn"),
    #     ("results/inthelarge/prDm6.json", "data/inthelarge/prDm6.tpn"),
    #     ("results/inthelarge/prEm6.json", "data/inthelarge/prEm6.tpn"),
    #     ("results/inthelarge/prFm6.json", "data/inthelarge/prFm6.tpn"),
    #     ("results/inthelarge/prGm6.json", "data/inthelarge/prGm6.tpn")
    # ]
    # for res_file, model_file in dir_pairs:
    #     if model_file.endswith(".tpn"):
    #         data_file = model_file.removesuffix("tpn") + "xes"
    #         xes_df = pm4py.read_xes(data_file)
    #         model_net, model_im, model_fm = ptn.import_from_tpn(model_file)
    #     else:
    #         xes_df = pm4py.read_xes(model_file)
    #         noise_threshold = 0.5 if "sepsis" in model_file else 0.2
    #         model_net, model_im, model_fm = pm4py.discover_petri_net_inductive(
    #             xes_df, noise_threshold=noise_threshold)
    #     xes_el = pm4py.convert_to_event_log(pm4py.format_dataframe(xes_df))

    #     if "bpic" in model_file:
    #         import random
    #         random.seed(1)
    #         traces = random.sample(xes_el, 1000)
    #     else:
    #         traces = xes_el

    #     with open(res_file, "r") as rf:
    #         file_data = json.load(rf)

    #     for i in range(len(traces)):
    #         trace = traces[i]
    #         trace_net, trace_net_im, trace_net_fm = ptn.construct_trace_net(
    #             trace, "concept:name", "concept:name")

    #         sync_net, sync_im, sync_fm, cost_function = ptn.construct_synchronous_product(
    #             model_net, model_im, model_fm, trace_net, trace_net_im,
    #             trace_net_fm)
    #         file_data[i]["sync_net_transitions"] = len(sync_net.transitions)
    #         file_data[i]["sync_net_places"] = len(sync_net.places)
    #     with open(res_file, "w") as wf:
    #         rstr = json.dumps(file_data, indent=4)
    #         wf.write(rstr)


def add_metrics():
    dir_pairs = [
        ("results/sepsis/sepsis_05.json",
         "data/sepsis/Sepsis Cases - Event Log.xes"),
        ("results/bpic17/bpic17_02.json",
         "data/bpic17/BPI Challenge 2017.xes"),
        ("results/bpic19/bpic19_02.json",
         "data/bpic19/BPI_Challenge_2019.xes"),
        ("results/inthelarge/prAm6.json", "data/inthelarge/prAm6.tpn"),
        ("results/inthelarge/prBm6.json", "data/inthelarge/prBm6.tpn"),
        ("results/inthelarge/prCm6.json", "data/inthelarge/prCm6.tpn"),
        ("results/inthelarge/prDm6.json", "data/inthelarge/prDm6.tpn"),
        ("results/inthelarge/prEm6.json", "data/inthelarge/prEm6.tpn"),
        ("results/inthelarge/prFm6.json", "data/inthelarge/prFm6.tpn"),
        ("results/inthelarge/prGm6.json", "data/inthelarge/prGm6.tpn")
    ]
    for res_file, model_file in dir_pairs:
        if model_file.endswith(".tpn"):
            data_file = model_file.removesuffix("tpn") + "xes"
            xes_df = pm4py.read_xes(data_file)
            model_net, model_im, model_fm = ptn.import_from_tpn(model_file)
        else:
            xes_df = pm4py.read_xes(model_file)
            noise_threshold = 0.5 if "sepsis" in model_file else 0.2
            model_net, model_im, model_fm = pm4py.discover_petri_net_inductive(
                xes_df, noise_threshold=noise_threshold)
        xes_el = pm4py.convert_to_event_log(pm4py.format_dataframe(xes_df))

        if "bpic" in model_file:
            import random
            random.seed(1)
            traces = random.sample(xes_el, 1000)
        else:
            traces = xes_el

        with open(res_file, "r") as rf:
            file_data = json.load(rf)

        for i in range(len(traces)):
            trace = traces[i]
            trace_net, trace_net_im, trace_net_fm = ptn.construct_trace_net(
                trace, "concept:name", "concept:name")
            t_labels = [x.label for x in trace_net.transitions]
            # Determining how much "exlusivity" a specific transition is connected to is not really feasible, so we just count the transitions that are  in some way exclusive with something
            exclusive_transitions = set()
            # Getting concurrency is impossible without the unfolding alignment, which I don't have stored...
            for t in model_net.transitions:
                for arc in t.in_arcs:
                    if len(arc.source.out_arcs) > 1:
                        if t.label in t_labels:
                            exclusive_transitions.add(t.label)

            # Get events in trace that are not in model
            missing_events = []
            for t1 in trace_net.transitions:
                found = False
                for t2 in model_net.transitions:
                    if t1.label == t2.label:
                        found = True
                if not found:
                    missing_events.append(t1.label)

            file_data[i]["missing_events"] = missing_events
            del file_data[i]["transition_outgoing_arc_pairs"]
            file_data[i]["exclusive_transitions"] = list(exclusive_transitions)
        with open(res_file, "w") as wf:
            rstr = json.dumps(file_data, indent=4)
            wf.write(rstr)


if __name__ == "__main__":
    # No heuristics stage one
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

    # stage_two_figures()
    # v_trace_length_plots()
    # v_sync_prod_plots("synchronous product transitions", "visited states",
    #   "unfolding")
    # table()
    # view_models()
    # color_plot("v", "t", "unf")
    # regress()
    # test_plot()
    regress_a_priori()
    # view_models()
# TODO check avg number of input places for each model. higher should mean slower?
