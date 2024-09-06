import os

import sklearn.decomposition
import sklearn.feature_selection
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
import statistics

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
        res = self.results_.predict(X)
        self.residuals = self.results_.resid
        return res


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


def e1(filepath):
    xes_df = pm4py.read_xes("data/sepsis/Sepsis Cases - Event Log.xes")
    model_net, model_im, model_fm = pm4py.discover_petri_net_inductive(xes_df)
    xes_el = pm4py.convert_to_event_log(pm4py.format_dataframe(xes_df))
    with open(filepath, "r") as f:
        res = json.load(f)

    sp = []
    for t in xes_el:
        trace_net, trace_net_im, trace_net_fm = ptn.construct_trace_net(
            t, "concept:name", "concept:name")
        sync_net, sync_im, sync_fm, cost_function = ptn.construct_synchronous_product(
            model_net, model_im, model_fm, trace_net, trace_net_im,
            trace_net_fm)
        sp.append(len(sync_net.transitions))

    table_data = {}

    fig_sp_v = plt.figure()
    fig_sp_v.suptitle("unfolding")
    ax_sp_v = fig_sp_v.add_subplot()
    ax_sp_v.set_xlabel("synchronous product transitions")
    ax_sp_v.set_ylabel("visisted states")

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

    unf_df = pd.DataFrame(unf_elapsed_time)
    print(unf_df.describe())

    dijkstra_elapsed_time = [x["dijkstra_elapsed_time"] for x in res]
    dijkstra_q = [x["dijkstra_q"] for x in res]
    dijkstra_v = [x["dijkstra_v"] for x in res]
    dijkstra_cost = [x["dijkstra_cost"] for x in res]

    dijkstra_df = pd.DataFrame(dijkstra_elapsed_time)
    print(dijkstra_df.describe())

    astar_elapsed_time = [x["astar_elapsed_time"] for x in res]
    astar_q = [x["astar_q"] for x in res]
    astar_v = [x["astar_v"] for x in res]
    astar_cost = [x["astar_cost"] for x in res]

    astar_df = pd.DataFrame(astar_elapsed_time)
    print(astar_df.describe())

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
    ax_sp_v.scatter(sp, unf_v)

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


def e2_figures(dir_names_color_title_pairs, subfolder_name):
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

                l_vt_no_deviations = ax_vt_no_deviations.scatter(
                    avg_v_no_deviations, avg_t_no_deviations, c=color)

                ax_v_no_deviations.scatter(params.breadth,
                                           params.depth,
                                           avg_v_no_deviations,
                                           c=color)
                ax_t_no_deviations.scatter(params.breadth,
                                           params.depth,
                                           avg_t_no_deviations,
                                           c=color)
                l_tl_v_no_deviations = ax_tl_v_no_deviations.scatter(
                    avg_trace_length, avg_v_no_deviations, c=color)
                l_sp_v_no_deviations = ax_sp_v_no_deviations.scatter(
                    sync_prod_size_no_deviation, avg_v_no_deviations, c=color)
            except ZeroDivisionError:
                pass

            try:
                avg_v_start_deviations = sum(v_start_deviations) / len(
                    v_start_deviations)
                avg_t_start_deviations = sum(t_start_deviations) / len(
                    t_start_deviations)

                l_vt_start_deviations = ax_vt_start_deviations.scatter(
                    avg_v_start_deviations, avg_t_start_deviations, c=color)

                ax_v_start_deviations.scatter(params.breadth,
                                              params.depth,
                                              avg_v_start_deviations,
                                              c=color)
                ax_t_start_deviations.scatter(params.breadth,
                                              params.depth,
                                              avg_t_start_deviations,
                                              c=color)
                l_tl_v_start_deviations = ax_tl_v_start_deviations.scatter(
                    avg_trace_length, avg_v_start_deviations, c=color)
                l_sp_v_start_deviations = ax_sp_v_start_deviations.scatter(
                    sync_prod_size_with_deviation,
                    avg_v_start_deviations,
                    c=color)
            except ZeroDivisionError:
                pass

            try:
                avg_v_halfway_deviations = sum(v_halfway_deviations) / len(
                    v_halfway_deviations)
                avg_t_halfway_deviations = sum(t_halfway_deviations) / len(
                    t_halfway_deviations)

                l_vt_halfway_deviations = ax_vt_halfway_deviations.scatter(
                    avg_v_halfway_deviations,
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
                l_tl_v_halfway_deviations = ax_tl_v_halfway_deviations.scatter(
                    avg_trace_length, avg_v_halfway_deviations, c=color)
                l_sp_v_halfway_deviations = ax_sp_v_halfway_deviations.scatter(
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

                l_vt_end_deviations = ax_vt_end_deviations.scatter(
                    avg_v_end_deviations, avg_t_end_deviations, c=color)

                ax_v_end_deviations.scatter(params.breadth,
                                            params.depth,
                                            avg_v_end_deviations,
                                            c=color)
                ax_t_end_deviations.scatter(params.breadth,
                                            params.depth,
                                            avg_t_end_deviations,
                                            c=color)
                l_tl_v_end_deviations = ax_tl_v_end_deviations.scatter(
                    avg_trace_length, avg_v_end_deviations, c=color)
                l_sp_v_end_deviations = ax_sp_v_end_deviations.scatter(
                    sync_prod_size_with_deviation,
                    avg_v_end_deviations,
                    c=color)
            except ZeroDivisionError:
                pass

        l_vt_no_deviations.set_label(title)
        l_vt_start_deviations.set_label(title)
        l_vt_halfway_deviations.set_label(title)
        l_vt_end_deviations.set_label(title)
        l_tl_v_no_deviations.set_label(title)
        l_tl_v_start_deviations.set_label(title)
        l_tl_v_halfway_deviations.set_label(title)
        l_tl_v_end_deviations.set_label(title)
        l_sp_v_no_deviations.set_label(title)
        l_sp_v_start_deviations.set_label(title)
        l_sp_v_halfway_deviations.set_label(title)
        l_vt_start_deviations.set_label(title)
        l_vt_halfway_deviations.set_label(title)
        l_vt_end_deviations.set_label(title)
        l_tl_v_no_deviations.set_label(title)
        l_tl_v_start_deviations.set_label(title)
        l_tl_v_halfway_deviations.set_label(title)
        l_tl_v_end_deviations.set_label(title)
        l_sp_v_no_deviations.set_label(title)
        l_sp_v_start_deviations.set_label(title)
        l_sp_v_halfway_deviations.set_label(title)
        l_sp_v_end_deviations.set_label(title)
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

    fig_vt_no_deviations.legend()
    fig_vt_start_deviations.legend()
    fig_vt_halfway_deviations.legend()
    fig_vt_end_deviations.legend()
    fig_tl_v_no_deviations.legend()
    fig_tl_v_start_deviations.legend()
    fig_tl_v_halfway_deviations.legend()
    fig_tl_v_end_deviations.legend()
    fig_sp_v_no_deviations.legend()
    fig_sp_v_start_deviations.legend()
    fig_sp_v_halfway_deviations.legend()
    fig_sp_v_end_deviations.legend()

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

    fig_sp_v_no_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/sp_v_no")
    fig_sp_v_start_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/sp_v_start")
    fig_sp_v_halfway_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/sp_v_halfway")
    fig_sp_v_end_deviations.savefig(
        f"figures/stage_one/{subfolder_name}/sp_v_end")
    # plt.show()


def e3_figures():
    res_file_path_color_title_pairs = [
        ("results/sepsis/sepsis_05.json", "#2f4f4f", "SP"),
        ("results/bpic17/bpic17_02.json", "#2e8b57", "BPIC17"),
        ("results/bpic19/bpic19_02.json", "#8b0000", "BPIC19"),
        ("results/inthelarge/prAm6.json", "#808000", "ITL prAm6"),
        ("results/inthelarge/prBm6.json", "#000080", "ITL prBm6"),
        ("results/inthelarge/prCm6.json", "#ff0000", "ITL prCm6"),
        ("results/inthelarge/prDm6.json", "#ffa500", "ITL prDm6"),
        ("results/inthelarge/prEm6.json", "#7cfc00", "ITL prEm6"),
        ("results/inthelarge/prFm6.json", "#ba55d3", "ITL prFm6"),
        ("results/inthelarge/prGm6.json", "#00ffff", "ITL prGm6"),
        ("results/hospital_billing/hospital_billing_02.json", "#0000ff", "HB"),
        ("results/bpic12/bpic12_02.json", "#f08080", "BPIC12"),
        ("results/bpic13/bpic13_02.json", "#1e90ff", "BPIC13"),
        ("results/traffic/traffic_02.json", "#ffff54", "TR"),
        ("results/bpic13/bpic13_split.json", "#dda0dd", "BPIC13_SPLIT"),
        ("results/sepsis/sepsis_split.json", "#ff1493", "SP_SPLIT"),
        ("results/bpic12/bpic12_split.json", "#f5deb3", "BPIC12_SPLIT"),
        ("results/bpic17/bpic17_split.json", "#98fb98", "BPIC17_SPLIT"),
        ("results/hospital_billing/hospital_billing_split.json", "#87cefa",
         "HB_SPLIT"),
        ("results/traffic/traffic_split.json", "#000000", "TR_SPLIT"),
    ]
    v = "visited states"
    vlog = "visited states (ln)"
    q = "queued states"
    qlog = "queued states (ln)"
    fit = "fitness"
    cost = "cost"
    c = "count"
    t = "time (s)"
    tlog = "time (ln(s))"
    tl = "trace length"
    sp = "synchronous product transitions"
    unf = "unfolding"
    astar = "astar"
    dijkstra = "dijkstra"

    fig_unf_v_t, ax_unf_v_t = plot_2d(v, t, unf)
    fig_unf_q_t, ax_unf_q_t = plot_2d(q, t, unf)
    fig_unf_tl_v, ax_unf_tl_v = plot_2d(tl, v, unf)
    fig_unf_sp_v, ax_unf_sp_v = plot_2d(sp, v, unf)
    fig_unf_sp_t, ax_unf_sp_t = plot_2d(sp, t, unf)
    fig_unf_vlog_tlog, ax_unf_vlog_tlog = plot_2d(vlog, tlog, unf)
    fig_unf_qlog_tlog, ax_unf_qlog_tlog = plot_2d(qlog, tlog, unf)
    fig_unf_sp_tlog, ax_unf_sp_tlog = plot_2d(sp, tlog, unf)
    fig_unf_fit_t, ax_unf_fit_t = plot_2d(fit, t, unf)
    fig_unf_cost_t, ax_unf_cost_t = plot_2d(cost, t, unf)

    for res_file_path, color, title in res_file_path_color_title_pairs:
        with open(res_file_path) as rf:
            contents = json.load(rf)

        fig_unf_t_bar, ax_unf_t_bar = plot_2d(t, c, f"{title} {unf}")
        fig_astar_t_bar, ax_astar_t_bar = plot_2d(t, c, f"{title} {astar}")
        fig_dijkstra_t_bar, ax_dijkstra_t_bar = plot_2d(
            t, c, f"{title} {dijkstra}")

        fig_unf_v_t_per, ax_unf_v_t_per = plot_2d(v, t, f"{title} {unf}")
        fig_unf_sp_v_per, ax_unf_sp_v_per = plot_2d(sp, v, f"{title} {unf}")
        fig_unf_sp_t_per, ax_unf_sp_t_per = plot_2d(sp, t, f"{title} {unf}")

        unf_elapsed_time_negative = []
        unf_elapsed_time = []
        astar_elapsed_time = []
        dijkstra_elapsed_time = []
        sync_net_transitions = []
        trace_length = []
        unf_v = []
        unf_q = []
        fits = []
        costs = []

        for entry in contents:
            unf_elapsed_time_negative.append(entry["unf_elapsed_time"])
            if entry["unf_elapsed_time"] != -1:
                unf_elapsed_time.append(entry["unf_elapsed_time"])
                astar_elapsed_time.append(entry["astar_elapsed_time"])
                dijkstra_elapsed_time.append(entry["dijkstra_elapsed_time"])
                sync_net_transitions.append(entry["sync_net_transitions"])
                trace_length.append(entry["trace_length"])
                unf_v.append(entry["unf_v"])
                unf_q.append(entry["unf_q"])
                fits.append(entry["astar_fitness"])
                costs.append(entry["unf_cost"])

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

        ax_unf_v_t.scatter(unf_v, unf_elapsed_time, c=color, label=title)
        ax_unf_q_t.scatter(unf_q, unf_elapsed_time, c=color, label=title)
        ax_unf_tl_v.scatter(trace_length, unf_v, c=color, label=title)
        ax_unf_sp_v.scatter(sync_net_transitions, unf_v, c=color, label=title)
        ax_unf_sp_t.scatter(sync_net_transitions,
                            unf_elapsed_time,
                            c=color,
                            label=title)
        ax_unf_cost_t.scatter(costs, unf_elapsed_time, c=color, label=title)
        ax_unf_fit_t.scatter(fits, unf_elapsed_time, c=color, label=title)

        unf_elapsed_time_log = np.log(unf_elapsed_time)
        unf_v_log = np.log(unf_v)
        unf_q_log = np.log(unf_q)
        ax_unf_sp_tlog.scatter(sync_net_transitions,
                               unf_elapsed_time_log,
                               c=color,
                               label=title)
        ax_unf_vlog_tlog.scatter(unf_v_log,
                                 unf_elapsed_time_log,
                                 c=color,
                                 label=title)
        ax_unf_qlog_tlog.scatter(unf_q_log,
                                 unf_elapsed_time_log,
                                 c=color,
                                 label=title)

        ax_unf_v_t_per.scatter(unf_v, unf_elapsed_time, c=color, label=title)
        ax_unf_sp_v_per.scatter(sync_net_transitions,
                                unf_v,
                                c=color,
                                label=title)
        ax_unf_sp_t_per.scatter(sync_net_transitions,
                                unf_elapsed_time,
                                c=color,
                                label=title)
        fig_unf_v_t_per.savefig(f"figures/stage_two/{title}_{unf}_v_t")
        fig_unf_sp_v_per.savefig(f"figures/stage_two/{title}_{unf}_sp_v")
        fig_unf_sp_t_per.savefig(f"figures/stage_two/{title}_{unf}_sp_t")

    fig_unf_v_t.legend()
    fig_unf_q_t.legend()
    fig_unf_tl_v.legend()
    fig_unf_sp_v.legend()
    fig_unf_sp_t.legend()
    fig_unf_sp_tlog.legend()
    fig_unf_vlog_tlog.legend()
    fig_unf_qlog_tlog.legend()
    fig_unf_fit_t.legend()
    fig_unf_cost_t.legend()

    fig_unf_v_t.savefig(f"figures/stage_two/{unf}_v_t")
    fig_unf_q_t.savefig(f"figures/stage_two/{unf}_q_t")
    fig_unf_tl_v.savefig(f"figures/stage_two/{unf}_tl_v")
    fig_unf_sp_v.savefig(f"figures/stage_two/{unf}_sp_v")
    fig_unf_sp_t.savefig(f"figures/stage_two/{unf}_sp_t")
    fig_unf_sp_tlog.savefig(f"figures/stage_two/{unf}_sp_tlog")
    fig_unf_vlog_tlog.savefig(f"figures/stage_two/{unf}_vlog_tlog")
    fig_unf_qlog_tlog.savefig(f"figures/stage_two/{unf}_qlog_tlog")
    fig_unf_fit_t.savefig(f"figures/stage_two/{unf}_fit_t")
    fig_unf_cost_t.savefig(f"figures/stage_two/{unf}_cost_t")
    # plt.show()


def e3_table():
    res_file_path_title_pairs = [
        ("results/sepsis/sepsis_05.json", "SP"),
        ("results/bpic17/bpic17_02.json", "BPIC17"),
        ("results/bpic19/bpic19_02.json", "BPIC19"),
        ("results/inthelarge/prAm6.json", "ITL prAm6"),
        ("results/inthelarge/prBm6.json", "ITL prBm6"),
        ("results/inthelarge/prCm6.json", "ITL prCm6"),
        ("results/inthelarge/prDm6.json", "ITL prDm6"),
        ("results/inthelarge/prEm6.json", "ITL prEm6"),
        ("results/inthelarge/prFm6.json", "ITL prFm6"),
        # ("results/inthelarge/prGm6.json", "ITL prGm6"),
        ("results/hospital_billing/hospital_billing_02.json", "HB"),
        ("results/bpic12/bpic12_02.json", "BPIC12"),
        ("results/bpic13/bpic13_02.json", "BPIC13"),
        ("results/traffic/traffic_02.json", "TR"),
        ("results/bpic13/bpic13_split.json", "BPIC13_SPLIT"),
        ("results/sepsis/sepsis_split.json", "SP_SPLIT"),
        ("results/bpic12/bpic12_split.json", "BPIC12_SPLIT"),
        ("results/bpic17/bpic17_split.json", "BPIC17_SPLIT"),
        ("results/hospital_billing/hospital_billing_split.json", "HB_SPLIT"),
        ("results/traffic/traffic_split.json", "TR_SPLIT"),
    ]
    means_median_pairs = []
    for res_file_path, title in res_file_path_title_pairs:
        with open(res_file_path) as rf:
            contents = json.load(rf)
        unf_table = {}
        unf_q = []
        unf_v = []
        unf_t = []
        unf_tl = []

        astar_table = {}
        astar_q = []
        astar_v = []
        astar_t = []
        astar_tl = []

        dijkstra_table = {}
        dijkstra_q = []
        dijkstra_v = []
        dijkstra_t = []
        dijkstra_tl = []

        unf_astar_table = {}
        tl_unf_astar_complete = []
        q_unf_unf_astar_complete = []
        v_unf_unf_astar_complete = []
        t_unf_unf_astar_complete = []

        q_astar_unf_astar_complete = []
        v_astar_unf_astar_complete = []
        t_astar_unf_astar_complete = []

        unf_dijkstra_table = {}
        tl_unf_dijkstra_complete = []
        q_unf_unf_dijkstra_complete = []
        v_unf_unf_dijkstra_complete = []
        t_unf_unf_dijkstra_complete = []

        q_dijkstra_unf_dijkstra_complete = []
        v_dijkstra_unf_dijkstra_complete = []
        t_dijkstra_unf_dijkstra_complete = []

        for entry in contents:
            if entry["unf_elapsed_time"] != -1:
                unf_q.append(entry["unf_q"])
                unf_v.append(entry["unf_v"])
                unf_t.append(entry["unf_elapsed_time"])
                unf_tl.append(entry["trace_length"])
            if entry["astar_elapsed_time"] != -1:
                astar_q.append(entry["astar_q"])
                astar_v.append(entry["astar_v"])
                astar_t.append(entry["astar_elapsed_time"])
                astar_tl.append(entry["trace_length"])
            if entry["dijkstra_elapsed_time"] != -1:
                dijkstra_q.append(entry["dijkstra_q"])
                dijkstra_v.append(entry["dijkstra_v"])
                dijkstra_t.append(entry["dijkstra_elapsed_time"])
                dijkstra_tl.append(entry["trace_length"])
            if entry["unf_elapsed_time"] != -1 and entry[
                    "astar_elapsed_time"] != -1:
                tl_unf_astar_complete.append(entry["trace_length"])
                q_unf_unf_astar_complete.append(entry["unf_q"])
                v_unf_unf_astar_complete.append(entry["unf_v"])
                t_unf_unf_astar_complete.append(entry["unf_elapsed_time"])

                q_astar_unf_astar_complete.append(entry["astar_q"])
                v_astar_unf_astar_complete.append(entry["astar_v"])
                t_astar_unf_astar_complete.append(entry["astar_elapsed_time"])
            if entry["unf_elapsed_time"] != -1 and entry[
                    "dijkstra_elapsed_time"] != -1:
                tl_unf_dijkstra_complete.append(entry["trace_length"])
                q_unf_unf_dijkstra_complete.append(entry["unf_q"])
                v_unf_unf_dijkstra_complete.append(entry["unf_v"])
                t_unf_unf_dijkstra_complete.append(entry["unf_elapsed_time"])

                q_dijkstra_unf_dijkstra_complete.append(entry["dijkstra_q"])
                v_dijkstra_unf_dijkstra_complete.append(entry["dijkstra_v"])
                t_dijkstra_unf_dijkstra_complete.append(
                    entry["dijkstra_elapsed_time"])

        means_median_pairs.append(
            ((sum(unf_t) / len(unf_t)), statistics.median(unf_t)))
        # Unf
        if len(unf_t) > 0:
            unf_table["samples"] = len(unf_q)
            unf_table["avg_trace_length"] = sum(unf_tl) / len(unf_tl)
            unf_table["min_trace_length"] = min(unf_tl)
            unf_table["max_trace_length"] = max(unf_tl)
            unf_table["avg_unf_q"] = sum(unf_q) / len(unf_q)
            unf_table["min_unf_q"] = min(unf_q)
            unf_table["max_unf_q"] = max(unf_q)
            unf_table["avg_unf_v"] = sum(unf_v) / len(unf_v)
            unf_table["min_unf_v"] = min(unf_v)
            unf_table["max_unf_v"] = max(unf_v)
            unf_table["avg_unf_t"] = sum(unf_t) / len(unf_t)
            unf_table["min_unf_t"] = min(unf_t)
            unf_table["max_unf_t"] = max(unf_t)
        if len(astar_t) > 0:
            # Astar
            astar_table["samples"] = len(astar_q)
            astar_table["avg_trace_length"] = sum(astar_tl) / len(astar_tl)
            astar_table["min_trace_length"] = min(astar_tl)
            astar_table["max_trace_length"] = max(astar_tl)
            astar_table["avg_astar_q"] = sum(astar_q) / len(astar_q)
            astar_table["min_astar_q"] = min(astar_q)
            astar_table["max_astar_q"] = max(astar_q)
            astar_table["avg_astar_v"] = sum(astar_v) / len(astar_v)
            astar_table["min_astar_v"] = min(astar_v)
            astar_table["max_astar_v"] = max(astar_v)
            astar_table["avg_astar_t"] = sum(astar_t) / len(astar_t)
            astar_table["min_astar_t"] = min(astar_t)
            astar_table["max_astar_t"] = max(astar_t)
        if len(dijkstra_t) > 0:
            # Dijkstra
            dijkstra_table["samples"] = len(dijkstra_q)
            dijkstra_table["avg_trace_length"] = sum(dijkstra_tl) / len(
                dijkstra_tl)
            dijkstra_table["min_trace_length"] = min(dijkstra_tl)
            dijkstra_table["max_trace_length"] = max(dijkstra_tl)
            dijkstra_table["avg_dijkstra_q"] = sum(dijkstra_q) / len(
                dijkstra_q)
            dijkstra_table["min_dijkstra_q"] = min(dijkstra_q)
            dijkstra_table["max_dijkstra_q"] = max(dijkstra_q)
            dijkstra_table["avg_dijkstra_v"] = sum(dijkstra_v) / len(
                dijkstra_v)
            dijkstra_table["min_dijkstra_v"] = min(dijkstra_v)
            dijkstra_table["max_dijkstra_v"] = max(dijkstra_v)
            dijkstra_table["avg_dijkstra_t"] = sum(dijkstra_t) / len(
                dijkstra_t)
            dijkstra_table["min_dijkstra_t"] = min(dijkstra_t)
            dijkstra_table["max_dijkstra_t"] = max(dijkstra_t)

        # Unf Astar both finished
        if len(t_astar_unf_astar_complete) > 0:
            unf_astar_table["samples"] = len(q_unf_unf_astar_complete)
            unf_astar_table["avg_trace_length"] = sum(
                tl_unf_astar_complete) / len(tl_unf_astar_complete)
            unf_astar_table["min_trace_length"] = min(tl_unf_astar_complete)
            unf_astar_table["max_trace_length"] = max(tl_unf_astar_complete)

            unf_astar_table["avg_unf_q"] = sum(q_unf_unf_astar_complete) / len(
                q_unf_unf_astar_complete)
            unf_astar_table["min_unf_q"] = min(q_unf_unf_astar_complete)
            unf_astar_table["max_unf_q"] = max(q_unf_unf_astar_complete)
            unf_astar_table["avg_unf_v"] = sum(v_unf_unf_astar_complete) / len(
                t_unf_unf_astar_complete)
            unf_astar_table["min_unf_v"] = min(v_unf_unf_astar_complete)
            unf_astar_table["max_unf_v"] = max(v_unf_unf_astar_complete)
            unf_astar_table["avg_unf_t"] = sum(t_unf_unf_astar_complete) / len(
                t_unf_unf_astar_complete)
            unf_astar_table["min_unf_t"] = min(t_unf_unf_astar_complete)
            unf_astar_table["max_unf_t"] = max(t_unf_unf_astar_complete)

            unf_astar_table["avg_astar_q"] = sum(
                q_astar_unf_astar_complete) / len(q_astar_unf_astar_complete)
            unf_astar_table["min_astar_q"] = min(q_astar_unf_astar_complete)
            unf_astar_table["max_astar_q"] = max(q_astar_unf_astar_complete)
            unf_astar_table["avg_astar_v"] = sum(
                v_astar_unf_astar_complete) / len(t_astar_unf_astar_complete)
            unf_astar_table["min_astar_v"] = min(v_astar_unf_astar_complete)
            unf_astar_table["max_astar_v"] = max(v_astar_unf_astar_complete)
            unf_astar_table["avg_astar_t"] = sum(
                t_astar_unf_astar_complete) / len(t_astar_unf_astar_complete)
            unf_astar_table["min_astar_t"] = min(t_astar_unf_astar_complete)
            unf_astar_table["max_astar_t"] = max(t_astar_unf_astar_complete)

        if len(t_unf_unf_dijkstra_complete) > 0:
            # Unf Dijkstra both finished
            unf_dijkstra_table["samples"] = len(q_unf_unf_dijkstra_complete)
            unf_dijkstra_table["avg_trace_length"] = sum(
                tl_unf_dijkstra_complete) / len(tl_unf_dijkstra_complete)
            unf_dijkstra_table["min_trace_length"] = min(
                tl_unf_dijkstra_complete)
            unf_dijkstra_table["max_trace_length"] = max(
                tl_unf_dijkstra_complete)

            unf_dijkstra_table["avg_unf_q"] = sum(
                q_unf_unf_dijkstra_complete) / len(q_unf_unf_dijkstra_complete)
            unf_dijkstra_table["min_unf_q"] = min(q_unf_unf_dijkstra_complete)
            unf_dijkstra_table["max_unf_q"] = max(q_unf_unf_dijkstra_complete)
            unf_dijkstra_table["avg_unf_v"] = sum(
                v_unf_unf_dijkstra_complete) / len(t_unf_unf_dijkstra_complete)
            unf_dijkstra_table["min_unf_v"] = min(v_unf_unf_dijkstra_complete)
            unf_dijkstra_table["max_unf_v"] = max(v_unf_unf_dijkstra_complete)
            unf_dijkstra_table["avg_unf_t"] = sum(
                t_unf_unf_dijkstra_complete) / len(t_unf_unf_dijkstra_complete)
            unf_dijkstra_table["min_unf_t"] = min(t_unf_unf_dijkstra_complete)
            unf_dijkstra_table["max_unf_t"] = max(t_unf_unf_dijkstra_complete)

            unf_dijkstra_table["avg_dijkstra_q"] = sum(
                q_dijkstra_unf_dijkstra_complete) / len(
                    q_dijkstra_unf_dijkstra_complete)
            unf_dijkstra_table["min_dijkstra_q"] = min(
                q_dijkstra_unf_dijkstra_complete)
            unf_dijkstra_table["max_dijkstra_q"] = max(
                q_dijkstra_unf_dijkstra_complete)
            unf_dijkstra_table["avg_dijkstra_v"] = sum(
                v_dijkstra_unf_dijkstra_complete) / len(
                    t_dijkstra_unf_dijkstra_complete)
            unf_dijkstra_table["min_dijkstra_v"] = min(
                v_dijkstra_unf_dijkstra_complete)
            unf_dijkstra_table["max_dijkstra_v"] = max(
                v_dijkstra_unf_dijkstra_complete)
            unf_dijkstra_table["avg_dijkstra_t"] = sum(
                t_dijkstra_unf_dijkstra_complete) / len(
                    t_dijkstra_unf_dijkstra_complete)
            unf_dijkstra_table["min_dijkstra_t"] = min(
                t_dijkstra_unf_dijkstra_complete)
            unf_dijkstra_table["max_dijkstra_t"] = max(
                t_dijkstra_unf_dijkstra_complete)
        print(res_file_path)
        print(unf_table)
        print(astar_table)
        print(dijkstra_table)
        print(unf_astar_table)
        print(unf_dijkstra_table)
    print(means_median_pairs)
    # mean_median_diffs = [((x[1] - x[0]) / x[0]) for x in means_median_pairs]
    # print(mean_median_diffs)


def e4_table():
    res_file_path_title_pairs = [
        ("results/sepsis/sepsis_05.json",
         "results/streaming/sepsis/sepsis_05.json", "SP"),
        ("results/bpic17/bpic17_02.json",
         "results/streaming/bpic17/bpic17_02.json", "BPIC17"),
        ("results/bpic19/bpic19_02.json",
         "results/streaming/bpic19/bpic19_02.json", "BPIC19"),
        ("results/inthelarge/prAm6.json",
         "results/streaming/inthelarge/prAm6.json", "ITL prAm6"),
        ("results/inthelarge/prBm6.json",
         "results/streaming/inthelarge/prBm6.json", "ITL prBm6"),
        ("results/inthelarge/prCm6.json",
         "results/streaming/inthelarge/prCm6.json", "ITL prCm6"),
        ("results/inthelarge/prDm6.json",
         "results/streaming/inthelarge/prDm6.json", "ITL prDm6"),
        ("results/inthelarge/prEm6.json",
         "results/streaming/inthelarge/prEm6.json", "ITL prEm6"),
        ("results/inthelarge/prFm6.json",
         "results/streaming/inthelarge/prFm6.json", "ITL prFm6"),
        # ("results/inthelarge/prGm6.json",
        #  "results/streaming/inthelarge/prGm6.json", "ITL prGm6"),
        ("results/hospital_billing/hospital_billing_02.json",
         "results/streaming/hospital_billing/hospital_billing_02.json", "HB"),
        ("results/bpic12/bpic12_02.json",
         "results/streaming/bpic12/bpic12_02.json", "BPIC12"),
        ("results/bpic13/bpic13_02.json",
         "results/streaming/bpic13/bpic13_02.json", "BPIC13"),
        ("results/traffic/traffic_02.json",
         "results/streaming/traffic/traffic_02.json", "TR"),
        ("results/bpic13/bpic13_split.json",
         "results/streaming/bpic13/bpic13_split.json", "BPIC13_SPLIT"),
        ("results/sepsis/sepsis_split.json",
         "results/streaming/sepsis/sepsis_split.json", "SP_SPLIT"),
        ("results/bpic12/bpic12_split.json",
         "results/streaming/bpic12/bpic12_split.json", "BPIC12_SPLIT"),
        ("results/bpic17/bpic17_split.json",
         "results/streaming/bpic17/bpic17_split.json", "BPIC17_SPLIT"),
        ("results/hospital_billing/hospital_billing_split.json",
         "results/streaming/hospital_billing/hospital_billing_split.json",
         "HB_SPLIT"),
        ("results/traffic/traffic_split.json",
         "results/streaming/traffic/traffic_split.json", "TR_SPLIT"),
    ]
    for res_file_path_offline, res_file_path_online, title in res_file_path_title_pairs:
        with open(res_file_path_offline) as rf:
            offline = json.load(rf)
        with open(res_file_path_online) as rf:
            online = json.load(rf)
        on_q = []
        on_v = []
        on_t = []

        q_off_complete = []
        v_off_complete = []
        t_off_complete = []

        q_on_complete = []
        v_on_complete = []
        t_on_complete = []

        for i in range(len(online)):
            entry_off = offline[i]
            entry_on = online[i]
            if entry_on["unf_elapsed_time"] != -1:
                on_q.append(entry_on["unf_q"])
                on_v.append(entry_on["unf_v"])
                on_t.append(entry_on["unf_elapsed_time"])
                # if entry_off["trace_length"] == entry_on["trace_length"]:
                if entry_on["unf_elapsed_time"] != -1 and entry_off[
                        "unf_elapsed_time"] != -1:
                    q_off_complete.append(entry_off["unf_q"])
                    v_off_complete.append(entry_off["unf_v"])
                    t_off_complete.append(entry_off["unf_elapsed_time"])

                    q_on_complete.append(entry_on["unf_q"])
                    v_on_complete.append(entry_on["unf_v"])
                    t_on_complete.append(entry_on["unf_elapsed_time"])
        # # Online
        # if len(on_t) > 0:
        #     print(f"online {title} samples: {len(on_t)}")
        #     print(
        #         f"on q {title}, {sum(on_q)/len(on_q)}, {min(on_q)}, {max(on_q)}"
        #     )
        #     print(
        #         f"on v {title}, {sum(on_v)/len(on_v)}, {min(on_v)}, {max(on_v)}"
        #     )
        #     print(
        #         f"on t {title}, {sum(on_t)/len(on_t)}, {min(on_t)}, {max(on_t)}"
        #     )

        # Both finished
        if len(t_on_complete) > 0:
            print(f"both {title} samples: {len(q_off_complete)}")
            print(
                f"off q {title}, {sum(q_off_complete)/len(q_off_complete)}, {min(q_off_complete)}, {max(q_off_complete)}"
            )
            print(
                f"off v {title}, {sum(v_off_complete)/len(v_off_complete)}, {min(v_off_complete)}, {max(v_off_complete)}"
            )
            print(
                f"off t {title}, {sum(t_off_complete)/len(t_off_complete)}, {min(t_off_complete)}, {max(t_off_complete)}"
            )
            print(
                f"on q {title}, {sum(q_on_complete)/len(q_on_complete)}, {min(q_on_complete)}, {max(q_on_complete)}"
            )
            print(
                f"on v {title}, {sum(v_on_complete)/len(v_on_complete)}, {min(v_on_complete)}, {max(v_on_complete)}"
            )
            print(
                f"on t {title}, {sum(t_on_complete)/len(t_on_complete)}, {min(t_on_complete)}, {max(t_on_complete)}"
            )


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
        # ("results/sepsis/sepsis_05.json",
        #  "data/sepsis/Sepsis Cases - Event Log.xes", 1.2307692307692308,
        #  1.391304347826087),
        # ("results/bpic17/bpic17_02.json", "data/bpic17/BPI Challenge 2017.xes",
        #  1.0555555555555556, 1.6521739130434783),
        # ("results/bpic19/bpic19_02.json", "data/bpic19/BPI_Challenge_2019.xes",
        #  1.1123595505617978, 1.8679245283018868),
        # ("results/inthelarge/prAm6.json", "data/inthelarge/prAm6.tpn",
        #  1.159779614325069, 1.2132564841498559),
        # ("results/inthelarge/prBm6.json", "data/inthelarge/prBm6.tpn",
        #  1.1798107255520505, 1.1798107255520505),
        # ("results/inthelarge/prCm6.json", "data/inthelarge/prCm6.tpn",
        #  1.1798107255520505, 1.1798107255520505),
        # ("results/inthelarge/prDm6.json", "data/inthelarge/prDm6.tpn",
        #  1.324009324009324, 1.0737240075614367),
        # ("results/inthelarge/prEm6.json", "data/inthelarge/prEm6.tpn",
        #  1.1781818181818182, 1.1696750902527075),
        # ("results/inthelarge/prFm6.json", "data/inthelarge/prFm6.tpn",
        #  1.2842809364548495, 1.0607734806629834),
        # ("results/hospital_billing/hospital_billing_02.json",
        #  "data/hospital_billing/Hospital Billing - Event Log.xes",
        #  1.0178571428571428, 1.78125),
        # ("results/bpic12/bpic12_02.json", "data/bpic12/BPI_Challenge_2012.xes",
        #  1.09375, 1.7073170731707317),
        # ("results/bpic13/bpic13_02.json",
        #  "data/bpic13/BPI_Challenge_2013_incidents.xes", 1.0833333333333333,
        #  1.3),
        ("results/traffic/traffic_02.json",
         "data/traffic/Road_Traffic_Fine_Management_Process.xes", 0, 0)
    ]
    colorss = [
        "red", "green", "blue", "pink", "purple", "magenta", "turquoise",
        "cyan", "yellow", "orange", "gray", "chocolate", "black"
    ]
    color_index = 0
    fig, ax = plot_2d("visited states", "elapsed time (s)", "unfolding")
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

    for res_file, model_file, exc_factor, conc_factor in dir_pairs:
        if model_file.endswith(".tpn"):
            model_net, model_im, model_fm = ptn.import_from_tpn(model_file)
        else:
            xes_df = pm4py.read_xes(model_file)
            noise_threshold = 0.5 if "sepsis" in model_file else 0.2
            model_net, model_im, model_fm = pm4py.discover_petri_net_inductive(
                xes_df, noise_threshold=noise_threshold)
            # pm4py.view_petri_net(model_net, model_im, model_fm)
        avg_t_outgoing_arc = sum(
            [len(t.out_arcs)
             for t in model_net.transitions]) / len(model_net.transitions)
        avg_p_outgoing_arc = sum([len(p.out_arcs) for p in model_net.places
                                  ]) / len(model_net.places)
        print(f"{model_file}, {avg_t_outgoing_arc}, {avg_p_outgoing_arc}")
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
                subx.append(t["astar_fitness"])
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
    # sizes = [x * 25 for x in sizes]
    # alphas=[x if x<= 1 else 1 for x in alphas]
    # times = [x*100 for x in times]
    # times = np.emath.logn(2, times)
    x = np.log(x)
    y = np.log(y)
    # colors=[[1,0,0,x] for x in alphas]
    ax.scatter(x, y, c=colors)
    # ax.set_xscale("log")
    # ax.set_yscale("log")
    plt.show()


def regress_a_priori():
    dir_pairs = [
        ("results/sepsis/sepsis_05.json", 1.2307692307692308,
         1.391304347826087),
        ("results/bpic17/bpic17_02.json", 1.0555555555555556,
         1.6521739130434783),
        ("results/bpic19/bpic19_02.json", 1.1123595505617978,
         1.8679245283018868),
        ("results/inthelarge/prAm6.json", 1.159779614325069,
         1.2132564841498559),
        ("results/inthelarge/prBm6.json", 1.1798107255520505,
         1.1798107255520505),
        ("results/inthelarge/prCm6.json", 1.1798107255520505,
         1.1798107255520505),
        ("results/inthelarge/prDm6.json", 1.324009324009324,
         1.0737240075614367),
        ("results/inthelarge/prEm6.json", 1.1781818181818182,
         1.1696750902527075),
        ("results/inthelarge/prFm6.json", 1.2842809364548495,
         1.0607734806629834),
        ("results/hospital_billing/hospital_billing_02.json",
         1.0178571428571428, 1.78125),
        ("results/bpic12/bpic12_02.json", 1.09375, 1.7073170731707317),
        ("results/bpic13/bpic13_02.json", 1.0833333333333333, 1.3),
        ("results/traffic/traffic_02.json", 1.3043478260869565,
         1.3043478260869565),
        ("results/bpic13/bpic13_split.json", 1.1666666666666667, 1.0),
        ("results/sepsis/sepsis_split.json", 1.125, 1.35),
        ("results/bpic12/bpic12_split.json", 1.0, 1.5217391304347827),
        ("results/bpic17/bpic17_split.json", 1.0, 1.3571428571428572),
        ("results/hospital_billing/hospital_billing_split.json",
         1.0333333333333334, 1.4090909090909092),
        ("results/traffic/traffic_split.json", 1.0588235294117647, 1.2)
    ]
    x = []
    y = []
    for res_file, conc_factor, exc_factor in dir_pairs:
        # x = []
        # y = []
        # print(res_file)
        with open(res_file) as rf:
            contents = json.load(rf)

        for t in contents:
            if t["unf_elapsed_time"] != -1:
                sync_prod_size = t["sync_net_transitions"]
                tl = t["trace_length"]
                model_size = t["model_net_transitions"]

                indep = [
                    # tl,
                    sync_prod_size,
                    # model_size,
                    exc_factor,
                    conc_factor,
                ]

                x.append(indep)
                y.append(np.log(t["unf_elapsed_time"]))

    # print(stats.spearmanr(x))

    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    # x, y, random_state=2)
    m = SMWrapper(sm.OLS)

    # Scale each feature
    # Use regular scale instead of robust scale because we don't have "true" outliers because our outliers aren't by random chance
    scaler = sklearn.preprocessing.StandardScaler()

    # Pipeline to not get dataleaks
    pipeline = sklearn.pipeline.Pipeline([
        ("scaler", scaler),
        ("model", m),
    ])

    pipeline.fit(x, y)
    score = pipeline.score(x, y)
    print(score)
    # print(pipeline["model"].results_.summary())
    for tab in pipeline["model"].results_.summary().tables:
        print(tab.as_latex_tabular())
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
    fig.suptitle("Plotting predictions")
    plt.tight_layout()
    plt.show()


def regress():
    dir_pairs = [
        ("results/sepsis/sepsis_05.json", 1.2307692307692308,
         1.391304347826087),
        ("results/bpic17/bpic17_02.json", 1.0555555555555556,
         1.6521739130434783),
        ("results/bpic19/bpic19_02.json", 1.1123595505617978,
         1.8679245283018868),
        ("results/inthelarge/prAm6.json", 1.159779614325069,
         1.2132564841498559),
        ("results/inthelarge/prBm6.json", 1.1798107255520505,
         1.1798107255520505),
        ("results/inthelarge/prCm6.json", 1.1798107255520505,
         1.1798107255520505),
        ("results/inthelarge/prDm6.json", 1.324009324009324,
         1.0737240075614367),
        ("results/inthelarge/prEm6.json", 1.1781818181818182,
         1.1696750902527075),
        ("results/inthelarge/prFm6.json", 1.2842809364548495,
         1.0607734806629834),
        ("results/hospital_billing/hospital_billing_02.json",
         1.0178571428571428, 1.78125),
        ("results/bpic12/bpic12_02.json", 1.09375, 1.7073170731707317),
        ("results/bpic13/bpic13_02.json", 1.0833333333333333, 1.3),
        ("results/traffic/traffic_02.json", 1.3043478260869565,
         1.3043478260869565),
        ("results/bpic13/bpic13_split.json", 1.1666666666666667, 1.0),
        ("results/sepsis/sepsis_split.json", 1.125, 1.35),
        ("results/bpic12/bpic12_split.json", 1.0, 1.5217391304347827),
        ("results/bpic17/bpic17_split.json", 1.0, 1.3571428571428572),
        ("results/hospital_billing/hospital_billing_split.json",
         1.0333333333333334, 1.4090909090909092),
        ("results/traffic/traffic_split.json", 1.0588235294117647, 1.2)
    ]
    x = []
    y = []
    for res_file, conc_factor, exc_factor in dir_pairs:
        # print(res_file)
        with open(res_file) as rf:
            contents = json.load(rf)

        for t in contents:
            if t["unf_elapsed_time"] != -1:
                sync_prod_size = t["sync_net_transitions"]
                tl = t["trace_length"]
                fit = t["astar_fitness"]
                cost = t["unf_cost"]
                v = t["unf_v"]
                q = t["unf_q"]

                indep = [
                    # tl,
                    sync_prod_size,
                    # exc_factor,
                    # conc_factor,
                    np.log(v),
                    np.log(q),
                    # fit,
                    # cost,
                ]

                x.append(indep)
                y.append(np.log(t["unf_elapsed_time"]))
    # print(stats.pearsonr(x))

    # x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(
    # x, y, random_state=2)

    m = SMWrapper(sm.OLS)

    # Scale each feature
    # Use regular scale instead of robust scale because we don't have "true" outliers because our outliers aren't by random chance
    scaler = sklearn.preprocessing.StandardScaler()

    # Pipeline to not get dataleaks
    pipeline = sklearn.pipeline.Pipeline([
        ("scaler", scaler),
        ("model", m),
    ])

    # selector = sklearn.feature_selection.RFE(
    #     pipeline,
    #     n_features_to_select=5,
    # )
    # selector.fit(x, y)
    # score = selector.score(x, y)
    # print(score)
    # print(selector["model"].results_.summary())
    pipeline.fit(x, y)
    score = pipeline.score(x, y)
    print(score)
    # print(pipeline["model"].results_.summary())
    for tab in pipeline["model"].results_.summary().tables:
        print(tab.as_latex_tabular())

    # print(pipeline["model"].results_.summary())
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


def fix_sp_count_split():
    dir_pairs = [
        ("results/bpic13/bpic13_split.json", "data/bpic13/bpic13_split.bpmn",
         "data/bpic13/BPI_Challenge_2013_incidents.xes"),
        ("results/sepsis/sepsis_split.json", "data/sepsis/sepsis_split.bpmn",
         "data/sepsis/Sepsis Cases - Event Log.xes"),
        ("results/bpic12/bpic12_split.json", "data/bpic12/bpic12_split.bpmn",
         "data/bpic12/BPI_Challenge_2012.xes"),
        ("results/bpic17/bpic17_split.json", "data/bpic17/bpic17_split.bpmn",
         "data/bpic17/BPI Challenge 2017.xes"),
        ("results/hospital_billing/hospital_billing_split.json",
         "data/hospital_billing/hospital_billing_split.bpmn",
         "data/hospital_billing/Hospital Billing - Event Log.xes"),
        ("results/traffic/traffic_split.json",
         "data/traffic/traffic_split.bpmn",
         "data/traffic/Road_Traffic_Fine_Management_Process.xes"),
    ]
    for res_file, model_file, data_file in dir_pairs:
        xes_df = pm4py.read_xes(data_file)
        model_net, model_im, model_fm = pm4py.convert_to_petri_net(
            pm4py.read_bpmn(model_file))
        xes_el = pm4py.convert_to_event_log(pm4py.format_dataframe(xes_df))

        if "bpic" in model_file or "hospital" in model_file or "traffic" in model_file:
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

            sync_net, sync_im, sync_fm, cost_function = ptn.construct_synchronous_product(
                model_net, model_im, model_fm, trace_net, trace_net_im,
                trace_net_fm)
            file_data[i]["sync_net_transitions"] = len(sync_net.transitions)
            file_data[i]["sync_net_places"] = len(sync_net.places)
            file_data[i]["model_net_transitions"] = len(model_net.transitions)
        with open(res_file, "w") as wf:
            rstr = json.dumps(file_data, indent=4)
            wf.write(rstr)


def fix_sp_count():
    # pass
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
        ("results/inthelarge/prGm6.json", "data/inthelarge/prGm6.tpn"),
        ("results/hospital_billing/hospital_billing_02.json",
         "data/hospital_billing/Hospital Billing - Event Log.xes"),
        ("results/bpic12/bpic12_02.json",
         "data/bpic12/BPI_Challenge_2012.xes"),
        ("results/bpic13/bpic13_02.json",
         "data/bpic13/BPI_Challenge_2013_incidents.xes"),
        ("results/traffic/traffic_02.json",
         "data/traffic/Road_Traffic_Fine_Management_Process.xes"),
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

        if "bpic" in model_file or "hospital" in model_file or "traffic" in model_file:
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

            sync_net, sync_im, sync_fm, cost_function = ptn.construct_synchronous_product(
                model_net, model_im, model_fm, trace_net, trace_net_im,
                trace_net_fm)
            file_data[i]["sync_net_transitions"] = len(sync_net.transitions)
            file_data[i]["sync_net_places"] = len(sync_net.places)
            file_data[i]["model_net_transitions"] = len(model_net.transitions)
        with open(res_file, "w") as wf:
            rstr = json.dumps(file_data, indent=4)
            wf.write(rstr)


def factors():
    datasets = [
        ("results/bpic13/bpic13_split.json", "data/bpic13/bpic13_split.bpmn",
         1.1666666666666667, 1.0),
        ("results/sepsis/sepsis_split.json", "data/sepsis/sepsis_split.bpmn",
         1.125, 1.35),
        ("results/bpic12/bpic12_split.json", "data/bpic12/bpic12_split.bpmn",
         1.0, 1.5217391304347827),
        ("results/bpic17/bpic17_split.json", "data/bpic17/bpic17_split.bpmn",
         1.0, 1.3571428571428572),
        ("results/hospital_billing/hospital_billing_split.json",
         "data/hospital_billing/hospital_billing_split.bpmn",
         1.0333333333333334, 1.4090909090909092),
        ("results/traffic/traffic_split.json",
         "data/traffic/traffic_split.bpmn", 1.0588235294117647, 1.2),
    ]
    for _, dataset in datasets:
        model_net, i, f = pm4py.convert_to_petri_net(pm4py.read_bpmn(dataset))
        avg_t_outgoing_arc = sum(
            [len(t.out_arcs)
             for t in model_net.transitions]) / len(model_net.transitions)
        avg_p_outgoing_arc = sum([len(p.out_arcs) for p in model_net.places
                                  ]) / len(model_net.places)
        print(f"{dataset}, {avg_t_outgoing_arc}, {avg_p_outgoing_arc}")


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
    dir_name_color_title_pairs_no_heuristic = [
        (CONCURRENT_RESULT_DIR_NO_HEURISTIC, CONCURRENT_MODEL_DIR, "pink",
         "C"),
        (CONCURRENT_CONCURRENT_NESTED_RESULT_DIR_NO_HEURISTIC,
         CONCURRENT_CONCURRENT_NESTED_MODEL_DIR, "red", "CN"),
        (EXCLUSIVE_RESULT_DIR_NO_HEURISTIC, EXCLUSIVE_MODEL_DIR, "turquoise",
         "E"),
        (EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR_NO_HEURISTIC,
         EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR, "blue", "EN"),
        (LOOP_RESULT_DIR_NO_HEURISTIC, LOOP_MODEL_DIR, "green", "L")
    ]
    e2_figures(dir_name_color_title_pairs_no_heuristic, "no_heuristic")

    # Heuristics stage one
    dir_name_color_title_pairs_heuristic = [
        (CONCURRENT_RESULT_DIR, CONCURRENT_MODEL_DIR, "pink", "C"),
        (CONCURRENT_CONCURRENT_NESTED_RESULT_DIR,
         CONCURRENT_CONCURRENT_NESTED_MODEL_DIR, "red", "CN"),
        (EXCLUSIVE_RESULT_DIR, EXCLUSIVE_MODEL_DIR, "turquoise", "E"),
        (EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR,
         EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR, "blue", "EN"),
        (LOOP_RESULT_DIR, LOOP_MODEL_DIR, "green", "L")
    ]
    e2_figures(dir_name_color_title_pairs_heuristic, "heuristic")
