import json
import cProfile
from coset_standard import *
from coset_streaming import *
import os
from settings import *
import time
from tqdm import tqdm
from enum import Enum
import random
import re
from functools import cmp_to_key
import signal


class TimeoutException(Exception):  # Custom exception class
    pass


def timeout_handler(signum, frame):  # Custom signal handler
    raise TimeoutException


class DeviationsOperations(Enum):
    REMOVE = 1
    REPLACE_WITH_INVERSE_PERCENTILE = 2


def print_stats(filepath):
    with open(filepath) as f:
        stats = json.load(f)
        # print(f"Avg trad q: {sum(stats["trad_q"])/len(stats["trad_q"])}")
        # print(f"Avg trad v: {sum(stats["trad_v"])/len(stats["trad_v"])}")
        # print(f"Avg unf q: {sum(stats["unf_q"])/len(stats["unf_q"])}")
        # print(f"Avg unf v: {sum(stats["unf_v"])/len(stats["unf_v"])}")


# Types of deviations for concurrent models: mandatory activity missing in 1 branch, multiple branches, multiple missing, different places


def view_model(data_filepath, model_filepath):
    pt = pm4py.read_ptml(model_filepath)
    model_net, model_im, model_fm = pm4py.convert_to_petri_net(pt)
    view_petri_net(model_net)
    return


def apply_trace_deviation(location_percentiles: list[float], operation, trace):
    trace_copy = deepcopy(trace)
    for percentile in location_percentiles:
        index = int((len(trace_copy._list) - 1) * percentile)
        if operation == DeviationsOperations.REMOVE.name:
            trace_copy._list.remove(trace_copy._list[index])
        if operation == DeviationsOperations.REPLACE_WITH_INVERSE_PERCENTILE.name:
            index_inverse = len(trace_copy._list) * (1 - percentile)
            trace_copy._list[index]["concept:name"] = trace_copy._list[
                index_inverse]["concept:name"]
    return trace_copy


def run_dijkstra_with_timer(model_net, model_im, model_fm, trace):
    dijkstra_start_time = time.time()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(100)
    try:
        dijkstra_alignment = conformance_diagnostics_alignments(
            trace,
            model_net,
            model_im,
            model_fm,
            variant_str="Variants.VERSION_DIJKSTRA_NO_HEURISTICS")
    except TimeoutException:
        return -1, -1, -1, -1, -1
    signal.alarm(0)

    dijkstra_elapsed_time = time.time() - dijkstra_start_time

    return dijkstra_elapsed_time, dijkstra_alignment[
        "queued_states"], dijkstra_alignment[
            "visited_states"], dijkstra_alignment["cost"], dijkstra_alignment[
                "fitness"]


def run_astar_with_timer(model_net, model_im, model_fm, trace):
    astar_start_time = time.time()
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(100)
    try:
        astar_alignment = conformance_diagnostics_alignments(
            trace, model_net, model_im, model_fm)
    except TimeoutException:
        return -1, -1, -1, -1, -1
    signal.alarm(0)

    astar_elapsed_time = time.time() - astar_start_time

    return astar_elapsed_time, astar_alignment[
        "queued_states"], astar_alignment["visited_states"], astar_alignment[
            "cost"], astar_alignment["fitness"]


def run_streaming(model_net, model_im, model_fm, trace):
    starting_trace = trace.__copy__()
    starting_trace._list = starting_trace._list[:1]
    stream = trace.__copy__()
    stream._list = stream._list[1:]

    # Add this event onto our bp
    start_time = time.time()
    trace_net, trace_net_im, trace_net_fm = construct_trace_net(
        starting_trace, "concept:name", "concept:name")
    sync_net, sync_im, sync_fm, cost_function = construct_synchronous_product(
        model_net, model_im, model_fm, trace_net, trace_net_im, trace_net_fm)

    extended_net = ExtendedSyncNetStreaming(sync_net, sync_im, sync_fm,
                                            trace_net_fm, cost_function)

    bp = BranchingProcessStream(extended_net, starting_trace)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(100)
    try:
        final_alignment, cost = bp.alignment_streaming(model_net, model_im,
                                                       model_fm, stream)
    except TimeoutException:
        return -1, bp.possible_extensions._queued, bp.possible_extensions._visited, -1, bp.unf_q_per_iteration, bp.unf_v_per_iteration

    signal.alarm(0)
    elapsed_time = time.time() - start_time
    return elapsed_time, bp.possible_extensions._queued, bp.possible_extensions._visited, cost, bp.unf_q_per_iteration, bp.unf_v_per_iteration


def run_offline(model_net, model_im, model_fm, trace):
    start_time = time.time()
    trace_net, trace_net_im, trace_net_fm = construct_trace_net(
        trace, "concept:name", "concept:name")

    sync_net, sync_im, sync_fm, cost_function = construct_synchronous_product(
        model_net, model_im, model_fm, trace_net, trace_net_im, trace_net_fm)

    sync_net_extended = ExtendedSyncNet(sync_net, sync_im, sync_fm,
                                        cost_function)

    bp = BranchingProcessStandard(sync_net_extended)

    bp.initialise_from_initial_marking()

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(100)
    try:
        alignment, cost = bp.astar()
    except TimeoutException:
        return -1, bp.possible_extensions._queued, bp.possible_extensions._visited, -1

    signal.alarm(0)
    elapsed_time = time.time() - start_time
    return elapsed_time, bp.possible_extensions._queued, bp.possible_extensions._visited, cost


def run_offline_remove_start_halfway_end(data_filepath, model_net, model_im,
                                         model_fm):
    results = []
    xes_df = pm4py.read_xes(data_filepath)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    for trace in tqdm(xes_el):
        if len(trace) > 3:
            deviating_trace = apply_trace_deviation(
                [0, 0.5, 1], DeviationsOperations.REMOVE.name, trace)
            elapsed_time, q, v, cost = run_offline(model_net, model_im,
                                                   model_fm, deviating_trace)
            trace_result = {}
            trace_result["elapsed_time"] = elapsed_time
            trace_result["q"] = q
            trace_result["v"] = v
            trace_result["cost"] = cost
            trace_result["trace_length"] = len(trace)
            results.append(trace_result)
        else:
            trace_result = {}
            trace_result["elapsed_time"] = -2
            trace_result["q"] = "NaN"
            trace_result["v"] = "NaN"
            trace_result["cost"] = "NaN"
            trace_result["trace_length"] = len(trace)
            results.append(trace_result)

    return results


def run_offline_remove_halfway(data_filepath, model_net, model_im, model_fm):
    results = []
    xes_df = pm4py.read_xes(data_filepath)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    for trace in tqdm(xes_el):
        deviating_trace = apply_trace_deviation(
            [0.5], DeviationsOperations.REMOVE.name, trace)
        elapsed_time, q, v, cost = run_offline(model_net, model_im, model_fm,
                                               deviating_trace)
        trace_result = {}
        trace_result["elapsed_time"] = elapsed_time
        trace_result["q"] = q
        trace_result["v"] = v
        trace_result["cost"] = cost
        trace_result["trace_length"] = len(trace)
        results.append(trace_result)

    return results


def run_offline_remove_end(data_filepath, model_net, model_im, model_fm):
    results = []
    xes_df = pm4py.read_xes(data_filepath)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    for trace in tqdm(xes_el):
        deviating_trace = apply_trace_deviation(
            [1], DeviationsOperations.REMOVE.name, trace)
        elapsed_time, q, v, cost = run_offline(model_net, model_im, model_fm,
                                               deviating_trace)
        trace_result = {}
        trace_result["elapsed_time"] = elapsed_time
        trace_result["q"] = q
        trace_result["v"] = v
        trace_result["cost"] = cost
        trace_result["trace_length"] = len(trace)
        results.append(trace_result)

    return results


def run_offline_remove_start(data_filepath, model_net, model_im, model_fm):
    results = []
    xes_df = pm4py.read_xes(data_filepath)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    for trace in tqdm(xes_el):
        deviating_trace = apply_trace_deviation(
            [0], DeviationsOperations.REMOVE.name, trace)
        elapsed_time, q, v, cost = run_offline(model_net, model_im, model_fm,
                                               deviating_trace)
        trace_result = {}
        trace_result["elapsed_time"] = elapsed_time
        trace_result["q"] = q
        trace_result["v"] = v
        trace_result["cost"] = cost
        trace_result["trace_length"] = len(trace)
        results.append(trace_result)

    return results


def run_offline_no_deviations(data_filepath, model_net, model_im, model_fm):
    results = []
    xes_df = pm4py.read_xes(data_filepath)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    for trace in tqdm(xes_el):
        elapsed_time, q, v, cost = run_offline(model_net, model_im, model_fm,
                                               trace)
        trace_result = {}
        trace_result["elapsed_time"] = elapsed_time
        trace_result["q"] = q
        trace_result["v"] = v
        trace_result["cost"] = cost
        trace_result["trace_length"] = len(trace)
        results.append(trace_result)

    return results


def preliminary():
    xes_df = pm4py.read_xes("data/Sepsis Cases - Event Log.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.5)
    view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    rnd = random.sample(xes_el, 10)
    for trace in tqdm(rnd):
        trace_result = {}

        # dijkstra_start_time = time.time()
        # dijkstra_alignment = conformance_diagnostics_alignments(
        #         trace,
        #         model_net,
        #         model_im,
        #         model_fm,
        #         variant_str="Variants.VERSION_DIJKSTRA_NO_HEURISTICS")
        # dijkstra_elapsed_time = time.time() -dijkstra_start_time

        # print(dijkstra_alignment)

        astar_start_time = time.time()
        astar_alignment = conformance_diagnostics_alignments(
            trace, model_net, model_im, model_fm)
        astar_elapsed_time = time.time() - astar_start_time

        print(astar_alignment)

        elapsed_time, q, v, cost = run_offline(model_net, model_im, model_fm,
                                               trace)
        trace_result["unf_elapsed_time"] = elapsed_time
        trace_result["unf_q"] = q
        trace_result["unf_v"] = v
        trace_result["unf_cost"] = cost

        # trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        # trace_result["dijkstra_q"] = dijkstra_alignment["queued_states"]
        # trace_result["dijkstra_v"] = dijkstra_alignment["visited_states"]
        # trace_result["dijkstra_cost"] = dijkstra_alignment["cost"]

        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_alignment["queued_states"]
        trace_result["astar_v"] = astar_alignment["visited_states"]
        trace_result["astar_cost"] = astar_alignment["cost"]
        results.append(trace_result)
    with open(f"preliminary_sepsis_05_noise.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def compare_by_breadth_depth(filename1, filename2):
    split1 = filename1.split("_")
    b1 = int(re.search(r"\d+", split1[0]).group())
    d1 = int(re.search(r"\d+", split1[1]).group())

    split2 = filename2.split("_")
    b2 = int(re.search(r"\d+", split2[0]).group())
    d2 = int(re.search(r"\d+", split2[1]).group())
    if b1 < b2:
        return -1
    if b1 > b2:
        return 1
    if b2 == b1:
        if d1 < d2:
            return -1
        if d1 > d2:
            return 1
    return 0


def run_artificial_model_offline(model_dir,
                                 result_dir,
                                 profile_cpu=False,
                                 profile_memory=False):
    files_in_dir = os.listdir(model_dir)
    files_in_dir.sort(key=cmp_to_key(compare_by_breadth_depth))
    for f in files_in_dir:
        if f.endswith(".ptml"):
            continue
        filename = f.removesuffix(".xes")
        split1 = filename.split("_")
        b1 = int(re.search(r"\d+", split1[0]).group())
        d1 = int(re.search(r"\d+", split1[1]).group())
        if model_dir == CONCURRENT_MODEL_DIR:
            if b1 > 12:
                continue
        ptml = filename + ".ptml"
        xes = filename + ".xes"
        pt = pm4py.read_ptml(f"{model_dir}/{ptml}")
        model_net, model_im, model_fm = pm4py.convert_to_petri_net(pt)
        if profile_cpu:
            pass

        # res = {}
        # print("Running no deviations...")
        # res["no_deviation"] = run_offline_no_deviations(
        #     f"{model_dir}/{xes}", model_net, model_im, model_fm)
        # print("Running start deviations...")
        # res["trace_deviation_start"] = run_offline_remove_start(
        #     f"{model_dir}/{xes}", model_net, model_im, model_fm)
        # print("Running halfway deviations...")
        # res["trace_deviation_halfway"] = run_offline_remove_halfway(
        #     f"{model_dir}/{xes}", model_net, model_im, model_fm)
        # print("Running end deviations...")
        # res["trace_deviation_end"] = run_offline_remove_end(
        #     f"{model_dir}/{xes}", model_net, model_im, model_fm)
        # print("Running combo deviations...")
        # res["trace_deviation_start_halfway_end"] = run_offline_remove_start_halfway_end(
        #     f"{model_dir}/{xes}", model_net, model_im, model_fm)
        print("Running combo deviations...")
        res = run_offline_remove_start_halfway_end(f"{model_dir}/{xes}",
                                                   model_net, model_im,
                                                   model_fm)
        with open(f"{result_dir}/{filename}.json", "r") as rf:
            file_data = json.load(rf)
            file_data["trace_deviation_start_halfway_end"] = res
        with open(f"{result_dir}/{filename}.json", "w") as wf:
            # rstr = json.dumps(res, indent=4)
            rstr = json.dumps(file_data, indent=4)
            wf.write(rstr)


def itl():
    datasets = ["prAm6", "prBm6", "prCm6", "prDm6", "prEm6", "prFm6", "prGm6"]
    for dataset in datasets:
        model_net, model_im, model_fm = import_from_tpn(
            f"data/inthelarge/{dataset}.tpn")
        xes_df = pm4py.read_xes(f"data/inthelarge/{dataset}.xes")
        xes_el = convert_to_event_log(format_dataframe(xes_df))
        results = []
        for trace in tqdm(xes_el):
            trace_result = {}

            unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
                model_net, model_im, model_fm, trace)
            trace_result["unf_elapsed_time"] = unf_elapsed_time
            trace_result["unf_q"] = unf_q
            trace_result["unf_v"] = unf_v
            trace_result["unf_cost"] = unf_cost

            astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
                model_net, model_im, model_fm, trace)
            trace_result["astar_elapsed_time"] = astar_elapsed_time
            trace_result["astar_q"] = astar_q
            trace_result["astar_v"] = astar_v
            trace_result["astar_cost"] = astar_cost
            trace_result["astar_fitness"] = astar_fitness

            dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
                model_net, model_im, model_fm, trace)
            trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
            trace_result["dijkstra_q"] = dijkstra_q
            trace_result["dijkstra_v"] = dijkstra_v
            trace_result["dijkstra_cost"] = dijkstra_cost
            trace_result["dijkstra_fitness"] = dijkstra_fitness

            trace_result["trace_length"] = len(trace)
            results.append(trace_result)
        with open(f"results/inthelarge/{dataset}.json", "w") as wf:
            rstr = json.dumps(results, indent=4)
            wf.write(rstr)


def sepsis():
    xes_df = pm4py.read_xes("data/sepsis/Sepsis Cases - Event Log.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.5)
    view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    for trace in tqdm(xes_el):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/sepsis/sepsis_05.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def sepsis_split():
    xes_df = pm4py.read_xes("data/sepsis/Sepsis Cases - Event Log.xes")
    bpmn = pm4py.read_bpmn("data/sepsis/sepsis_split.bpmn")
    model_net, model_im, model_fm = pm4py.convert_to_petri_net(bpmn)
    view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    for trace in tqdm(xes_el):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/sepsis/sepsis_split.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def bpic17():
    xes_df = pm4py.read_xes("data/bpic17/BPI Challenge 2017.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.2)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/bpic17/bpic17_02.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def bpic17_split():
    xes_df = pm4py.read_xes("data/bpic17/BPI Challenge 2017.xes")
    bpmn = pm4py.read_bpmn("data/bpic17/bpic17_split.bpmn")
    model_net, model_im, model_fm = pm4py.convert_to_petri_net(bpmn)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/bpic17/bpic17_split.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def bpic19():
    xes_df = pm4py.read_xes("data/bpic19/BPI_Challenge_2019.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.2)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/bpic19/bpic19_02.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def traffic():
    xes_df = pm4py.read_xes(
        "data/traffic/Road_Traffic_Fine_Management_Process.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.2)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/traffic/traffic_02.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def traffic_split():
    xes_df = pm4py.read_xes(
        "data/traffic/Road_Traffic_Fine_Management_Process.xes")
    bpmn = pm4py.read_bpmn("data/traffic/traffic_split.bpmn")
    model_net, model_im, model_fm = pm4py.convert_to_petri_net(bpmn)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/traffic/traffic_split.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def hospital_billing():
    xes_df = pm4py.read_xes(
        "data/hospital_billing/Hospital Billing - Event Log.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.2)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/hospital_billing/hospital_billing_02.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def hospital_billing_split():
    xes_df = pm4py.read_xes(
        "data/hospital_billing/Hospital Billing - Event Log.xes")
    bpmn = pm4py.read_bpmn("data/hospital_billing/hospital_billing_split.bpmn")
    model_net, model_im, model_fm = pm4py.convert_to_petri_net(bpmn)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/hospital_billing/hospital_billing_split.json",
              "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def bpic13():
    xes_df = pm4py.read_xes("data/bpic13/BPI_Challenge_2013_incidents.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.2)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/bpic13/bpic13_02.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def bpic13_split():
    xes_df = pm4py.read_xes("data/bpic13/BPI_Challenge_2013_incidents.xes")
    bpmn = pm4py.read_bpmn("data/bpic13/bpic13_split.bpmn")
    model_net, model_im, model_fm = pm4py.convert_to_petri_net(bpmn)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/bpic13/bpic13_split.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def bpic12():
    xes_df = pm4py.read_xes("data/bpic12/BPI_Challenge_2012.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.2)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/bpic12/bpic12_02.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def bpic12_split():
    xes_df = pm4py.read_xes("data/bpic12/BPI_Challenge_2012.xes")
    bpmn = pm4py.read_bpmn("data/bpic12/bpic12_split.bpmn")
    model_net, model_im, model_fm = pm4py.convert_to_petri_net(bpmn)
    # view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost

        astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_q
        trace_result["astar_v"] = astar_v
        trace_result["astar_cost"] = astar_cost
        trace_result["astar_fitness"] = astar_fitness

        dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            model_net, model_im, model_fm, trace)
        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_q
        trace_result["dijkstra_v"] = dijkstra_v
        trace_result["dijkstra_cost"] = dijkstra_cost
        trace_result["dijkstra_fitness"] = dijkstra_fitness

        trace_result["trace_length"] = len(trace)
        results.append(trace_result)
    with open(f"results/bpic12/bpic12_split.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def bpic15():
    datasets = ["BPIC15_1", "BPIC15_2", "BPIC15_3", "BPIC15_4", "BPIC15_5"]
    for dataset in datasets:
        xes_df = pm4py.read_xes(f"data/bpic15/{dataset}.xes")
        model_net, model_im, model_fm = discover_petri_net_inductive(
            xes_df, noise_threshold=0.2)
        # view_petri_net(model_net, model_im, model_fm)
        xes_el = convert_to_event_log(format_dataframe(xes_df))
        results = []
        random.seed(1)
        rnd = random.sample(xes_el, 1000)
        for trace in tqdm(rnd):
            trace_result = {}

            unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
                model_net, model_im, model_fm, trace)
            trace_result["unf_elapsed_time"] = unf_elapsed_time
            trace_result["unf_q"] = unf_q
            trace_result["unf_v"] = unf_v
            trace_result["unf_cost"] = unf_cost

            astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
                model_net, model_im, model_fm, trace)
            trace_result["astar_elapsed_time"] = astar_elapsed_time
            trace_result["astar_q"] = astar_q
            trace_result["astar_v"] = astar_v
            trace_result["astar_cost"] = astar_cost
            trace_result["astar_fitness"] = astar_fitness

            dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
                model_net, model_im, model_fm, trace)
            trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
            trace_result["dijkstra_q"] = dijkstra_q
            trace_result["dijkstra_v"] = dijkstra_v
            trace_result["dijkstra_cost"] = dijkstra_cost
            trace_result["dijkstra_fitness"] = dijkstra_fitness

            trace_result["trace_length"] = len(trace)
            results.append(trace_result)
        with open(f"results/bpic15/{dataset}.json", "w") as wf:
            rstr = json.dumps(results, indent=4)
            wf.write(rstr)


def itl_inductive():
    datasets = ["prAm6", "prBm6", "prCm6", "prDm6", "prEm6", "prFm6", "prGm6"]
    for dataset in datasets:
        xes_df = pm4py.read_xes(f"data/inthelarge/{dataset}.xes")
        model_net, model_im, model_fm = discover_petri_net_inductive(
            xes_df, noise_threshold=0.2)
        # view_petri_net(model_net, model_im, model_fm)
        xes_el = convert_to_event_log(format_dataframe(xes_df))
        results = []
        for trace in tqdm(xes_el):
            trace_result = {}

            unf_elapsed_time, unf_q, unf_v, unf_cost = run_offline(
                model_net, model_im, model_fm, trace)
            trace_result["unf_elapsed_time"] = unf_elapsed_time
            trace_result["unf_q"] = unf_q
            trace_result["unf_v"] = unf_v
            trace_result["unf_cost"] = unf_cost

            astar_elapsed_time, astar_q, astar_v, astar_cost, astar_fitness = run_astar_with_timer(
                model_net, model_im, model_fm, trace)
            trace_result["astar_elapsed_time"] = astar_elapsed_time
            trace_result["astar_q"] = astar_q
            trace_result["astar_v"] = astar_v
            trace_result["astar_cost"] = astar_cost
            trace_result["astar_fitness"] = astar_fitness

            # dijkstra_elapsed_time, dijkstra_q, dijkstra_v, dijkstra_cost, dijkstra_fitness = run_dijkstra_with_timer(
            #     model_net, model_im, model_fm, trace)
            # trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
            # trace_result["dijkstra_q"] = dijkstra_q
            # trace_result["dijkstra_v"] = dijkstra_v
            # trace_result["dijkstra_cost"] = dijkstra_cost
            # trace_result["dijkstra_fitness"] = dijkstra_fitness

            trace_result["trace_length"] = len(trace)
            results.append(trace_result)
        with open(f"results/inthelarge_mined/{dataset}.json", "w") as wf:
            rstr = json.dumps(results, indent=4)
            wf.write(rstr)


def sepsis_streaming():
    xes_df = pm4py.read_xes("data/sepsis/Sepsis Cases - Event Log.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.5)
    view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    for trace in tqdm(xes_el):
        # Store now because trace will be modified later
        trace_len = len(trace)
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost, unf_q_per_iteration, unf_v_per_iteration = run_streaming(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost
        trace_result["unf_q_per_iteration"] = unf_q_per_iteration
        trace_result["unf_v_per_iteration"] = unf_v_per_iteration

        trace_result["trace_length"] = trace_len
        results.append(trace_result)
    with open(f"results/streaming/sepsis/sepsis_05.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def bpic17_streaming():
    xes_df = pm4py.read_xes("data/bpic17/BPI Challenge 2017.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.2)
    view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        # Store now because trace will be modified later
        trace_len = len(trace)
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost, unf_q_per_iteration, unf_v_per_iteration = run_streaming(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost
        trace_result["unf_q_per_iteration"] = unf_q_per_iteration
        trace_result["unf_v_per_iteration"] = unf_v_per_iteration

        trace_result["trace_length"] = trace_len
        results.append(trace_result)
    with open(f"results/streaming/bpic17/bpic17_02.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def bpic19_streaming():
    xes_df = pm4py.read_xes("data/bpic19/BPI_Challenge_2019.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.2)
    view_petri_net(model_net, model_im, model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    random.seed(1)
    rnd = random.sample(xes_el, 1000)
    for trace in tqdm(rnd):
        # Store now because trace will be modified later
        trace_len = len(trace)
        trace_result = {}

        unf_elapsed_time, unf_q, unf_v, unf_cost, unf_q_per_iteration, unf_v_per_iteration = run_streaming(
            model_net, model_im, model_fm, trace)
        trace_result["unf_elapsed_time"] = unf_elapsed_time
        trace_result["unf_q"] = unf_q
        trace_result["unf_v"] = unf_v
        trace_result["unf_cost"] = unf_cost
        trace_result["unf_q_per_iteration"] = unf_q_per_iteration
        trace_result["unf_v_per_iteration"] = unf_v_per_iteration

        trace_result["trace_length"] = trace_len
        results.append(trace_result)
    with open(f"results/streaming/bpic19/bpic19_02.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)


def main(profile_cpu=False, profile_memory=False):
    dir_pairs = [(CONCURRENT_MODEL_DIR, CONCURRENT_RESULT_DIR),
                 (CONCURRENT_CONCURRENT_NESTED_MODEL_DIR,
                  CONCURRENT_CONCURRENT_NESTED_RESULT_DIR),
                 (EXCLUSIVE_MODEL_DIR, EXCLUSIVE_RESULT_DIR),
                 (EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR,
                  EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR),
                 (LOOP_MODEL_DIR, LOOP_RESULT_DIR)]
    for model_dir, result_dir in dir_pairs:
        run_artificial_model_offline(model_dir, result_dir, profile_cpu,
                                     profile_memory)


def itl_streaming():
    datasets = ["prAm6", "prBm6", "prCm6", "prDm6", "prEm6", "prFm6", "prGm6"]
    for dataset in datasets:
        model_net, model_im, model_fm = import_from_tpn(
            f"data/inthelarge/{dataset}.tpn")
        xes_df = pm4py.read_xes(f"data/inthelarge/{dataset}.xes")
        xes_el = convert_to_event_log(format_dataframe(xes_df))
        results = []
        for trace in tqdm(xes_el):
            # Store now because trace will be modified later
            trace_len = len(trace)
            trace_result = {}

            unf_elapsed_time, unf_q, unf_v, unf_cost, unf_q_per_iteration, unf_v_per_iteration = run_streaming(
                model_net, model_im, model_fm, trace)
            trace_result["unf_elapsed_time"] = unf_elapsed_time
            trace_result["unf_q"] = unf_q
            trace_result["unf_v"] = unf_v
            trace_result["unf_cost"] = unf_cost
            trace_result["unf_q_per_iteration"] = unf_q_per_iteration
            trace_result["unf_v_per_iteration"] = unf_v_per_iteration

            trace_result["trace_length"] = trace_len
            results.append(trace_result)
        with open(f"results/streaming/inthelarge/{dataset}.json", "w") as wf:
            rstr = json.dumps(results, indent=4)
            wf.write(rstr)


if __name__ == "__main__":
    # main()
    # preliminary()
    # sepsis()
    # itl()
    # bpic17()
    # bpic19()
    # itl_inductive()
    # sepsis_streaming()
    # bpic17_streaming()
    # bpic19_streaming()
    # itl_streaming()
    # traffic()
    # hospital()
    # bpic12()
    # bpic13()
    # hospital_billing()
    # hospital()
    # bpic15()
    sepsis_split()
    traffic_split()
    hospital_billing_split()
    bpic12_split()
    bpic13_split()
    bpic17_split()
    # pass
