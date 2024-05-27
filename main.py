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

class DeviationsOperations(Enum):
    REMOVE = 1
    REPLACE_WITH_INVERSE_PERCENTILE = 2

def print_stats(filepath):
    with open(filepath) as f:
        stats = json.load(f)
        print(f"Avg trad q: {sum(stats["trad_q"])/len(stats["trad_q"])}")
        print(f"Avg trad v: {sum(stats["trad_v"])/len(stats["trad_v"])}")
        print(f"Avg unf q: {sum(stats["unf_q"])/len(stats["unf_q"])}")
        print(f"Avg unf v: {sum(stats["unf_v"])/len(stats["unf_v"])}")

# Types of deviations for concurrent models: mandatory activity missing in 1 branch, multiple branches, multiple missing, different places

def view_model(data_filepath,model_filepath):
    pt = pm4py.read_ptml(model_filepath)
    model_net, model_im, model_fm = pm4py.convert_to_petri_net(pt)
    view_petri_net(model_net)
    return

def apply_trace_deviation(location_percentiles:list[float],operation,trace):
    trace_copy = deepcopy(trace)
    for percentile in location_percentiles:
        index = len(trace_copy._list) * percentile
        if operation ==DeviationsOperations.REMOVE.name:
            trace_copy._list.remove(trace_copy._list[index])
        if operation == DeviationsOperations.REPLACE_WITH_INVERSE_PERCENTILE.name:
            index_inverse = len(trace_copy._list) * (1-percentile)
            trace_copy._list[index]["concept:name"] = trace_copy._list[index_inverse]["concept:name"]
    return trace_copy 

def run_coset_standard(model_net, model_im, model_fm, trace):
    cost_mapping = {
        MoveTypes.LOG.name: 10000,
        MoveTypes.MODEL.name: 10000,
        MoveTypes.SYNC.name: 0,
        MoveTypes.MODEL_SILENT.name: 1,
        MoveTypes.DUMMY.name: 0
    }  

    start_time = time.time()
    trace_net, trace_net_im, trace_net_fm = construct_trace_net(
            trace, "concept:name", "concept:name")

    sync_net, sync_im, sync_fm = construct_synchronous_product(
            model_net, model_im, model_fm, trace_net, trace_net_im,
            trace_net_fm)
        
    sync_net_extended = ExtendedSyncNet(sync_net, sync_im, sync_fm)

    bp = BranchingProcessStandard(sync_net_extended)

    bp.initialize_from_initial_marking(cost_mapping)
    alignment,cost = bp.astar(cost_mapping)
    elapsed_time = time.time() - start_time
    return elapsed_time,bp.possible_extensions._queued,bp.possible_extensions._visited,cost


def run_coset_standard_remove_halfway(data_filepath, model_net,model_im,model_fm):
    total_traces = 0
    unf_q = []
    unf_v = []
    unf_elapsed_time = []
    xes_df = pm4py.read_xes(data_filepath)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    for trace in tqdm(xes_el):
        deviating_trace = apply_trace_deviation([0.5],DeviationsOperations.REMOVE.name,trace)
        elapsed_time,q,v,cost = run_coset_standard(model_net,model_im,model_fm,deviating_trace)
        unf_elapsed_time.append(elapsed_time)
        unf_q.append(q)
        unf_v.append(v)

    results = {}
    results["total_traces"] = total_traces
    results["unf_q"] = unf_q
    results["unf_v"] = unf_v
    results["unf_elapsed_time"] = unf_elapsed_time


    return results


def run_coset_standard_remove_end(data_filepath, model_net,model_im,model_fm):
    total_traces = 0
    unf_q = []
    unf_v = []
    unf_elapsed_time = []
    xes_df = pm4py.read_xes(data_filepath)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    for trace in tqdm(xes_el):
        deviating_trace = apply_trace_deviation([1],DeviationsOperations.REMOVE.name,trace)
        elapsed_time,q,v,cost = run_coset_standard(model_net,model_im,model_fm,deviating_trace)
        unf_elapsed_time.append(elapsed_time)
        unf_q.append(q)
        unf_v.append(v)

    results = {}
    results["total_traces"] = total_traces
    results["unf_q"] = unf_q
    results["unf_v"] = unf_v
    results["unf_elapsed_time"] = unf_elapsed_time


    return results

def run_coset_standard_remove_start(data_filepath, model_net,model_im,model_fm):
    total_traces = 0
    unf_q = []
    unf_v = []
    unf_elapsed_time = []
    xes_df = pm4py.read_xes(data_filepath)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    for trace in tqdm(xes_el):
        deviating_trace = apply_trace_deviation([0],DeviationsOperations.REMOVE.name,trace)
        elapsed_time,q,v,cost = run_coset_standard(model_net,model_im,model_fm,deviating_trace)
        unf_elapsed_time.append(elapsed_time)
        unf_q.append(q)
        unf_v.append(v)

    results = {}
    results["total_traces"] = total_traces
    results["unf_q"] = unf_q
    results["unf_v"] = unf_v
    results["unf_elapsed_time"] = unf_elapsed_time


    return results

def run_coset_standard_no_deviations(data_filepath, model_net,model_im,model_fm):
    total_traces = 0
    unf_q = []
    unf_v = []
    unf_elapsed_time = []
    xes_df = pm4py.read_xes(data_filepath)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    for trace in tqdm(xes_el):
        elapsed_time,q,v,cost = run_coset_standard(model_net,model_im,model_fm,trace)
        unf_elapsed_time.append(elapsed_time)
        unf_q.append(q)
        unf_v.append(v)

    results = {}
    results["total_traces"] = total_traces
    results["unf_q"] = unf_q
    results["unf_v"] = unf_v
    results["unf_elapsed_time"] = unf_elapsed_time


    return results

def run_stream_no_deviations(data_filepath, model_filepath):
    total_traces = 0
    unf_q = []
    unf_v = []
    unf_q_per_iteration = []
    unf_v_per_iteration = []
    unf_elapsed_time = []
    cost_mapping = {
        MoveTypes.LOG.name: 10000,
        MoveTypes.MODEL.name: 10000,
        MoveTypes.SYNC.name: 0,
        MoveTypes.MODEL_SILENT.name: 1,
        MoveTypes.DUMMY.name: 0
    }
    xes_df = pm4py.read_xes(data_filepath)
    pt = pm4py.read_ptml(model_filepath)
    model_net, model_im, model_fm = pm4py.convert_to_petri_net(pt)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    for trace in tqdm(xes_el):
        total_traces += 1

        stream = Stream(trace, (500000000, 1000000000))

        # Get the first slice of the trace after "connecting" to the stream
        operation, operation_value = stream.pop(0)

        # Initialize other stuffs
        trace_net, trace_net_im, trace_net_fm = construct_trace_net(
            operation_value, "concept:name", "concept:name")
        sync_net, sync_im, sync_fm = construct_synchronous_product(
            model_net, model_im, model_fm, trace_net, trace_net_im,
            trace_net_fm)
        
        extended_net = ExtendedSyncNetStreaming(sync_net, sync_im, sync_fm,
                                                trace_net_fm)
        # Start balling
        bp = BranchingProcessStream(extended_net, stream)
        bp.initialize_from_initial_marking(cost_mapping)
        start_time = time.time()
        final_alignment = bp.alignment_streaming(model_net, model_im, model_fm,
                                                 cost_mapping)
        end_time = time.time()
        unf_elapsed_time.append(end_time-start_time)

        # print(
        #     f"Qd {bp.possible_extensions._queued}, Vd {bp.possible_extensions._visited}"
        # )
        unf_q.append(bp.possible_extensions._queued)
        unf_v.append(bp.possible_extensions._visited)
        unf_q_per_iteration.append(bp.unf_q_per_iteration)
        unf_v_per_iteration.append(bp.unf_v_per_iteration)        
        
    results = {}
    results["total_traces"] = total_traces
    results["unf_q"] = unf_q
    results["unf_v"] = unf_v
    results["unf_q_per_iteration"] = unf_q_per_iteration
    results["unf_v_per_iteration"] = unf_v_per_iteration
    results["unf_elapsed_time"] = unf_elapsed_time

    return results    

def preliminary():
    xes_df = pm4py.read_xes("data/Sepsis Cases - Event Log.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0.5)
    view_petri_net(model_net,model_im,model_fm)
    xes_el = convert_to_event_log(format_dataframe(xes_df))
    results = []
    rnd = random.sample(xes_el,10)
    for trace in tqdm(rnd):
        trace_result = {}

        dijkstra_start_time = time.time()
        dijkstra_alignment = conformance_diagnostics_alignments(
                trace,
                model_net,
                model_im,
                model_fm,
                variant_str="Variants.VERSION_DIJKSTRA_NO_HEURISTICS")
        dijkstra_elapsed_time = time.time() -dijkstra_start_time

        print(dijkstra_alignment)

        astar_start_time = time.time()
        astar_alignment = conformance_diagnostics_alignments(
                trace,
                model_net,
                model_im,
                model_fm)
        astar_elapsed_time= time.time() - astar_start_time

        print(astar_alignment)

        elapsed_time,q,v,cost= run_coset_standard(model_net,model_im,model_fm,trace)
        trace_result["unf_elapsed_time"] = elapsed_time
        trace_result["unf_q"] = q
        trace_result["unf_v"] = v
        trace_result["unf_cost"] = cost

        trace_result["dijkstra_elapsed_time"] = dijkstra_elapsed_time
        trace_result["dijkstra_q"] = dijkstra_alignment["queued_states"]
        trace_result["dijkstra_v"] = dijkstra_alignment["visited_states"]
        trace_result["dijkstra_cost"] = dijkstra_alignment["cost"]

        trace_result["astar_elapsed_time"] = astar_elapsed_time
        trace_result["astar_q"] = astar_alignment["queued_states"]
        trace_result["astar_v"] = astar_alignment["visited_states"]
        trace_result["astar_cost"] = astar_alignment["cost"]
        results.append(trace_result)
    with open(f"preliminary_sepsis_05_noise.json", "w") as wf:
        rstr = json.dumps(results, indent=4)
        wf.write(rstr)

def main(profile_cpu=False,profile_memory=False):
    # conc = os.listdir(CONCURRENT_MODEL_DIR)
    # for f in conc:
    #     if f.endswith(".ptml"):
    #         continue
    #     filename = f.removesuffix(".xes")
    #     ptml = filename + ".ptml"
    #     xes = filename + ".xes"
    #     view_model(f"{CONCURRENT_MODEL_DIR}/{xes}",f"{CONCURRENT_MODEL_DIR}/{ptml}")
    #     if profile_cpu:
    #         with cProfile.Profile() as pr:
    #             res = run_coset_standard_no_deviations(f"{CONCURRENT_MODEL_DIR}/{xes}",f"{CONCURRENT_MODEL_DIR}/{ptml}")
    #         pr.dump_stats(f"{CONCURRENT_RESULT_DIR}/{filename}_no_deviations.prof")

    #     res = run_coset_standard_no_deviations(f"{CONCURRENT_MODEL_DIR}/{xes}",f"{CONCURRENT_MODEL_DIR}/{ptml}")        
    #     with open(f"{CONCURRENT_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)

    #     res = run_coset_standard_1_deviation_at_start(f"{CONCURRENT_MODEL_DIR}/{xes}",f"{CONCURRENT_MODEL_DIR}/{ptml}")        
    #     with open(f"{CONCURRENT_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)


    # conc_conc = os.listdir(CONCURRENT_CONCURRENT_NESTED_MODEL_DIR)
    # for f in conc_conc:
    #     if f.endswith(".ptml"):
    #         continue
    #     print(f)
    #     filename = f.removesuffix(".xes")
    #     ptml = filename + ".ptml"
    #     xes = filename + ".xes"

    #     if profile_cpu:
    #         with cProfile.Profile() as pr:
    #             res = run_coset_standard_no_deviations(f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
    #         pr.dump_stats(f"{CONCURRENT_CONCURRENT_NESTED_RESULT_DIR}/{filename}_no_deviations.prof")

    #     res = run_coset_standard_no_deviations(f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
    #     with open(f"{CONCURRENT_CONCURRENT_NESTED_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)

    #     res = run_coset_standard_1_deviation_at_start(f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
    #     with open(f"{CONCURRENT_CONCURRENT_NESTED_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)


    # conc_ex = os.listdir(CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR)
    # for f in conc_ex:
    #     if f.endswith(".ptml"):
    #         continue
    #     filename = f.removesuffix(".xes")
    #     ptml = filename + ".ptml"
    #     xes = filename + ".xes"

    #     if profile_cpu:
    #         with cProfile.Profile() as pr:
    #             res = run_coset_standard_no_deviations(f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
    #         pr.dump_stats(f"{CONCURRENT_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_no_deviations.prof")

    #     res = run_coset_standard_no_deviations(f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
    #     with open(f"{CONCURRENT_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)

    #     res = run_coset_standard_1_deviation_at_start(f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
    #     with open(f"{CONCURRENT_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)

    # ex = os.listdir(EXCLUSIVE_MODEL_DIR)
    # for f in ex:
    #     if f.endswith(".ptml"):
    #         continue
    #     filename = f.removesuffix(".xes")
    #     ptml = filename + ".ptml"
    #     xes = filename + ".xes"

    #     if profile_cpu:
    #         with cProfile.Profile() as pr:
    #             res = run_coset_standard_no_deviations(f"{EXCLUSIVE_MODEL_DIR}/{xes}",f"{EXCLUSIVE_MODEL_DIR}/{ptml}")
    #         pr.dump_stats(f"{EXCLUSIVE_RESULT_DIR}/{filename}_no_deviations.prof")

    #     res = run_coset_standard_no_deviations(f"{EXCLUSIVE_MODEL_DIR}/{xes}",f"{EXCLUSIVE_MODEL_DIR}/{ptml}")
    #     with open(f"{EXCLUSIVE_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)

    #     res = run_coset_standard_1_deviation_at_start(f"{EXCLUSIVE_MODEL_DIR}/{xes}",f"{EXCLUSIVE_MODEL_DIR}/{ptml}")
    #     with open(f"{EXCLUSIVE_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)

    # ex_conc = os.listdir(EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR)
    # for f in ex_conc:
    #     if f.endswith(".ptml"):
    #         continue
    #     filename = f.removesuffix(".xes")
    #     ptml = filename + ".ptml"
    #     xes = filename + ".xes"

    #     if profile_cpu:
    #         with cProfile.Profile() as pr:
    #             res = run_coset_standard_no_deviations(f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
    #         pr.dump_stats(f"{EXCLUSIVE_CONCURRENT_NESTED_RESULT_DIR}/{filename}_no_deviations.prof")

    #     res = run_coset_standard_no_deviations(f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
    #     with open(f"{EXCLUSIVE_CONCURRENT_NESTED_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)

    #     res = run_coset_standard_1_deviation_at_start(f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
    #     with open(f"{EXCLUSIVE_CONCURRENT_NESTED_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)


    # ex_ex = os.listdir(EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR)
    # for f in ex_ex:
    #     if f.endswith(".ptml"):
    #         continue
    #     filename = f.removesuffix(".xes")
    #     ptml = filename + ".ptml"
    #     xes = filename + ".xes"

    #     if profile_cpu:
    #         with cProfile.Profile() as pr:
    #             res = run_coset_standard_no_deviations(f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
    #         pr.dump_stats(f"{EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_no_deviations.prof")   

    #     res = run_coset_standard_no_deviations(f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
    #     with open(f"{EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)

    #     res = run_coset_standard_1_deviation_at_start(f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
    #     with open(f"{EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
    #         rstr = json.dumps(res, indent=4)
    #         wf.write(rstr)
    pass


if __name__ == "__main__":
    preliminary()