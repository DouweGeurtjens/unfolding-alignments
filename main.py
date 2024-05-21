import json
import cProfile
from coset_standard import *
from coset_streaming import *
import os
from settings import *
import time
from tqdm import tqdm

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

def run_coset_standard_1_deviation_at_start(data_filepath,model_filepath):
    total_traces = 0
    unf_q = []
    unf_v = []
    unf_elapsed_time = []
    cost_mapping = {
        MoveTypes.LOG.name: 1000,
        MoveTypes.MODEL.name: 1000,
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

        trace_net, trace_net_im, trace_net_fm = construct_trace_net(
            trace, "concept:name", "concept:name")
        
        # Deviate the first transition, raise exception if something is wrong
        if len(trace_net_im) >1:
            raise Exception
        
        for place in trace_net_im:
            if len(place.out_arcs) > 1:
                raise Exception
            for arc in place.out_arcs:
                arc.target.label = "DEVIATION"

        sync_net, sync_im, sync_fm = construct_synchronous_product(
            model_net, model_im, model_fm, trace_net, trace_net_im,
            trace_net_fm)
        sync_net_extended = ExtendedSyncNet(sync_net, sync_im, sync_fm)

        bp = BranchingProcessStandard(sync_net_extended)

        bp.initialize_from_initial_marking(cost_mapping)
        start_time = time.time()
        alignment = bp.astar(cost_mapping)
        end_time = time.time()
        unf_elapsed_time.append(end_time-start_time)

        # print(
        #     f"Qd {bp.possible_extensions._queued}, Vd {bp.possible_extensions._visited}"
        # )
        unf_q.append(bp.possible_extensions._queued)
        unf_v.append(bp.possible_extensions._visited)

    results = {}
    results["total_traces"] = total_traces
    results["unf_q"] = unf_q
    results["unf_v"] = unf_v
    results["unf_elapsed_time"] = unf_elapsed_time

    return results    

def run_coset_standard_no_deviations(data_filepath, model_filepath):
    total_traces = 0
    unf_q = []
    unf_v = []
    unf_elapsed_time = []
    cost_mapping = {
        MoveTypes.LOG.name: 1000,
        MoveTypes.MODEL.name: 1000,
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

        trace_net, trace_net_im, trace_net_fm = construct_trace_net(
            trace, "concept:name", "concept:name")

        sync_net, sync_im, sync_fm = construct_synchronous_product(
            model_net, model_im, model_fm, trace_net, trace_net_im,
            trace_net_fm)
        
        sync_net_extended = ExtendedSyncNet(sync_net, sync_im, sync_fm)

        bp = BranchingProcessStandard(sync_net_extended)

        bp.initialize_from_initial_marking(cost_mapping)
        start_time = time.time()
        alignment = bp.astar(cost_mapping)
        end_time = time.time()
        unf_elapsed_time.append(end_time-start_time)

        # print(
        #     f"Qd {bp.possible_extensions._queued}, Vd {bp.possible_extensions._visited}"
        # )
        unf_q.append(bp.possible_extensions._queued)
        unf_v.append(bp.possible_extensions._visited)

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
        MoveTypes.LOG.name: 1000,
        MoveTypes.MODEL.name: 1000,
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

def main(profile_cpu=False,profile_memory=False):
    conc = os.listdir(CONCURRENT_MODEL_DIR)
    for f in conc:
        if f.endswith(".ptml"):
            continue
        filename = f.removesuffix(".xes")
        ptml = filename + ".ptml"
        xes = filename + ".xes"
        view_model(f"{CONCURRENT_MODEL_DIR}/{xes}",f"{CONCURRENT_MODEL_DIR}/{ptml}")
        if profile_cpu:
            with cProfile.Profile() as pr:
                res = run_coset_standard_no_deviations(f"{CONCURRENT_MODEL_DIR}/{xes}",f"{CONCURRENT_MODEL_DIR}/{ptml}")
            pr.dump_stats(f"{CONCURRENT_RESULT_DIR}/{filename}_no_deviations.prof")

        res = run_coset_standard_no_deviations(f"{CONCURRENT_MODEL_DIR}/{xes}",f"{CONCURRENT_MODEL_DIR}/{ptml}")        
        with open(f"{CONCURRENT_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)

        res = run_coset_standard_1_deviation_at_start(f"{CONCURRENT_MODEL_DIR}/{xes}",f"{CONCURRENT_MODEL_DIR}/{ptml}")        
        with open(f"{CONCURRENT_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)


    conc_conc = os.listdir(CONCURRENT_CONCURRENT_NESTED_MODEL_DIR)
    for f in conc_conc:
        if f.endswith(".ptml"):
            continue
        print(f)
        filename = f.removesuffix(".xes")
        ptml = filename + ".ptml"
        xes = filename + ".xes"

        if profile_cpu:
            with cProfile.Profile() as pr:
                res = run_coset_standard_no_deviations(f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
            pr.dump_stats(f"{CONCURRENT_CONCURRENT_NESTED_RESULT_DIR}/{filename}_no_deviations.prof")

        res = run_coset_standard_no_deviations(f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
        with open(f"{CONCURRENT_CONCURRENT_NESTED_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)

        res = run_coset_standard_1_deviation_at_start(f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
        with open(f"{CONCURRENT_CONCURRENT_NESTED_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)


    conc_ex = os.listdir(CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR)
    for f in conc_ex:
        if f.endswith(".ptml"):
            continue
        filename = f.removesuffix(".xes")
        ptml = filename + ".ptml"
        xes = filename + ".xes"

        if profile_cpu:
            with cProfile.Profile() as pr:
                res = run_coset_standard_no_deviations(f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
            pr.dump_stats(f"{CONCURRENT_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_no_deviations.prof")

        res = run_coset_standard_no_deviations(f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
        with open(f"{CONCURRENT_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)

        res = run_coset_standard_1_deviation_at_start(f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{CONCURRENT_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
        with open(f"{CONCURRENT_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)

    ex = os.listdir(EXCLUSIVE_MODEL_DIR)
    for f in ex:
        if f.endswith(".ptml"):
            continue
        filename = f.removesuffix(".xes")
        ptml = filename + ".ptml"
        xes = filename + ".xes"

        if profile_cpu:
            with cProfile.Profile() as pr:
                res = run_coset_standard_no_deviations(f"{EXCLUSIVE_MODEL_DIR}/{xes}",f"{EXCLUSIVE_MODEL_DIR}/{ptml}")
            pr.dump_stats(f"{EXCLUSIVE_RESULT_DIR}/{filename}_no_deviations.prof")

        res = run_coset_standard_no_deviations(f"{EXCLUSIVE_MODEL_DIR}/{xes}",f"{EXCLUSIVE_MODEL_DIR}/{ptml}")
        with open(f"{EXCLUSIVE_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)

        res = run_coset_standard_1_deviation_at_start(f"{EXCLUSIVE_MODEL_DIR}/{xes}",f"{EXCLUSIVE_MODEL_DIR}/{ptml}")
        with open(f"{EXCLUSIVE_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)

    ex_conc = os.listdir(EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR)
    for f in ex_conc:
        if f.endswith(".ptml"):
            continue
        filename = f.removesuffix(".xes")
        ptml = filename + ".ptml"
        xes = filename + ".xes"

        if profile_cpu:
            with cProfile.Profile() as pr:
                res = run_coset_standard_no_deviations(f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
            pr.dump_stats(f"{EXCLUSIVE_CONCURRENT_NESTED_RESULT_DIR}/{filename}_no_deviations.prof")

        res = run_coset_standard_no_deviations(f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
        with open(f"{EXCLUSIVE_CONCURRENT_NESTED_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)

        res = run_coset_standard_1_deviation_at_start(f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_CONCURRENT_NESTED_MODEL_DIR}/{ptml}")
        with open(f"{EXCLUSIVE_CONCURRENT_NESTED_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)


    ex_ex = os.listdir(EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR)
    for f in ex_ex:
        if f.endswith(".ptml"):
            continue
        filename = f.removesuffix(".xes")
        ptml = filename + ".ptml"
        xes = filename + ".xes"

        if profile_cpu:
            with cProfile.Profile() as pr:
                res = run_coset_standard_no_deviations(f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
            pr.dump_stats(f"{EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_no_deviations.prof")   

        res = run_coset_standard_no_deviations(f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
        with open(f"{EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_no_deviations.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)

        res = run_coset_standard_1_deviation_at_start(f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{xes}",f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/{ptml}")
        with open(f"{EXCLUSIVE_EXCLUSIVE_NESTED_RESULT_DIR}/{filename}_1_deviation_at_start.json", "w") as wf:
            rstr = json.dumps(res, indent=4)
            wf.write(rstr)


if __name__ == "__main__":
    main()