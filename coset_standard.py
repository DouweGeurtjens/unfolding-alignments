from copy import deepcopy, copy
import pandas as pd
from pm4py import PetriNet, Marking, discover_petri_net_inductive, view_petri_net, format_dataframe, convert_to_event_log, conformance_diagnostics_alignments
# Use TypeAlias because the type notation in introduced in 3.12 isn't support by yapf yet
from typing import TypeAlias
import cProfile, pstats
from collections import Counter
import heapq
from operator import itemgetter
# TODO fix imports
from petrinet import *
from itertools import product, combinations
from coset_base import *
# TODO ALWAYS REMEMBER TO KEEP IN MIND THAT CONDITIONS/EVENTS ARE NOT DIRECTLY COMPARABLE TO PLACES/TRANSITIONS
# TODO THIS MEANS THAT SET OPERATIONS WILL ONLY WORK BETWEEN THE SAME TYPE, SO NO conditions.intersection.places

# TODO there is some issue with the synchronous product and invisible transitions
# TODO set unions take a lot of time, perhaps it's possible (in some cases) to use a linked list for faster extension?
PlaceID: TypeAlias = int
TransitionID: TypeAlias = int


class BranchingProcessStandard(BranchingProcess):

    def __init__(self, net: ExtendedNet) -> None:
        super().__init__(net)

    def initialize_from_initial_marking(self, cost_mapping):
        for place in self.underlying_net.places:
            self.conditions[place.properties[NetProperties.ID.name]] = set()

        added_conditions = set()
        for place in self.underlying_net.im:
            condition = Condition(place.properties[NetProperties.ID.name])
            self.conditions[place.properties[NetProperties.ID.name]].add(
                condition)
            added_conditions.add(condition)

        new_possible_extensions = self.compute_pe(added_conditions)
        # TODO make proper cost function
        new_possible_extensions_with_cost = [
            self.pe_to_astar_search(x, cost_mapping)
            for x in new_possible_extensions
        ]
        self.possible_extensions.push_many(new_possible_extensions_with_cost)
        # Update the seen extensions
        # TODO check if this makes any sense
        # self.extensions_seen.update((new_possible_extensions))

    def is_co_related(self, conditions: frozenset[Condition]):
        # Start from our conditions, follow down the local configuration via a queue of events
        # Mark the input conditions of an event with the event it was visited from
        # If we find a condition which was already visited from a different event there is a conflict
        # Meanwhile, check if the conditions we are marking are in the set we are testing for co-relation
        # If one is, they are in causal relation
        queue: list[Event] = []
        marks = {}

        # Starting conditions are visited from nothings, so we use a negative integer
        for c in conditions:
            if c.input_event is not None:
                queue.append(c.input_event)

        while len(queue) > 0:
            item = queue.pop(0)
            visited_from = item
            for c in item.input_conditions:
                # Causal so not in co-relation
                if c in conditions:
                    return False

                # This condition has been visited before
                if c in marks:
                    # It has been visited from a different transition
                    if marks[c] != visited_from:
                        return False
                # If possible, add input event on the queue
                elif c.input_event is not None:
                    queue.append(c.input_event)
                # Mark this condition
                marks[c] = visited_from

        return True

    def compute_pe(
        self, added_conditions: set[Condition]
    ) -> set[tuple[TransitionID, frozenset[Condition]]]:
        # For each transition in the net, find a co-set which is a superset
        # Those transitions are possible extensions
        new_possible_extensions = set()

        # TODO the only transitions that can be enabled are those present in the new_co_sets postset right?
        postset_transition_ids = set()
        for condition in added_conditions:
            # Basically any transition that is in the postset of a place in the marking of the co-set
            postset_transition_ids.update(
                get_postset_ids(
                    self.underlying_net.get_net_node_by_id(
                        condition.net_place_id)))

        for transition_id in postset_transition_ids:
            transition = self.underlying_net.get_net_node_by_id(transition_id)
            preset_ids = get_preset_ids(transition)
            # Get conditions with place_id in preset_ids
            conditions = []
            for place_id in preset_ids:
                conditions.append(self.conditions[place_id])

            # Try all combinations of conditions for co-relation
            combs = product(*conditions)
            for comb in combs:
                comb_set = frozenset(comb)
                pe = PossibleExtension(
                    transition.properties[NetProperties.ID.name], comb_set,
                    None)
                if pe not in self.extensions_seen and pe not in self.possible_extensions.in_pq:
                    if self.is_co_related(comb_set):
                        local_configuration = self.get_full_configuration_from_marking(
                            comb_set)
                        pe.local_configuration = local_configuration
                        new_possible_extensions.add(pe)
                        # break

        return new_possible_extensions

    def astar(self, cost_mapping):
        while len(self.possible_extensions.pq) > 0:
            astar_item: AStarItem = self.possible_extensions.pop()
            print(self.possible_extensions._visited)
            self.extensions_seen.add(astar_item.pe)

            # This event is a cut-off event
            if len(
                    self.cut_off_events.intersection(
                        astar_item.pe.local_configuration.nodes)) != 0:
                continue

            # Do the  extension
            added_event, added_conditions = self.extension_to_bp_node(
                astar_item.pe, astar_item.g)

            # TODO make this nicer
            # We allow only a single place as final marking
            if len(added_conditions) == 1:
                if self.underlying_net.get_net_node_by_id(
                        list(added_conditions)
                    [0].net_place_id) in self.underlying_net.fm:
                    print(
                        f"Found alignment with g {astar_item.g} and f {astar_item.f}"
                    )
                    if astar_item.g >= 1000:
                        raise Exception
                    return added_conditions

            # Compute the  new  PE
            new_possible_extensions = self.compute_pe(added_conditions)

            # Compute new costs for each PE
            new_possible_extensions_with_cost = [
                self.pe_to_astar_search(x, cost_mapping)
                for x in new_possible_extensions
            ]
            # Add the PEs with cost onto the priority queue
            self.possible_extensions.push_many(
                new_possible_extensions_with_cost)

            # Check for cut-offs
            if self.is_cut_off(added_event):
                self.cut_off_events.add(added_event)


def main():
    total_traces = 0
    trad_q = []
    trad_v = []
    unf_q = []
    unf_v = []
    fit = []
    cost_mapping = {
        MoveTypes.LOG.name: 1000,
        MoveTypes.MODEL.name: 1000,
        MoveTypes.SYNC.name: 0,
        MoveTypes.MODEL_SILENT.name: 1,
        MoveTypes.DUMMY.name: 0
    }
    # model_net, model_im, model_fm = import_from_tpn("./inthelarge/prAm6.tpn")
    # xes_df = pm4py.read_xes("./inthelarge/prAm6.xes")
    # model_net, model_im, model_fm = pm4py.read_pnml(
    #     "./banktransfer/model/original/banktransfer_opennet.pnml", True)
    # xes_df = pm4py.read_xes("./banktransfer/logs/2000-all-nonoise.xes")
    xes_df = pm4py.read_xes("data/Sepsis Cases - Event Log.xes")
    model_net, model_im, model_fm = discover_petri_net_inductive(
        xes_df, noise_threshold=0)

    xes_el = convert_to_event_log(format_dataframe(xes_df))
    view_petri_net(model_net, model_im, model_fm)
    for trace in xes_el:
        total_traces += 1
        # print(total_traces)
        # trace: pm4py.objects.log.obj.Trace = xes_el[i]
        # for e in trace:
        #     if e["concept:name"] == "ER Sepsis Triage":
        #         e["concept:name"] = "DEVIATION"

        # trad_alignment = conformance_diagnostics_alignments(
        #     trace,
        #     model_net,
        #     model_im,
        #     model_fm,
        #     variant_str="Variants.VERSION_DIJKSTRA_NO_HEURISTICS")
        # print(trad_alignment)
        # trad_q.append(trad_alignment["queued_states"])
        # trad_v.append(trad_alignment["visited_states"])
        # fit.append(trad_alignment["fitness"])

        trace_net, trace_net_im, trace_net_fm = construct_trace_net(
            trace, "concept:name", "concept:name")

        sync_net, sync_im, sync_fm = construct_synchronous_product(
            model_net, model_im, model_fm, trace_net, trace_net_im,
            trace_net_fm)
        if len(sync_net.transitions) < 300:
            continue
        # if len(sync_net.transitions) > 150:
        #     view_petri_net(sync_net)
        print(len(sync_net.transitions))
        sync_net_extended = ExtendedNet(sync_net, sync_im, sync_fm)

        # bp = BranchingProcess(sync_net_extended)

        # bp.initialize_from_initial_marking(cost_mapping)

        # alignment = bp.astar(cost_mapping)

        # print(
        #     f"Qd {bp.possible_extensions._queued}, Vd {bp.possible_extensions._visited}"
        # )
        # unf_q.append(bp.possible_extensions._queued)
        # unf_v.append(bp.possible_extensions._visited)

        # new_configuration = bp.get_full_configuration_from_marking(alignment)
        # configuration_net = bp.convert_nodes_to_net(new_configuration.nodes)
        # view_petri_net(configuration_net)
        bp2 = BranchingProcessStandard(sync_net_extended)
        bp2.initialize_from_initial_marking(cost_mapping)
        alignment2 = bp2.astar(cost_mapping)
        print(
            f"Qd {bp2.possible_extensions._queued}, Vd {bp2.possible_extensions._visited}"
        )
        unf_q.append(bp2.possible_extensions._queued)
        unf_v.append(bp2.possible_extensions._visited)
        # # nodes = set()
        # # for k, v in bp2.conditions_by_place_id.items():
        # #     nodes.update(v)
        # # for k, v in bp2.events.items():
        # #     nodes.update(v)
        # # view_petri_net(bp2.convert_nodes_to_net(nodes))

        # new_configuration2 = bp2.get_full_configuration_from_marking(
        #     alignment2)
        # configuration_net2 = bp2.convert_nodes_to_net(new_configuration2.nodes)
        # view_petri_net(configuration_net2)
    results = {}
    results["total_traces"] = total_traces
    results["fit"] = fit
    results["trad_q"] = trad_q
    results["trad_v"] = trad_v
    results["unf_q"] = unf_q
    results["unf_v"] = unf_v

    with open("results/dump.json", "w") as f:
        import json
        rstr = json.dumps(results, indent=4)
        f.write(rstr)

        # view_petri_net(configuration_net)
        # view_petri_net(trace_net)
    # net, im, fm = build_petri_net("testnet_no_cycles.csv")
    # view_petri_net(net)
    # extended_net = ExtendedNet(net, im, fm)
    # bp = BranchingProcess(extended_net)
    # bp.initialize_from_initial_marking()

    # alignment = bp.extend_naive()
    # new_configuration = bp.get_full_configuration_from_marking(alignment)
    # configuration_net = bp.convert_nodes_to_net(new_configuration.nodes)
    # view_petri_net(configuration_net)
    # final_conditions = bp.find_all_final_conditions()
    # for condition in final_conditions:
    #     new_configuration = bp.get_full_configuration_from_marking({condition})
    #     configuration_net = bp.convert_nodes_to_net(new_configuration.nodes)
    #     view_petri_net(configuration_net)

    # df = pd.read_csv("testnet_complex.csv", sep=",")
    # df_trace = df.loc[df["CaseID"] == "c5"]
    # df_trace = df_trace.drop([22])
    # print(df_trace)

    # model, model_im, model_fm = construct_net(df)
    # trace_net, trace_net_im, trace_net_fm = construct_net(df_trace)
    # sync_net, sync_im, sync_fm = construct_synchronous_product(
    #     model, model_im, model_fm, trace_net, trace_net_im, trace_net_fm)
    # view_petri_net(sync_net, sync_im, sync_fm)
    # el = convert_to_event_log(
    #     format_dataframe(df_trace,
    #                      activity_key="Activity",
    #                      case_id="CaseID",
    #                      timestamp_key="Timestamp"))
    # # view_petri_net(s, si, sf)
    # sync_net_extended = ExtendedNet(sync_net, sync_im, sync_fm)
    # bp = BranchingProcess(sync_net_extended)

    # cost_mapping = {
    #     MoveTypes.LOG.name: 1000,
    #     MoveTypes.MODEL.name: 1000,
    #     MoveTypes.SYNC.name: 0,
    #     MoveTypes.MODEL_SILENT.name: 1,
    #     MoveTypes.DUMMY.name: 0
    # }

    # bp.initialize_from_initial_marking(cost_mapping)

    # alignment = bp.extend_naive(cost_mapping)
    # print(
    #     f"Qd {bp.possible_extensions._queued}, Vd {bp.possible_extensions._visited}"
    # )
    # new_configuration = bp.get_full_configuration_from_marking(alignment)
    # configuration_net = bp.convert_nodes_to_net(new_configuration.nodes)
    # view_petri_net(configuration_net)

    # print(
    #     conformance_diagnostics_alignments(
    #         df_trace,
    #         model,
    #         model_im,
    #         model_fm,
    #         activity_key="Activity",
    #         case_id_key="CaseID",
    #         timestamp_key="Timestamp",
    #         variant_str="Variants.VERSION_DIJKSTRA_NO_HEURISTICS"))

    # final_conditions = bp.find_all_final_conditions()
    # for condition in final_conditions:
    #     new_configuration = bp.get_full_configuration_from_marking({condition})
    #     configuration_net = bp.convert_nodes_to_net(new_configuration.nodes)
    #     view_petri_net(configuration_net)


if __name__ == "__main__":
    # main()
    with cProfile.Profile() as pr:
        main()
    pr.dump_stats("results/dump.prof")
