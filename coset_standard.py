from copy import deepcopy, copy
import pandas as pd
from pm4py import PetriNet, Marking, discover_petri_net_inductive, view_petri_net, format_dataframe, convert_to_event_log, conformance_diagnostics_alignments
# Use TypeAlias because the type notation in introduced in 3.12 isn't support by yapf yet
from typing import TypeAlias
import cProfile, pstats
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

    def __init__(self, net: ExtendedSyncNet) -> None:
        super().__init__(net)

    def astar(self, cost_mapping):
        while len(self.possible_extensions.pq) > 0:
            astar_item: AStarItem = self.possible_extensions.pop()
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
                    print(f"Found alignment with cost {astar_item.g}")
                    return added_conditions, astar_item.g

            # Compute the  new  PE
            transition_ids_to_check = set()
            for condition in added_conditions:
                # Basically any transition that is in the postset of a place in the marking of the co-set
                transition_ids_to_check.update(
                    get_postset_ids(
                        self.underlying_net.get_net_node_by_id(
                            condition.net_place_id)))

            new_possible_extensions = self.compute_pe(transition_ids_to_check)

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


if __name__ == "__main__":
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
    # config = bp.get_full_configuration_from_marking(alignment)
    # alignment_net = bp.convert_nodes_to_net(config.nodes)
    # view_petri_net(alignment_net)
    for trace in xes_el:
        # rm_e = None
        # for e in trace:
        #     if e["concept:name"] == "ER Registration":
        #         rm_e = e
        # trace._list.remove(rm_e)
        trace_net, trace_net_im, trace_net_fm = construct_trace_net(
            trace, "concept:name", "concept:name")
        sync_net, sync_im, sync_fm = construct_synchronous_product(
            model_net, model_im, model_fm, trace_net, trace_net_im,
            trace_net_fm)
        extended_net = ExtendedSyncNet(sync_net, sync_im, sync_fm)
        print(len(sync_net.transitions))
        bp = BranchingProcessStandard(extended_net)
        bp.initialize_from_initial_marking(cost_mapping)
        with cProfile.Profile() as pr:
            alignment, _ = bp.astar(cost_mapping)
            conf = bp.get_full_configuration_from_marking(alignment)
            view_petri_net(bp.convert_nodes_to_net(conf.nodes))
        pr.dump_stats("prof.prof")
        print(
            f"Qd {bp.possible_extensions._queued}, Vd {bp.possible_extensions._visited}"
        )
        break
