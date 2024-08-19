from copy import deepcopy, copy
import pandas as pd
from pm4py import PetriNet, Marking, discover_petri_net_inductive, view_petri_net, format_dataframe, convert_to_event_log, conformance_diagnostics_alignments
# Use TypeAlias because the type notation in introduced in 3.12 isn't support by yapf yet
from typing import TypeAlias
import cProfile, pstats
from collections import Counter, deque
import heapq
from operator import itemgetter
# TODO fix imports
from petrinet import *
from itertools import product, combinations
from coset_base import *
from random import Random
import time
from coset_standard import BranchingProcessStandard
# TODO ALWAYS REMEMBER TO KEEP IN MIND THAT CONDITIONS/EVENTS ARE NOT DIRECTLY COMPARABLE TO PLACES/TRANSITIONS
# TODO THIS MEANS THAT SET OPERATIONS WILL ONLY WORK BETWEEN THE SAME TYPE, SO NO conditions.intersection.places

# TODO there is some issue with the synchronous product and invisible transitions
# TODO set unions take a lot of time, perhaps it's possible (in some cases) to use a linked list for faster extension?
PlaceID: TypeAlias = int
TransitionID: TypeAlias = int


class BranchingProcessStream(BranchingProcess):

    def __init__(self, net: ExtendedSyncNetStreaming, starting_trace) -> None:
        super().__init__(net)
        # When we find a prefix alignment we don't stop, we instead add it to this dict
        self.iteration = 0
        self.prefix_alignments = {}
        # This should be a trace containing only the first event on the stream
        self.trace = starting_trace

        # For evaluation
        self.unf_q_per_iteration = {}
        self.unf_v_per_iteration = {}
        # TODO how to impmlement?
        # self.unf_elapsed_time_per_iteration = {}

    def re_initialize(self, transition_ids_to_check):
        # TODO don't need to check all transitions maybe? only those affected by the
        for place in self.underlying_net.places:
            if place.properties[NetProperties.ID.name] not in self.conditions:
                self.conditions[place.properties[
                    NetProperties.ID.name]] = set()

        new_possible_extensions = self.compute_pe(transition_ids_to_check)
        new_possible_extensions_with_cost = [
            self.pe_to_astar_search(x) for x in new_possible_extensions
        ]
        self.possible_extensions.push_many(new_possible_extensions_with_cost)
        # Update the seen extensions
        # TODO check if this makes any sense
        # self.extensions_seen.update((new_possible_extensions))

    def check_goal_state(self, added_conditions, check_final_marking):
        #First check if we found a prefix
        if not check_final_marking:
            for added_condition in added_conditions:
                if self.underlying_net.get_net_node_by_id(
                        added_condition.net_place_id
                ) in self.underlying_net.trace_fm:
                    # print("Found prefix alignment")
                    return True, "prefix"

        # TODO make a way to check if a trace is complete
        # Only check for the true FM if our trace is complete
        if check_final_marking:
            if len(added_conditions) == 1:
                if self.underlying_net.get_net_node_by_id(
                        list(added_conditions)
                    [0].net_place_id) in self.underlying_net.fm:
                    # print(f"Found full alignment")
                    return True, "full"

        return False, None

    def store_prefix_alignment(self, alignment, cost):
        # Only store the prefix alignment if it's lower cost than the previous
        if self.iteration not in self.prefix_alignments:
            self.prefix_alignments[self.iteration] = (alignment, cost)

            # Evaluation
            self.unf_q_per_iteration[
                self.iteration] = self.possible_extensions._queued
            self.unf_v_per_iteration[
                self.iteration] = self.possible_extensions._visited
        # TODO isn't the first we find optimal by definition of the search?
        else:
            previous_cost = self.prefix_alignments[self.iteration][1]
            if cost < previous_cost:
                self.prefix_alignments[self.iteration] = (alignment, cost)
                # Update evaluation if we finder lower cost
                self.unf_q_per_iteration[
                    self.iteration] = self.possible_extensions._queued
                self.unf_v_per_iteration[
                    self.iteration] = self.possible_extensions._visited

    def astar(self, check_final_marking=False):
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

            # We allow only a single place as final marking
            goal_reached, prefix_or_full = self.check_goal_state(
                added_conditions, check_final_marking)
            if goal_reached:
                # TODO only store least-cost prefix
                if prefix_or_full == "prefix":
                    self.store_prefix_alignment(added_conditions, astar_item.g)
                    return None
                if prefix_or_full == "full":
                    return added_conditions, astar_item.g

            transition_ids_to_check = set()
            for condition in added_conditions:
                # Basically any transition that is in the postset of a place in the marking of the co-set
                transition_ids_to_check.update(
                    get_postset_ids(
                        self.underlying_net.get_net_node_by_id(
                            condition.net_place_id)))

            # Compute the  new  PE
            new_possible_extensions = self.compute_pe(transition_ids_to_check)

            # Compute new costs for each PE
            new_possible_extensions_with_cost = [
                self.pe_to_astar_search(x) for x in new_possible_extensions
            ]
            # Add the PEs with cost onto the priority queue
            self.possible_extensions.push_many(
                new_possible_extensions_with_cost)

            # Check for cut-offs
            if self.is_cut_off(added_event):
                self.cut_off_events.add(added_event)

    def alignment_streaming(self, model_net, model_im, model_fm, stream):
        # The trace simply simulates a stream by picking one event at a time
        # We initliased the BP with an event on the trace so we must initialise and look for a prefix alignment first
        self.initialise_from_initial_marking()
        alignment = self.astar()
        while len(stream) != 0:
            event = stream._list.pop(0)
            # Add this event onto our bp
            self.trace.append(event)

            trace_net, trace_net_im, trace_net_fm = construct_trace_net(
                self.trace, "concept:name", "concept:name")

            sync_net, sync_im, sync_fm, cost_function = construct_synchronous_product(
                model_net, model_im, model_fm, trace_net, trace_net_im,
                trace_net_fm)
            # view_petri_net(sync_net)
            update_new_synchronous_product_streaming(self.underlying_net,
                                                     sync_net)
            extended_net = ExtendedSyncNetStreaming(sync_net, sync_im, sync_fm,
                                                    trace_net_fm,
                                                    cost_function)

            old_transition_ids = set(x.properties[NetProperties.ID.name]
                                     for x in self.underlying_net.transitions)
            new_transition_ids = set(x.properties[NetProperties.ID.name]
                                     for x in extended_net.transitions)
            transition_ids_to_check = new_transition_ids.difference(
                old_transition_ids)

            self.underlying_net = extended_net
            self.re_initialize(transition_ids_to_check)
            self.iteration += 1
            # Compute prefix alignment
            alignment = self.astar()
            # # If we return something we're done
            # if alignment:
            #     return alignment

        # Stream ran out, continue processing to get the full alignment
        alignment = self.astar(True)
        # print("done")
        return alignment


if __name__ == "__main__":
    pass
    # # model_net, model_im, model_fm = import_from_tpn("./inthelarge/prAm6.tpn")
    # # xes_df = pm4py.read_xes("./inthelarge/prAm6.xes")
    # # model_net, model_im, model_fm = pm4py.read_pnml(
    # #     "./banktransfer/model/original/banktransfer_opennet.pnml", True)
    # # xes_df = pm4py.read_xes("./banktransfer/logs/2000-all-nonoise.xes")
    # xes_df = pm4py.read_xes("data/Sepsis Cases - Event Log.xes")
    # model_net, model_im, model_fm = discover_petri_net_inductive(
    #     xes_df, noise_threshold=0)
    # xes_el = convert_to_event_log(format_dataframe(xes_df))
    # # config = bp.get_full_configuration_from_marking(alignment)
    # # alignment_net = bp.convert_nodes_to_net(config.nodes)
    # # view_petri_net(alignment_net)
    # for trace in xes_el:
    #     event = trace.__list.pop(0)
    #     stream = pm4py.objects.log.obj.Trace(trace)
    #     # Add this event onto our bp
    #     trace_net, trace_net_im, trace_net_fm = construct_trace_net(
    #         stream, "concept:name", "concept:name")
    #     sync_net, sync_im, sync_fm = construct_synchronous_product(
    #         model_net, model_im, model_fm, trace_net, trace_net_im,
    #         trace_net_fm)
    #     extended_net = ExtendedSyncNetStreaming(sync_net, sync_im, sync_fm,
    #                                             trace_net_fm)
    #     bp = BranchingProcessStream(extended_net, stream)
    #     final_alignment = bp.alignment_streaming(model_net, model_im, model_fm)
    #     conf = bp.get_full_configuration_from_marking(final_alignment)
    #     net = bp.convert_nodes_to_net(conf.nodes)
    #     view_petri_net(net)
    #     for k, v in bp.prefix_alignments.items():
    #         print(k)
    #         alignment = v[0]
    #         conf = bp.get_full_configuration_from_marking(alignment)
    #         net = bp.convert_nodes_to_net(conf.nodes)
    #         # view_petri_net(net)
    #     print(
    #         f"Qd {bp.possible_extensions._queued}, Vd {bp.possible_extensions._visited}"
    #     )
