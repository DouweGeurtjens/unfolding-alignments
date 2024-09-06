from copy import deepcopy, copy
import pandas as pd
from pm4py import PetriNet, Marking, discover_petri_net_inductive, view_petri_net, format_dataframe, convert_to_event_log, conformance_diagnostics_alignments
# Use TypeAlias because the type notation in introduced in 3.12 isn't support by yapf yet
from typing import TypeAlias
import cProfile, pstats
import heapq
from operator import itemgetter
from petrinet import *
from itertools import product, combinations
from coset_base import *

PlaceID: TypeAlias = int
TransitionID: TypeAlias = int


class BranchingProcessStandard(BranchingProcess):

    def __init__(self, net: ExtendedSyncNet) -> None:
        super().__init__(net)

    def astar(self):
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
            if len(added_conditions) == 1:
                if self.underlying_net.get_net_node_by_id(
                        list(added_conditions)
                    [0].net_place_id) in self.underlying_net.fm:
                    # print(f"Found alignment with cost {astar_item.g}")
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
                self.pe_to_astar_search(x) for x in new_possible_extensions
            ]
            # Add the PEs with cost onto the priority queue
            self.possible_extensions.push_many(
                new_possible_extensions_with_cost)

            # Check for cut-offs
            if self.is_cut_off(added_event):
                self.cut_off_events.add(added_event)


if __name__ == "__main__":
    pass
