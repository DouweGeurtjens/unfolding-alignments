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


class BranchingProcessBookkeep(BranchingProcess):

    def __init__(self, net: ExtendedNet) -> None:
        super().__init__(net)
        self.co_sets: set[frozenset[Condition]] = set()

    def initialize_from_initial_marking(self, cost_mapping):
        for place in self.underlying_net.places:
            self.conditions[place.properties[NetProperties.ID.name]] = set()

        co_set = set()

        for place in self.underlying_net.im:
            condition = Condition(place.properties[NetProperties.ID.name])
            self.conditions[place.properties[NetProperties.ID.name]].add(
                condition)
            co_set.add(condition)

        self.co_sets.add(frozenset(co_set))
        new_possible_extensions = self.compute_pe({frozenset(co_set)})
        # TODO make proper cost function
        new_possible_extensions_with_cost = [
            self.pe_to_astar_search(x, cost_mapping)
            for x in new_possible_extensions
        ]
        self.possible_extensions.push_many(new_possible_extensions_with_cost)
        # Update the seen extensions
        # TODO check if this makes any sense
        # self.extensions_seen.update((new_possible_extensions))

    def update_co_set(
            self, new_event: Event,
            new_conditions: set[Condition]) -> set[frozenset[Condition]]:
        new_co_sets = set()
        # TODO make powerset of all cosets, then a dict mapping powerset elements to cosets
        # then we can check much faster if something is a subset of a co_set
        # problem, memory
        # TODO also try making a mapping from condition -> set(coset) where the key is in the coset
        for co_set in self.co_sets:
            # TODO subset or equality?
            if new_event.input_conditions.issubset(co_set):
                # Preset conditions is contained in this coset, update the coset
                new_co_set = co_set.difference(
                    new_event.input_conditions).union(new_conditions)
                new_co_sets.add(frozenset(new_co_set))
        self.co_sets.update(new_co_sets)
        return new_co_sets

    def compute_pe(
        self, new_co_sets: set[frozenset[Condition]]
    ) -> set[tuple[TransitionID, frozenset[Condition]]]:
        # For each transition in the net, find a co-set which is a superset
        # Those transitions are possible extensions
        new_possible_extensions = set()

        # TODO looping over all transitions is a bit slow, how do we optimize this?
        # TODO the only transitions that can be enabled are those present in the new_co_sets postset right?
        for co_set in new_co_sets:
            # Basically the marking of the co-set
            co_set_place_ids = set([x.net_place_id for x in co_set])

            # Basically any transition that is in the postset of a place in the marking of the co-set
            co_set_postset_ids = set()

            for condition in co_set:
                co_set_postset_ids.update(
                    get_postset_ids(
                        self.underlying_net.get_net_node_by_id(
                            condition.net_place_id)))

            # For each transition in the postset of the coset places, check if it's enabled by the co-set
            for transition_id in co_set_postset_ids:
                transition = self.underlying_net.get_net_node_by_id(
                    transition_id)
                preset_ids = get_preset_ids(transition)
                if preset_ids.issubset(co_set_place_ids):
                    conditions_matching_preset = frozenset(
                        [x for x in co_set if x.net_place_id in preset_ids])
                    pe = PossibleExtension(
                        transition.properties[NetProperties.ID.name],
                        conditions_matching_preset, None)
                    if pe not in self.extensions_seen and pe not in self.possible_extensions.in_pq:
                        # Update the local configuration here so we do this expensive computation less often
                        # NOTE the configuration does not yet include this possible extension
                        local_configuration = self.get_full_configuration_from_marking(
                            set(conditions_matching_preset))
                        pe.local_configuration = local_configuration
                        new_possible_extensions.add(pe)

        return new_possible_extensions

    def get_configuration_cost(self, configuration: Configuration,
                               cost_mapping):
        configuration_cost = 0
        for node in configuration.nodes:
            if isinstance(node, Event):
                configuration_cost += cost_mapping[
                    self.underlying_net.get_net_node_by_id(
                        node.net_transition_id).properties[
                            NetProperties.MOVE_TYPE.name]]
        return configuration_cost

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
                    print(
                        f"Found alignment with g {astar_item.g} and f {astar_item.f}"
                    )
                    return added_conditions

            # Compute  the new  co-sets
            new_co_sets = self.update_co_set(added_event, added_conditions)

            # Compute the  new  PE
            new_possible_extensions = self.compute_pe(new_co_sets)

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
    pass
