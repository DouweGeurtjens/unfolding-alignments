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

# TODO ALWAYS REMEMBER TO KEEP IN MIND THAT CONDITIONS/EVENTS ARE NOT DIRECTLY COMPARABLE TO PLACES/TRANSITIONS
# TODO THIS MEANS THAT SET OPERATIONS WILL ONLY WORK BETWEEN THE SAME TYPE, SO NO conditions.intersection.places

# TODO there is some issue with the synchronous product and invisible transitions
# TODO set unions take a lot of time, perhaps it's possible (in some cases) to use a linked list for faster extension?
PlaceID: TypeAlias = int
TransitionID: TypeAlias = int


# A place in a branching process
class Condition:

    def __init__(self, net_place_id: int, input_event=None) -> None:
        # A condition has a single input event
        self.input_event = input_event

        # The place in the PetriNet model this condition refers to
        self.net_place_id = net_place_id

        self.id = IDGEN.generate_id()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Condition):
            return self.id == other.id
        return NotImplemented

    def __hash__(self) -> int:
        return self.id


# A transition in a branching process
class Event:

    def __init__(self, net_transition_id: int,
                 input_conditions: set[Condition], local_configuration,
                 local_configuration_cost) -> None:
        # An event can have multiple input conditions
        self.input_conditions = input_conditions

        # The transition in the PetriNet model this event refers to
        self.net_transition_id = net_transition_id

        # Bookkeep the local configuration and it's cost to increase performance
        # TODO local configuration does not include the event in the nodes, but the cost does include the cost of the event
        # TODO is that correct?
        self.local_configuration = local_configuration
        self.local_configuration_cost = local_configuration_cost
        self.id = IDGEN.generate_id()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Event):
            return self.id == other.id
        return NotImplemented

    def __hash__(self) -> int:
        return self.id


# The items being put on the priority queue
class AStarItem:

    def __init__(self, f, g, h, pe) -> None:
        self.f = f
        self.g = g
        self.h = h
        self.pe = pe

    def __lt__(self, other):
        # Tiebreaker
        if self.f == other.f:
            return id(self) < id(other)

        return self.f < other.f


# A priority queue that can check in O(1) time if an item is in it
class PriorityQueue:

    def __init__(self) -> None:
        self.pq: heapq[AStarItem] = []
        self.in_pq: set[PossibleExtension] = set()
        self._queued = 0
        self._visited = 0

    def pop(self):
        ret = heapq.heappop(self.pq)
        self.in_pq.remove(ret.pe)
        self._visited += 1
        return ret

    def push(self, item: AStarItem):
        # TODO also update items that are already in the pq
        heapq.heappush(self.pq, item)
        self.in_pq.add(item.pe)
        self._queued += 1

    def push_many(self, items: set):
        for item in items:
            self.push(item)


class Configuration:

    def __init__(self) -> None:
        self.nodes: set[Condition | Event] = set()

    # TODO unsure about this
    # def __eq__(self, other: object) -> bool:
    #     return super().__eq__(other)


# A possible extension to a branching process
class PossibleExtension:

    def __init__(self, transition_id: TransitionID,
                 conditions: frozenset[Condition],
                 local_configuration: Configuration) -> None:
        # The transition in the underlying net
        self.transition_id = transition_id

        # The input conditions associated with this extension
        self.conditions = conditions
        self.local_configuration = local_configuration

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PossibleExtension):
            return (self.transition_id,
                    self.conditions) == (other.transition_id, other.conditions)
        return NotImplemented

    def __hash__(self) -> int:
        return (self.transition_id, self.conditions).__hash__()


class BranchingProcess:

    def __init__(self, net: ExtendedNet) -> None:
        self.possible_extensions = PriorityQueue()

        # A BP is a set of conditions and events (nodes)
        # We track events by their marking, so we can more quickly find cut-off events
        self.events: dict[frozenset[Counter], list[Event]] = {}
        # We track conditions by their place ID
        self.conditions: dict[PlaceID, set[Condition]] = {}

        # TODO optimize this
        # We track which extensions we have seen as a tuple (net transition ID, set of conditions)
        self.extensions_seen: set[tuple[TransitionID,
                                        frozenset[Condition]]] = set()
        self.cut_off_events: set[Event] = set()

        # A branching process has an underlying PetriNet
        self.underlying_net: ExtendedNet = net

    def initialize_from_initial_marking(self, cost_mapping):
        pass

    def fire_configuration(self, configuration: Configuration):
        presets = Counter()
        postsets = Counter()
        marking = Counter()
        for node in configuration.nodes:
            if isinstance(node, Event):
                net_transition = self.underlying_net.get_net_node_by_id(
                    node.net_transition_id)
                presets.update(get_preset_ids(net_transition))
                postsets.update(get_postset_ids(net_transition))

        postsets.subtract(presets)
        for k in postsets:
            if postsets[k] > 0:
                marking[k] = postsets[k]

        return marking

    def is_cut_off(self, event: Event) -> bool:
        local_configuration_marking = frozenset(
            self.fire_configuration(event.local_configuration).items())

        # TODO this is slow, how do we optimize?
        # TODO use a dict of marking -> [event] so we can look up by marking?
        if local_configuration_marking not in self.events:
            return False

        for bp_event in self.events[local_configuration_marking]:
            if bp_event.local_configuration_cost < event.local_configuration_cost:
                return True

        return False

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

    # Just some pseudocode
    # We start with a configuration with the IM
    # Pop a configuration from the heapq
    # Check all enabled transitions
    # Check which transitions are mutually exclusive
    #   For each of these, create a configuration - transition pair
    #   The mutually exclusive transition can now be removed from the enabled transition list
    #       This is because these transitions can exist only in this specific configuration
    # For all configuration - transition pairs, add the remaining enabled transitions (these are concurrently enabled regardless of chosen mutually exclusive transition)
    # Remove all pairs where no transitions are enabled
    #   This can happen when there are only mutually exclusive transitions enabled
    # Also, remove all pairs that are already in the closed list (we've seen this marking + enabled transitions before)
    # From the remaining pairs, get the configuration with the lowest expected cost to reach the final marking (heuristic)
    # Fire this transition, add the result into the heapq with the updated cost after firing
    # The rest of the configurations are added to the heapq with the cost of the popped configuration (we didn't fire them so their cost didn't change)
    #   Only do this if it's not already on the heapq
    #   TODO see todo for simplification
    # When a configuration reaches the FM, it's done and we found the least-cost unfolding
    def pe_to_astar_search(self, pe: PossibleExtension, cost_mapping):
        # NOTE the configuration does not yet include the possible extension
        configuration_cost = self.get_configuration_cost(
            pe.local_configuration, cost_mapping)

        # Get the cost of this transition
        pe_cost = cost_mapping[self.underlying_net.get_net_node_by_id(
            pe.transition_id).properties[NetProperties.MOVE_TYPE.name]]

        # Sum the cost of the configuration and the cost of this transition, since this transition is not yet in the configuration
        g = configuration_cost + pe_cost
        h = 0
        f = g + h

        return AStarItem(f, g, h, pe)

    def astar(self, cost_mapping):
        pass

    def extension_to_bp_node(self, extension: PossibleExtension,
                             cost) -> tuple[Event, set[Condition]]:
        # NOTE the configuration does not yet include the possible extension
        new_event = Event(extension.transition_id, set(extension.conditions),
                          extension.local_configuration, cost)

        # Add the new_event to it's own local configuration
        new_event.local_configuration.nodes.add(new_event)

        # Fire the local configuration so we can store it keyed by marking
        # TODO bookkeep this maybe?
        net_marking = frozenset(
            self.fire_configuration(new_event.local_configuration).items())

        # Add event keyed by marking
        if net_marking in self.events:
            self.events[net_marking].append(new_event)
        else:
            self.events[net_marking] = [new_event]

        # Add the conditions corresponding to the places of the postset of the transition corresponding to the event
        postset_ids = get_postset_ids(
            self.underlying_net.get_net_node_by_id(extension.transition_id))
        added_conditions = set()
        for place_id in postset_ids:
            new_condition = Condition(place_id, new_event)
            self.conditions[place_id].add(new_condition)
            added_conditions.add(new_condition)

        return new_event, added_conditions

    # TODO move this to configuration class?
    def get_full_configuration_from_marking(
            self, marking: set[Condition]) -> Configuration:
        configuration = Configuration()
        stack = deque()
        for condition in marking:
            if condition not in configuration.nodes:
                stack.append(condition)
                configuration.nodes.add(condition)
        while len(stack) != 0:
            item = stack.pop()
            configuration.nodes.add(item)
            if isinstance(item, Event):
                for condition in item.input_conditions:
                    if condition not in configuration.nodes:
                        stack.append(condition)
            if isinstance(item, Condition):
                if item.input_event is not None:
                    if item.input_event not in configuration.nodes:
                        stack.append(item.input_event)

        return configuration

    def bp_marking_to_net_marking_ids(
            self, bp_marking: set[Condition]) -> set[PlaceID]:
        # Given a marking in the branching process, find the corresponding marking in the underlying net
        ret = set()

        for condition in bp_marking:
            ret.add(condition.net_place_id)

        return ret

    def convert_nodes_to_net(self,
                             nodes: set[Condition, Event] = None) -> PetriNet:
        ret = PetriNet()

        for node in nodes:
            if isinstance(node, Event):
                net_transition = self.underlying_net.get_net_node_by_id(
                    node.net_transition_id)
                new_transition = PetriNet.Transition(node.id,
                                                     net_transition.label)

                for input_condition in node.input_conditions:
                    new_place = PetriNet.Place(input_condition.id)
                    ret.places.add(new_place)
                    add_arc_from_to(new_place, new_transition, ret)

                ret.transitions.add(new_transition)

        for node in nodes:
            if isinstance(node, Condition):
                if node.input_event:
                    target = None
                    source = None
                    for p in ret.places:
                        if p.name == node.id:
                            target = p
                    for t in ret.transitions:
                        if t.name == node.input_event.id:
                            source = t
                    if target and source:
                        add_arc_from_to(source, target, ret)

        return ret


if __name__ == "__main__":
    pass
