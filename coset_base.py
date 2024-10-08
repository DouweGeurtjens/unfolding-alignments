from copy import deepcopy, copy
import sys
import pandas as pd
from pm4py import PetriNet, Marking, discover_petri_net_inductive, view_petri_net, format_dataframe, convert_to_event_log, conformance_diagnostics_alignments
# Use TypeAlias because the type notation in introduced in 3.12 isn't support by yapf yet
from typing import TypeAlias
import cProfile, pstats
from collections import Counter, deque
import heapq
from operator import itemgetter
from petrinet import *
from itertools import product, combinations

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
        heapq.heappush(self.pq, item)
        self.in_pq.add(item.pe)
        self._queued += 1

    def push_many(self, items: set):
        for item in items:
            self.push(item)


class Configuration:

    def __init__(self) -> None:
        self.nodes: set[Condition | Event] = set()


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

    def __init__(self,
                 net: ExtendedSyncNet | ExtendedSyncNetStreaming) -> None:
        self.possible_extensions = PriorityQueue()

        # A BP is a set of conditions and events (nodes)
        # We track events by their marking, so we can more quickly find cut-off events
        self.events: dict[frozenset[Counter], list[Event]] = {}
        # We track conditions by their place ID
        self.conditions: dict[PlaceID, set[Condition]] = {}

        # We track which extensions we have seen as a tuple (net transition ID, set of conditions)
        self.extensions_seen: set[tuple[TransitionID,
                                        frozenset[Condition]]] = set()
        self.cut_off_events: set[Event] = set()

        # A branching process has an underlying PetriNet
        self.underlying_net: ExtendedSyncNet | ExtendedSyncNetStreaming = net

    def is_co_related(self, conditions: frozenset[Condition]):
        # Start from our conditions, follow down the local configuration via a queue of events
        # Mark the input conditions of an event with the event it was visited from
        # If we find a condition which was already visited from a different event there is a conflict
        # Meanwhile, check if the conditions we are marking are in the set we are testing for co-relation
        # If one is, they are in causal relation
        queue: deque[Event] = deque()
        marks = {}

        # Starting conditions are visited from nothings, so we use a negative integer
        for c in conditions:
            if c.input_event is not None:
                queue.append(c.input_event)

        while len(queue) > 0:
            item = queue.popleft()
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

    def initialise_from_initial_marking(self):
        for place in self.underlying_net.places:
            self.conditions[place.properties[NetProperties.ID.name]] = set()

        added_conditions = set()
        for place in self.underlying_net.im:
            condition = Condition(place.properties[NetProperties.ID.name])
            self.conditions[place.properties[NetProperties.ID.name]].add(
                condition)
            added_conditions.add(condition)

        transition_ids_to_check = set()
        for condition in added_conditions:
            # Basically any transition that is in the postset of a place in the marking of the co-set
            transition_ids_to_check.update(
                get_postset_ids(
                    self.underlying_net.get_net_node_by_id(
                        condition.net_place_id)))

        new_possible_extensions = self.compute_pe(transition_ids_to_check)

        new_possible_extensions_with_cost = [
            self.pe_to_astar_search(x) for x in new_possible_extensions
        ]
        self.possible_extensions.push_many(new_possible_extensions_with_cost)

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

        if local_configuration_marking not in self.events:
            return False

        for bp_event in self.events[local_configuration_marking]:
            if bp_event.local_configuration_cost < event.local_configuration_cost:
                return True

        return False

    def get_configuration_cost(self, configuration: Configuration):
        configuration_cost = 0
        for node in configuration.nodes:
            if isinstance(node, Event):
                configuration_cost += self.underlying_net.cost_function[
                    self.underlying_net.get_net_node_by_id(
                        node.net_transition_id)]
        return configuration_cost

    def pe_to_astar_search(self, pe: PossibleExtension):
        # NOTE the configuration does not yet include the possible extension
        configuration_cost = self.get_configuration_cost(
            pe.local_configuration)

        # Get the cost of this transition
        pe_cost = self.underlying_net.cost_function[
            self.underlying_net.get_net_node_by_id(pe.transition_id)]

        configuration_marking = self.fire_configuration(pe.local_configuration)

        transition_preset = get_preset(
            self.underlying_net.get_net_node_by_id(pe.transition_id))
        transition_postset = get_postset(
            self.underlying_net.get_net_node_by_id(pe.transition_id))

        im = Marking()
        for c in configuration_marking:
            if self.underlying_net.get_net_node_by_id(
                    c) not in transition_preset:
                im[self.underlying_net.get_net_node_by_id(c)] = 1
        for c in transition_postset:
            im[c] = 1

        # Sum the cost of the configuration and the cost of this transition, since this transition is not yet in the configuration
        g = configuration_cost + pe_cost
        # Set to zero to get results without heuristic
        h = pm4py.solve_marking_equation(self.underlying_net, im,
                                         self.underlying_net.fm,
                                         self.underlying_net.cost_function)
        if h == None:
            # No heuristic possible?
            print("Bad")
        f = g + h

        return AStarItem(f, g, h, pe)

    def compute_pe(
        self, transition_ids_to_check
    ) -> set[tuple[TransitionID, frozenset[Condition]]]:
        # For each transition in the net, find a co-set which is a superset
        # Those transitions are possible extensions
        new_possible_extensions = set()

        for transition_id in transition_ids_to_check:
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

        return new_possible_extensions

    def astar(self):
        pass

    def extension_to_bp_node(self, extension: PossibleExtension,
                             cost) -> tuple[Event, set[Condition]]:
        # NOTE the configuration does not yet include the possible extension
        new_event = Event(extension.transition_id, set(extension.conditions),
                          extension.local_configuration, cost)

        # Add the new_event to it's own local configuration
        new_event.local_configuration.nodes.add(new_event)

        # Fire the local configuration so we can store it keyed by marking
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

    def get_full_configuration_from_marking(
            self, marking: set[Condition]) -> Configuration:
        configuration = Configuration()

        for condition in marking:
            self._get_full_configuration_from_marking_helper(
                condition, configuration)

        return configuration

    def _get_full_configuration_from_marking_helper(
            self, condition: Condition, configuration: Configuration):
        # No need to further explore here down this branch
        if condition in configuration.nodes:
            return

        configuration.nodes.add(condition)
        if condition.input_event is not None:
            configuration.nodes.add(condition.input_event)

            for c in condition.input_event.input_conditions:
                self._get_full_configuration_from_marking_helper(
                    c, configuration)

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
                    source = None
                    for p in ret.places:
                        if p.name == input_condition.id:
                            source = p
                    if not source:
                        source = PetriNet.Place(input_condition.id)

                    ret.places.add(source)
                    add_arc_from_to(source, new_transition, ret)

                ret.transitions.add(new_transition)

        for node in nodes:
            if isinstance(node, Condition):
                if node.input_event:
                    target = None
                    source = None
                    for p in ret.places:
                        if p.name == node.id:
                            target = p

                    if not target:
                        target = PetriNet.Place(node.id)
                        ret.places.add(target)

                    for t in ret.transitions:
                        if t.name == node.input_event.id:
                            source = t

                    if target and source:
                        add_arc_from_to(source, target, ret)

        return ret


if __name__ == "__main__":
    pass
