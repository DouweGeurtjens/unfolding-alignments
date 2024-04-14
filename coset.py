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
from itertools import product

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
        self.conditions: set[Condition] = set()

        # TODO optimize this
        # We track which extensions we have seen as a tuple (net transition ID, set of conditions)
        self.extensions_seen: set[tuple[TransitionID,
                                        frozenset[Condition]]] = set()

        self.co_sets: set[frozenset[Condition]] = set()
        self.cut_off_events: set[Event] = set()

        # A branching process has an underlying PetriNet
        self.underlying_net: ExtendedNet = net

        # New coset approach
        self.conditions_by_place_id: dict[PlaceID, set[Condition]] = {}

    def initialize_from_initial_marking2(self, cost_mapping):
        for place in self.underlying_net.places:
            self.conditions_by_place_id[place.properties[
                NetProperties.ID.name]] = set()
        added_conditions = set()
        for place in self.underlying_net.im:
            condition = Condition(place.properties[NetProperties.ID.name])
            self.conditions_by_place_id[place.properties[
                NetProperties.ID.name]].add(condition)
            added_conditions.add(condition)

        # self.co_sets.add(frozenset(co_set))
        new_possible_extensions = self.compute_pe2(added_conditions)
        # TODO make proper cost function
        new_possible_extensions_with_cost = [
            self.pe_to_astar_search(x, cost_mapping)
            for x in new_possible_extensions
        ]
        self.possible_extensions.push_many(new_possible_extensions_with_cost)
        # Update the seen extensions
        # TODO check if this makes any sense
        # self.extensions_seen.update((new_possible_extensions))

    def initialize_from_initial_marking(self, cost_mapping):
        co_set = set()

        for place in self.underlying_net.im:
            condition = Condition(place.properties[NetProperties.ID.name])
            self.conditions.add(condition)
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

    def is_co_related(self, conditions: frozenset[Condition]):
        # Start from our conditions, follow down the local configuration via a queue of events
        # Mark the input conditions of an event with the event it was visited from
        # If we find a condition which was already visited from a different event there is a conflict
        # Meanwhile, check if the conditions we are marking are in the set we are testing for co-relation
        # If one is, they are in causal relation
        queue: list[Event] = []
        marks = {}
        visited_from = -1

        # Starting conditions are visited from nothings, so we use a negative integer
        for c in conditions:
            if c.input_event is not None:
                if c in marks:
                    if marks[c] == visited_from:
                        return False
                queue.append(c.input_event)
                marks[c] = visited_from
                visited_from -= 1

        while len(queue) > 0:
            item = queue.pop(0)
            visited_from = item
            for c in item.input_conditions:
                # Causal so not in co-relation
                if c in conditions:
                    return False
                if c.input_event is not None:
                    # This condition has been visited before
                    if c in marks:
                        # It has been visited from a different transition
                        if marks[c] != visited_from:
                            return False

                    queue.append(c.input_event)
                    marks[c] = visited_from

        return True

    def compute_pe2(
        self, added_conditions: set[Condition]
    ) -> set[tuple[TransitionID, frozenset[Condition]]]:
        # For each transition in the net, find a co-set which is a superset
        # Those transitions are possible extensions
        new_possible_extensions = set()

        # TODO looping over all transitions is a bit slow, how do we optimize this?
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
                conditions.append(self.conditions_by_place_id[place_id])
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
                        break

        return new_possible_extensions

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

    def astar2(self, cost_mapping):
        while len(self.possible_extensions.pq) > 0:
            astar_item: AStarItem = self.possible_extensions.pop()
            # print(self.possible_extensions._visited)
            self.extensions_seen.add(astar_item.pe)

            # This event is a cut-off event
            if len(
                    self.cut_off_events.intersection(
                        astar_item.pe.local_configuration.nodes)) != 0:
                continue

            # Do the  extension
            added_event, added_conditions = self.extension_to_bp_node2(
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

            # Compute the  new  PE
            new_possible_extensions = self.compute_pe2(added_conditions)

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

    def extension_to_bp_node2(self, extension: PossibleExtension,
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
            self.conditions_by_place_id[place_id].add(new_condition)
            added_conditions.add(new_condition)

        return new_event, added_conditions

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
            self.conditions.add(new_condition)
            added_conditions.add(new_condition)

        return new_event, added_conditions

    # TODO move this to configuration class?
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
                    new_place = PetriNet.Place(input_condition.id)
                    new_arc = PetriNet.Arc(new_place, new_transition)

                    ret.places.add(new_place)
                    ret.arcs.add(new_arc)

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
                        new_arc = PetriNet.Arc(source, target)
                        ret.arcs.add(new_arc)

        return ret


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
    xes_df = pm4py.read_xes("Sepsis Cases - Event Log.xes")
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

        # view_petri_net(sync_net)

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
        bp2 = BranchingProcess(sync_net_extended)
        bp2.initialize_from_initial_marking2(cost_mapping)
        alignment2 = bp2.astar2(cost_mapping)
        print(
            f"Qd {bp2.possible_extensions._queued}, Vd {bp2.possible_extensions._visited}"
        )
        unf_q.append(bp2.possible_extensions._queued)
        unf_v.append(bp2.possible_extensions._visited)

        new_configuration2 = bp2.get_full_configuration_from_marking(
            alignment2)
        configuration_net2 = bp2.convert_nodes_to_net(new_configuration2.nodes)
        view_petri_net(configuration_net2)
    results = {}
    results["total_traces"] = total_traces
    results["fit"] = fit
    results["trad_q"] = trad_q
    results["trad_v"] = trad_v
    results["unf_q"] = unf_q
    results["unf_v"] = unf_v

    with open("dump.json", "w") as f:
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
    pr.dump_stats("dump.prof")
