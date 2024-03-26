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
                 input_conditions: set[Condition]) -> None:
        # An event can have multiple input conditions
        self.input_conditions = input_conditions

        # The transition in the PetriNet model this event refers to
        self.net_transition_id = net_transition_id

        self.id = IDGEN.generate_id()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Event):
            return self.id == other.id
        return NotImplemented

    def __hash__(self) -> int:
        return self.id


# A possible extension to a branching process
class PossibleExtension:

    def __init__(self, transition_id: TransitionID,
                 conditions: frozenset[Condition]) -> None:
        # The transition in the underlying net
        self.transition_id = transition_id

        # The input conditions associated with this extension
        self.conditions = conditions

    def __eq__(self, other: object) -> bool:
        if isinstance(other, PossibleExtension):
            return (self.transition_id,
                    self.conditions) == (other.transition_id, other.conditions)
        return NotImplemented

    def __hash__(self) -> int:
        return (self.transition_id, self.conditions).__hash__()


# The items being put on the priority queue
class AStarItem:

    def __init__(self, f, g, h, pe: PossibleExtension) -> None:
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

    def __init__(self, fm: set[Condition]) -> None:
        super().__init__(fm)
        self.im: set[Condition] = set()
        self.nodes: set[Condition | Event] = set()

    # TODO unsure about this
    # def __eq__(self, other: object) -> bool:
    #     return super().__eq__(other)


class UnfoldingAlignment:

    def __init__(self) -> None:
        pass


class BranchingProcess:

    def __init__(self, net: ExtendedNet) -> None:
        self.possible_extensions = PriorityQueue()

        # A BP is a set of conditions and events (nodes)
        self.events: set[Event] = set()
        self.conditions: set[Condition] = set()

        # TODO optimize this
        # We track which extensions we have seen as a tuple (net transition ID, set of conditions)
        self.extensions_seen: set[tuple[TransitionID,
                                        frozenset[Condition]]] = set()

        self.co_sets: set[frozenset[Condition]] = set()
        self.cut_off_events: set[Event] = set()

        # A branching process has an underlying PetriNet
        self.underlying_net: ExtendedNet = net

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
        self.extensions_seen.update((new_possible_extensions))

    def update_co_set(
            self, new_event: Event,
            new_conditions: set[Condition]) -> set[frozenset[Condition]]:
        new_co_sets = set()
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

    def is_cut_off(self, event: Event, cost_mapping) -> bool:
        local_configuration = self.get_full_configuration_from_marking(
            event.input_conditions)
        local_configuration_cost = self.get_configuration_cost(
            local_configuration, cost_mapping)
        local_configuration_marking = self.fire_configuration(
            local_configuration)

        for bp_event in self.events:
            bp_local_configuration = self.get_full_configuration_from_marking(
                bp_event.input_conditions)
            bp_local_configuration_cost = self.get_configuration_cost(
                bp_local_configuration, cost_mapping)
            bp_local_configuration_marking = self.fire_configuration(
                bp_local_configuration)
            if bp_local_configuration_marking == local_configuration_marking and bp_local_configuration_cost < local_configuration_cost:
                return True

        return False

    def compute_pe(
        self, new_co_sets: set[frozenset[Condition]]
    ) -> set[tuple[TransitionID, frozenset[Condition]]]:
        # For each transition in the net, find a co-set which is a superset
        # Those transitions are possible extensions
        new_possible_extensions = set()

        for transition in self.underlying_net.transitions:
            preset_ids = get_preset_ids(transition)
            for co_set in new_co_sets:
                co_set_place_ids = set([x.net_place_id for x in co_set])
                if preset_ids.issubset(co_set_place_ids):
                    conditions_matching_preset = frozenset(
                        [x for x in co_set if x.net_place_id in preset_ids])

                    # TODO Check if this in the the BP already
                    # TODO checking whether something is in the BP should be based on (net_transition, conditions), so using self.nodes probably won't work unless we make a special hash
                    # TODO update heap conditions properly
                    pe = PossibleExtension(
                        transition.properties[NetProperties.ID.name],
                        conditions_matching_preset)
                    if pe not in self.extensions_seen and pe not in self.possible_extensions.in_pq:
                        new_possible_extensions.add(pe)

        return new_possible_extensions

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

    def pe_to_astar_search(self, pe: PossibleExtension, cost_mapping):
        pe_cost = cost_mapping[self.underlying_net.get_net_node_by_id(
            pe.transition_id).properties[NetProperties.MOVE_TYPE.name]]
        # TODO origin.g is not enough, we need the cost of pe.conditions
        # TODO to compute the cost of the pe.conditions, track the cost of the local configuration as a set so we don't duplicate
        conf = self.get_full_configuration_from_marking(pe.conditions)
        configuration_cost = self.get_configuration_cost(conf, cost_mapping)
        g = configuration_cost + pe_cost
        # TODO make this a better heuristic
        # TODO we don't know our configurations marking, so that makes heuristics much more difficult
        h = 0
        f = g + h
        return AStarItem(f, g, h, pe)

    def astar(self, cost_mapping):
        # TODO cycles currently generate infinite configurations, fix this!
        # Start from the configuration final markings (for now randomly, later best first approach)
        # Find the enabled transitions
        # If an OR branch would occur, make a new configuration for each branch and store the configuration

        while len(self.possible_extensions.pq) > 0:
            astar_item: AStarItem = self.possible_extensions.pop()
            self.extensions_seen.add(astar_item.pe)

            local_configuration = self.get_full_configuration_from_marking(
                astar_item.pe.conditions)
            # This event is a cut-off event
            if len(self.cut_off_events.intersection(
                    local_configuration.nodes)) != 0:
                continue

            # Do the  extension
            added_event, added_conditions = self.extension_to_bp_node(
                astar_item.pe)

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
            if self.is_cut_off(added_event, cost_mapping):
                self.cut_off_events.add(added_event)

    def extension_to_bp_node(
            self,
            extension: PossibleExtension) -> tuple[Event, set[Condition]]:

        new_event = Event(extension.transition_id, set(extension.conditions))
        self.events.add(new_event)
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
        configuration = Configuration(marking)

        for condition in marking:
            self._get_full_configuration_from_marking_helper(
                condition, configuration)

        return configuration

    def _get_full_configuration_from_marking_helper(
            self, condition: Condition, configuration: Configuration):
        configuration.nodes.add(condition)
        if condition.input_event is None:
            configuration.im.add(condition)
        else:
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
    cost_mapping = {
        MoveTypes.LOG.name: 1000,
        MoveTypes.MODEL.name: 1000,
        MoveTypes.SYNC.name: 0,
        MoveTypes.MODEL_SILENT.name: 1,
        MoveTypes.DUMMY.name: 0
    }
    bpi = pm4py.read_xes("Sepsis Cases - Event Log.xes")
    bpiel = convert_to_event_log(format_dataframe(bpi))
    bla, bla2, bla3 = discover_petri_net_inductive(bpi, noise_threshold=0.5)
    view_petri_net(bla, bla2, bla3)
    for i in range(0, 1):
        bpi_trace = bpiel[i]
        with cProfile.Profile() as pr:
            trad_alignment = conformance_diagnostics_alignments(
                bpi_trace,
                bla,
                bla2,
                bla3,
                variant_str="Variants.VERSION_DIJKSTRA_NO_HEURISTICS")
            print(trad_alignment)
        pr.dump_stats("trad.prof")

        with cProfile.Profile() as pr:
            trace_net, trace_net_im, trace_net_fm = construct_trace_net(
                bpi_trace, "concept:name", "concept:name")

            sync_net, sync_im, sync_fm = construct_synchronous_product(
                bla, bla2, bla3, trace_net, trace_net_im, trace_net_fm)

            sync_net_extended = ExtendedNet(sync_net, sync_im, sync_fm)

            bp = BranchingProcess(sync_net_extended)

            bp.initialize_from_initial_marking(cost_mapping)

            alignment = bp.astar(cost_mapping)
            new_configuration = bp.get_full_configuration_from_marking(
                alignment)
            print(
                f"Qd {bp.possible_extensions._queued}, Vd {bp.possible_extensions._visited}"
            )
        pr.dump_stats("unfold.prof")
        configuration_net = bp.convert_nodes_to_net(new_configuration.nodes)
        view_petri_net(configuration_net)
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
    main()
    # pm4py.
    # with cProfile.Profile() as pr:
    #     main()
    # pr.dump_stats("program.prof")
