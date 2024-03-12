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


class PriorityQueue():

    def __init__(self, item_type) -> None:
        self.item_type = item_type
        self.pq: heapq[item_type] = []

    def pop(self):
        return heapq.heappop(self.pq)

    def push(self, item):
        # if not isinstance(item, self.item_type):
        #     raise TypeError
        # TODO push only items that are not in the pq yet
        # TODO also update items that are already in the pq
        heapq.heappush(self.pq, item)

    def push_many(self, items: set):
        for item in items:
            self.push(item)


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


class LightweightConfiguration:

    def __init__(self, fm: set[Condition]) -> None:
        # TODO Check this IM stuff?
        # self.im = []
        self.fm = fm

    # TODO this doesn't hold behaviourally I think, only for reachability
    # def __eq__(self, other: object) -> bool:
    #     if isinstance(other, (LightweightConfiguration, Configuration)):
    #         return self.fm == other.fm
    #     return NotImplemented


class Configuration(LightweightConfiguration):

    def __init__(self, fm: set[Condition]) -> None:
        super().__init__(fm)
        self.im: set[Condition] = set()
        self.nodes: set[Condition | Event] = set()

    # TODO unsure about this
    # def __eq__(self, other: object) -> bool:
    #     return super().__eq__(other)


class BranchingProcess:

    def __init__(self, net: ExtendedNet) -> None:
        # A branching process is a set of nodes (conditions and events)
        self.nodes: set[Condition | Event] = set()
        self.possible_extensions = PriorityQueue(
            tuple[int, int, tuple[TransitionID, frozenset[Condition]]])
        # TODO optimize this
        # We track which extensions we have seen as a tuple (net transition ID, set of conditions)
        self.extensions_seen: set[tuple[TransitionID,
                                        frozenset[Condition]]] = set()

        self.co_sets: set[frozenset[Condition]] = set()

        # A branching process has an underlying PetriNet
        self.underlying_net: ExtendedNet = net

    def initialize_from_initial_marking(self, cost_mapping):
        co_set = set()

        for place in self.underlying_net.im:
            condition = Condition(place.properties[NetProperties.ID])
            self.nodes.add(condition)
            co_set.add(condition)

        self.co_sets.add(frozenset(co_set))
        new_possible_extensions = self.compute_pe({frozenset(co_set)})
        # TODO make proper cost function
        new_possible_extensions_with_cost = [
            (cost_mapping[self.underlying_net.get_net_node_by_id(
                x[0]).properties[NetProperties.MOVE_TYPE.name]], id(x), x)
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
                # TODO remove old co_set???
        self.co_sets.update(new_co_sets)
        return new_co_sets

    def compute_pe(
        self, new_co_sets: set[frozenset[Condition]]
    ) -> set[tuple[TransitionID, frozenset[Condition]]]:
        # TODO picking things from pe should be done according to the search strategy
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
                    pe = (transition.properties[NetProperties.ID],
                          conditions_matching_preset)
                    if pe not in self.extensions_seen:
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

    def a_star(self):
        initial_configuration = self.configurations.pop()
        # Make a priority queue of configurations
        open = []

        heapq.heappush(open,
                       (0.0, id(initial_configuration), initial_configuration))
        found = False

        # Marking + set of transitions
        closed = []
        while len(open) > 0:
            p = heapq.heappop(open)
            f = p[0]
            popped_configuration = p[2]
            net_marking_ids = self.bp_marking_to_net_marking_ids(
                popped_configuration.fm)

            # TODO we want to get a list of combinations of transitions which can fire together, that also fixes or branches
            enabled_transition_ids = self.underlying_net.get_enabled_net_transition_ids(
                net_marking_ids)

            if len(enabled_transition_ids) == 0:
                print(f"Finished configuration with cost {f}")
                self.finished_configurations.add(popped_configuration)
                return

            # Check if any transitions would cause an OR branch to occur
            or_branches = self.underlying_net.check_or_branch(
                enabled_transition_ids, net_marking_ids)

            # Use a list instead of a set because we will create duplicate configurations
            # This is okay, because after firing them, they will no longer be duplicates
            configuration_transition_ids_pairs = []

            # Add the initial configuration with no tranistions to fire, we will update this later
            # TODO does this make sense?
            configuration_transition_ids_pairs.append(
                [popped_configuration, set()])

            for transition_id in or_branches:
                # Make a duplicate of our current configuration's marking
                new_fm = copy(popped_configuration.fm)

                # Create a new configuration with the marking
                new_configuration = LightweightConfiguration(new_fm)

                # Add it to the list of new configurations with the mutually exclusive transtion to fire
                configuration_transition_ids_pairs.append(
                    [new_configuration, {transition_id}])

                # In any other configuration we don't want to fire this transition again, so we remove it from the enabled_transitions set
                enabled_transition_ids.remove(transition_id)

            # We extend each set of transition ids to fire in our configuration_transition_ids_pairs
            for _, transition_ids in configuration_transition_ids_pairs:
                transition_ids.update(enabled_transition_ids)

            # Keep only new configurations with enabled transitions
            configuration_transition_ids_pairs = [
                pair for pair in configuration_transition_ids_pairs
                if len(pair[1]) > 0
            ]

            # Now for the search part, we fire only the configuration with the lowest expected cost, then put everything on the queue

            # Remove all items that are already in closed
            already_in_closed = []
            for pair in configuration_transition_ids_pairs:
                if (pair[0].fm, pair[1]) in closed:
                    already_in_closed.append(pair)

            for pair in already_in_closed:
                configuration_transition_ids_pairs.remove(pair)

            lowest_expected_cost_pair = min(configuration_transition_ids_pairs,
                                            key=lambda pair: len(pair[1]))

            # Update our closed list because we've already done this marking + transition combo
            # TODO check if this actually works
            closed.append((lowest_expected_cost_pair[0].fm,
                           lowest_expected_cost_pair[1]))

            # Remove it from the confgiuration_transition_ids_pairs so we don't accidentally add it again even though we already processed this one
            configuration_transition_ids_pairs.remove(
                lowest_expected_cost_pair)

            # Fire all transitions
            for transition_id in lowest_expected_cost_pair[1]:
                nodes_to_add = self.fire_lightweight_configuration(
                    lowest_expected_cost_pair[0], transition_id)
                self.nodes = self.nodes.union(nodes_to_add)

            # Update the cost of this configuration and put it on the priority queue
            new_f = f + len(lowest_expected_cost_pair[1])

            # Put the item on the priority queue
            heapq.heappush(open, (new_f, id(
                lowest_expected_cost_pair[0]), lowest_expected_cost_pair[0]))

            # Put the remaining configurations back on the priority queue with the original f as their cost
            # TODO update costs of things already in open?
            # TODO can we simplify like this?
            #       These new configurations are the same as the popped configuration
            #       So if there are any remaining configurations, we simply add the popped configuration back on the heapq
            #       We can do this because we only track the configuration anyway, no configuration - enabled transition pairs here
            if len(configuration_transition_ids_pairs) > 0:
                already_in_open = False
                for conf in open:
                    if conf[2].fm == popped_configuration.fm:
                        # Already in open
                        already_in_open = True
                        # TODO update cost?
                if not already_in_open:
                    heapq.heappush(open, p)
            # for pair in configuration_transition_ids_pairs:
            #     already_in_open = False
            #     # TODO this is very slow
            #     for thing in open:
            #         if thing[2].fm == pair[0].fm:
            #             already_in_open = True
            #             # TODO update cost?
            #     if not already_in_open:
            #         heapq.heappush(open, (f, id(pair[0]), pair[0]))

        # In our new configuration we want to fire all transitions that are NOT mutually exclusive, plus the transition that WAS mutually exclusive
        # The non-mutual exclusive transitions are stored in the enabled_transitions

        # So, each transition has a cost
        # This means we should fire transitions 1 by 1
        # 	If we're smart and optimize we can probably fire multiple at the same time, but this is too complicated for now
        # When there are mutually exclusive transitions, we want to make new configurations like usual
        # So this means, that when checking for which transitions we can fire, we should check over ALL current configurations
        # 	Obviously a bit inefficient right now, but we can optimize later
        # Then, by A-star, we fire the one with lowest cost

        # This doesn't sound totally right
        # 	Our optimization should come from the fact that we can fire concurrent transitions faster than in a state-transition system
        # Our cost needs to be based on extending a whole configuration
        # 	This makes our implementation a bit easier too I think
        # With mutually exclusive transitions this gets a bit weird though
        # So for each mutually exclusive transition we make a new configuration, but we modify the enabled transition set to not include the other mutually exclusive transitions?
        # Then we can compute the total cost of firing the configuration (the mutually exclusive transition as well as all other possibly enabled transitions)

    def extend_naive(self, cost_mapping):
        # TODO cycles currently generate infinite configurations, fix this!
        # Start from the configuration final markings (for now randomly, later best first approach)
        # Find the enabled transitions
        # If an OR branch would occur, make a new configuration for each branch and store the configuration

        while len(self.possible_extensions.pq) > 0:
            cost, _, pe = self.possible_extensions.pop()
            # Do the  extension
            added_event, added_conditions = self.extension_to_bp_node(pe)

            # TODO make this nicer
            # We allow only a single place as final marking
            if len(added_conditions) == 1:
                if self.underlying_net.get_net_node_by_id(
                        list(added_conditions)
                    [0].net_place_id) in self.underlying_net.fm:
                    print(f"Found alignment with cost {cost}")
                    return added_conditions
            #  Compute  the new  co-sets
            new_co_sets = self.update_co_set(added_event, added_conditions)
            #  Compute the  new  PE
            new_possible_extensions = self.compute_pe(new_co_sets)
            # TODO proper cost function
            new_possible_extensions_with_cost = [
                (cost + cost_mapping[self.underlying_net.get_net_node_by_id(
                    x[0]).properties[NetProperties.MOVE_TYPE.name]], id(x), x)
                for x in new_possible_extensions
            ]
            # TODO think of where to put extensions_seen
            self.possible_extensions.push_many(
                new_possible_extensions_with_cost)
            self.extensions_seen.update((new_possible_extensions))

    def extension_to_bp_node(
        self, extension: tuple[TransitionID, frozenset[Configuration]]
    ) -> tuple[Event, set[Condition]]:

        new_event = Event(extension[0], set(extension[1]))
        self.nodes.add(new_event)
        postset_ids = get_postset_ids(
            self.underlying_net.get_net_node_by_id(extension[0]))
        added_conditions = set()
        for place_id in postset_ids:
            new_condition = Condition(place_id, new_event)
            self.nodes.add(new_condition)
            added_conditions.add(new_condition)

        return new_event, added_conditions

    # TODO make this more efficient
    # TODO doesn't work if the FM is multiple places
    # We don't know which events are "final" in the BP
    # We try to find them here by checking which conditions are not reffered to by any event
    def find_all_final_conditions(self):
        final_conditions = set()

        net_fm_ids = [
            x.properties[NetProperties.ID] for x in self.underlying_net.fm
        ]

        for node in self.nodes:
            if isinstance(node, Condition) and node.net_place_id in net_fm_ids:
                final_conditions.add(node)
        return final_conditions

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

    # Fire a configuration without updating the branching process
    def fire_lightweight_configuration(
            self, configuration: LightweightConfiguration,
            transition_id: TransitionID) -> set[Event | Condition]:

        # Return the events and conditions that should be added to the BP later
        ret: set[Event | Condition] = set()
        # For each condition in the configuration's final makring, check if it is in the preset
        # If it is, it's an input condition for our new event, and we can "consume" the token
        preset_ids = get_preset_ids(
            self.underlying_net.get_net_node_by_id(transition_id))
        input_conditions = set()
        for condition in configuration.fm:
            if condition.net_place_id in preset_ids:
                # Add this condition as an input condition for our new event
                input_conditions.add(condition)

        # Consume the tokens at once so we don't adjust our loop condition
        configuration.fm = configuration.fm.difference(input_conditions)
        # Create the new event
        new_event = Event(transition_id, input_conditions)
        ret.add(new_event)

        # Add the postset to the configuration and the branching process
        postset_ids = get_postset_ids(
            self.underlying_net.get_net_node_by_id(transition_id))
        for place_id in postset_ids:
            new_condition = Condition(place_id, new_event)
            configuration.fm.add(new_condition)
            ret.add(new_condition)

        return ret

    def convert_nodes_to_net(self,
                             nodes: set[Condition, Event] = None) -> PetriNet:
        if nodes is None:
            nodes = self.nodes
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

    # Check if a configuration has reached the EXACT final marking of the underlying net
    def check_configuration_complete(
            self, configuration: LightweightConfiguration | Configuration):
        net_marking_ids = []
        configuration_marking_ids = []
        for p in self.underlying_net.fm:
            net_marking_ids.append(p.properties[NetProperties.ID])

        for c in configuration.fm:
            configuration_marking_ids.append(c.net_place_id)

        return Counter(net_marking_ids) == Counter(configuration_marking_ids)


def get_postset(
    node: PetriNet.Place | PetriNet.Transition
) -> set[PetriNet.Place | PetriNet.Transition]:
    ret = set()

    for arc in node.out_arcs:
        ret.add(arc.target)

    return ret


def get_postset_ids(
        node: PetriNet.Place | PetriNet.Transition
) -> set[PlaceID | TransitionID]:
    ret = set()

    for arc in node.out_arcs:
        ret.add(arc.target.properties[NetProperties.ID])

    return ret


def get_preset(
    node: PetriNet.Place | PetriNet.Transition
) -> set[PetriNet.Place | PetriNet.Transition]:
    ret = set()

    for arc in node.in_arcs:
        ret.add(arc.source)

    return ret


def get_preset_ids(
        node: PetriNet.Place | PetriNet.Transition
) -> set[PlaceID | TransitionID]:
    ret = set()

    for arc in node.in_arcs:
        ret.add(arc.source.properties[NetProperties.ID])

    return ret


# def sort_petrinet_node_list(nodes: list[PetriNet.Place | PetriNet.Transition]):

#     def key(node: PetriNet.Place | PetriNet.Transition):
#         return node.properties["id"]

#     nodes.sort(key)


def build_petri_net(filepath: str) -> tuple[PetriNet, Marking, Marking]:
    """Discover a Petri Net from an event log

    Args:
        filepath (str): The event log in CSV fomat

    Returns:
        tuple[PetriNet, Marking, Marking]: The discovered Petri Net, initial marking, and final marking
    """
    df = pd.read_csv(filepath, sep=",")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    net, im, fm = discover_petri_net_inductive(df,
                                               activity_key="Activity",
                                               case_id_key="CaseID",
                                               timestamp_key="Timestamp")
    return net, im, fm


def main():
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

    df = pd.read_csv("testnet_complex.csv", sep=",")
    df_trace = df.loc[df["CaseID"] == "c1"]
    # df_trace = df_trace.drop([0, 2])
    print(df_trace)

    model, model_im, model_fm = construct_net(df)
    trace_net, trace_net_im, trace_net_fm = construct_net(df_trace)
    sync_net, sync_im, sync_fm = construct_synchronous_product(
        model, model_im, model_fm, trace_net, trace_net_im, trace_net_fm)
    view_petri_net(sync_net, sync_im, sync_fm)
    sync_net_extended = ExtendedNet(sync_net, sync_im, sync_fm)
    bp = BranchingProcess(sync_net_extended)

    cost_mapping = {
        MoveTypes.LOG.name: 1,
        MoveTypes.MODEL.name: 1,
        MoveTypes.SYNC.name: 0,
        MoveTypes.MODEL_SILENT.name: 0,
        MoveTypes.DUMMY.name: 0
    }

    bp.initialize_from_initial_marking(cost_mapping)

    alignment = bp.extend_naive(cost_mapping)
    new_configuration = bp.get_full_configuration_from_marking(alignment)
    configuration_net = bp.convert_nodes_to_net(new_configuration.nodes)
    view_petri_net(configuration_net)

    # final_conditions = bp.find_all_final_conditions()
    # for condition in final_conditions:
    #     new_configuration = bp.get_full_configuration_from_marking({condition})
    #     configuration_net = bp.convert_nodes_to_net(new_configuration.nodes)
    #     view_petri_net(configuration_net)


if __name__ == "__main__":
    main()
    # with cProfile.Profile() as pr:
    #     main()
    # pr.dump_stats("program.prof")
