import pandas as pd
from pm4py import PetriNet, Marking, discover_petri_net_inductive

# TODO check wherever lists are used if sets are possible
# TODO ALWAYS REMEMBER TO KEEP IN MIND THAT CONDITIONS/EVENTS ARE NOT DIRECTLY COMPARABLE TO PLACES/TRANSITIONS
# TODO THIS MEANS THAT SET OPERATIONS WILL ONLY WORK BETWEEN THE SAME TYPE, SO NO conditions.intersection.places

# class SimpleTransition:

#     def __init__(self, transition: PetriNet.Transition) -> None:
#         self._underlying_transition = transition

# class SimplePlace:

#     def __init__(self, place: PetriNet.Place) -> None:
#         self._underlying_place = place


class IDGenerator:

    def __init__(self) -> None:
        self.counter = 0

    def generate_id(self) -> int:
        self.counter += 1
        return self.counter


IDGEN = IDGenerator()


# A place in a branching process
class Condition:
    # TODO overload __hash__ if needed
    def __init__(self, net_place, input_event: Event | None = None) -> None:
        # A condition has a single input event
        self.input_event = input_event
        # The place in the PetriNet model this condition refers to
        self.net_place = net_place
        self.id = IDGEN.generate_id()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Condition):
            return self.id == other.id
        return NotImplemented


# A transition in a branching process
class Event:
    # TODO overload __hash__ if needed
    def __init__(self, net_transition: PetriNet.Transition,
                 input_conditions: set[Condition]) -> None:
        # An event can have multiple input conditions
        self.input_conditions = input_conditions
        # The transition in the PetriNet model this event refers to
        self.net_transition = net_transition
        self.id = IDGEN.generate_id()

    def __eq__(self, other: object) -> bool:
        if isinstance(other, Event):
            return self.id == other.id
        return NotImplemented


class LightweightConfiguration:

    def __init__(self, fm: list[Condition]) -> None:
        # TODO Check this IM stuff?
        # self.im = []
        self.fm = fm


class Configuration(LightweightConfiguration):

    def __init__(self, fm: list[Condition]) -> None:
        super().__init__(fm)
        self.im: list[Condition] = []
        self.nodes: set[Condition | Event] = set()


class BranchingProcess:

    def __init__(self, net: PetriNet) -> None:
        # A branching process is a set of nodes (conditions and events)
        # TODO set, list, queue or something else?
        self.nodes: set[Condition | Event] = set()

        # Track the final nodes of a configuration (we can work backwards to retrieve the full configuration)
        #
        # TODO we need to keep track of configurations, so whenver we branch (1 place enables two transitions?) we should store both events as the endpoint of our configuration
        self.configurations: set[LightweightConfiguration] = set()

        # A branching process has an underlying PetriNet
        self.underlying_net: PetriNet = net

        # The initial marking of the branching process, set later
        # TODO, for now, assume single IM for easy reasoning
        self.im = []

    def initialize_from_initial_marking(self, im: Marking):
        configuration_fm = []
        for place in im:
            condition = Condition(place.properties["id"])
            configuration_fm.append(condition)
            self.nodes.add(condition)
            self.im.append(condition)

        configuration = Configuration(configuration_fm)
        self.configurations.add(configuration)

    def extend_naive(self):
        # Start from the configuration endpoints
        # Take an endpoint (for now randomly, later best first approach)
        # Find the enabled transitions
        # If an OR branch would occur, do each branch and store the configuration

        configuration = self.configurations.pop()

        net_marking = self.bp_marking_to_net_marking(configuration.fm)

        enabled_transitions = get_enabled_net_transitions(net_marking)

        # Check if any transitions would cause an OR branch to occur
        or_branches = check_or_branch(enabled_transitions, net_marking)

        # TODO Fix this later
        for place, transitions in or_branches.items():
            # TODO Do some special stuff here
            # TODO Make a configuration for all place->transition combinations here, then fire those transitions
            # TODO Then remove the transitions from enabled_transitions (they have already been fired)
            pass

        for enabled_transition in enabled_transitions:
            # TODO When a transition is enabled, we want to "fire" it on the configuration
            pass

    def get_configuration_from_marking(
            self, marking: list[Condition]) -> Configuration:
        configuration = Configuration(marking)

        for condition in marking:
            self._get_configuration_from_marking_helper(
                condition, configuration)

        return configuration

    def _get_configuration_from_marking_helper(self, condition: Condition,
                                               configuration: Configuration):
        configuration.nodes.add(condition)
        if condition.input_event is None:
            configuration.im.append(condition)
        else:
            configuration.nodes.add(condition.input_event)

            for c in condition.input_event.input_conditions:
                self._get_configuration_from_marking_helper(c, configuration)

    def bp_marking_to_net_marking(
            self, marking: list[Condition]) -> list[PetriNet.Place]:
        # Given a marking in the branching process, find the corresponding marking in the underlying net
        ret = []

        for condition in marking:
            ret.append(condition.net_place)

        return ret

    # TODO This currently modifies the branching process, maybe make this more explicit?
    def fire_configuration(self, configuration: LightweightConfiguration
                           | Configuration, transition: PetriNet.Transition):
        # For each condition in the configuration's final makring, check if it is in the preset
        # If it is, it's an input condition for our new event, and we can "consume" the token
        preset = get_preset(transition)
        input_conditions = set()
        for condition in configuration.fm:
            if condition.net_place in preset:
                # Add this condition as an input condition for our new event
                input_conditions.add(condition)
                # Consume the token
                configuration.fm.remove(condition)

        # Create the new event
        event = Event(transition, input_conditions)
        self.nodes.add(event)

        # Add the postset to the configuration and the branching process
        postset = get_postset(transition.net_transition)
        for place in postset:
            new_condition = Condition(place, event)
            configuration.fm.append(new_condition)
            self.nodes.add(condition)


def check_or_branch(
    enabled_transitions: set[PetriNet.Transition], marking: set[PetriNet.Place]
) -> dict[PetriNet.Place, set[PetriNet.Transition]]:
    ret = {}

    for place in marking:
        postset = get_postset(place)
        if len(postset) > 1:
            # This place can cause multiple transitions to be enabled
            # We note down which transitions by checking the enabled_transtions and marking each one that is present in the postset of the place
            mutually_exclusive_transitions = enabled_transitions.intersection(
                postset)
            ret[place] = mutually_exclusive_transitions

    return ret


def get_enabled_net_transitions(
        marking: set[PetriNet.Place]) -> set[PetriNet.Transition]:
    # Given a marking in the underlying net, get all enabled transitions
    # For each place in the marking, get the postset transitions
    # Then, for each transition, check if its preset places are ALL present in the given marking
    # Only those transitions are enabled
    ret = set()
    transitions = []
    for place in marking:
        transitions.extend(get_postset(place))

    for transition in transitions:
        enabled = is_transition_enabled(transition, marking)
        if enabled:
            ret.add(transition)

    return ret


def is_transition_enabled(transition: PetriNet.Transition,
                          marking: set[PetriNet.Place]) -> bool:
    # Checks if a given transition is enabled
    preset = get_preset(transition)
    return preset.issubset(marking)


def get_postset(
    node: PetriNet.Place | PetriNet.Transition
) -> set[PetriNet.Place | PetriNet.Transition]:
    ret = set()

    for arc in node.out_arcs:
        ret.add(arc.target)

    return ret


def get_preset(
    node: PetriNet.Place | PetriNet.Transition
) -> set[PetriNet.Place | PetriNet.Transition]:
    ret = set()

    for arc in node.in_arcs:
        ret.add(arc.source)

    return ret


def sort_petrinet_node_list(nodes: list[PetriNet.Place | PetriNet.Transition]):

    def key(node: PetriNet.Place | PetriNet.Transition):
        return node.properties["id"]

    nodes.sort(key)


# def petrinet_node_equality(
#         node1: PetriNet.Place | PetriNet.Transition,
#         node2: PetriNet.Place | PetriNet.Transition) -> bool:
#     return node1.properties["id"] == node2.properties["id"]


# Extend the PetriNet with some additional information like explicit IDs
def extend_net(net: PetriNet):
    for p in net.places:
        p.properties["id"] = IDGEN.generate_id()
    for t in net.transitions:
        t.properties["id"] = IDGEN.generate_id()


def build_petri_net(filepath: str) -> tuple[PetriNet, Marking, Marking]:
    df = pd.read_csv(filepath, sep=",")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    net, im, fm = discover_petri_net_inductive(df,
                                               activity_key="Activity",
                                               case_id_key="CaseID",
                                               timestamp_key="Timestamp")
    return net, im, fm


def main():
    net, im, fm = build_petri_net("testnet_no_cycles.csv")
    extend_net(net)
    bp = BranchingProcess(net)
    bp.initialize_from_initial_marking(im)


if __name__ == "__main__":
    main()
