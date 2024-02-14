import pandas as pd
import pm4py
from copy import deepcopy
from itertools import product

# TODO MAJOR: Currently, corresponding_places and corresponding_transitions just return a list of places/transitions
# This is not correct, because a bp can have multiple places pointing to the same underlying petri-net node
# This gets especially complicated when we try to find corresponding nodes for multiple nodes at once
# Because now we need to know which nodes belong "together"
# My current idea to fix this is to have the corresponding_whatever functions return a list of lists
# The inner lists represent a permutation of BP nodes that match the underlying net nodes
# e.g. we try to find the bp nodes corresponding to underlying net nodes [p1,p2]
# we find bp nodes [p1_1, p2_1, p1_2, p2_2], then the result of the function would be [[p1_1,p2_1],[p1_1,p2_2],[p1_2,p2_1],[p1_2,p2_2]]
# Also, currently the function allows for returning of partial matchings,
# e.g. we try to find the bp nodes corresponding to underlying net nodes [p1,p2]
# The function might return [p1] or [p2] if one or the other is not present in the BP
# This should probably not be allowed


class ExtendedPetriNet():
    # The general idea here is to make the PetriNet more suitable for unfolding
    def __init__(self,
                 net: pm4py.PetriNet = pm4py.PetriNet(),
                 im: pm4py.Marking = None,
                 fm: pm4py.Marking = None):
        self.net = net
        self.im = im
        self.fm = fm
        self.place_id_counter = 0
        self.transition_id_counter = 0
        self.arc_id_counter = 0
        self._initialize_ids()

    # TODO Overwrite the classes for Place, Transition, and Arc to do this
    def _initialize_ids(self):
        """Give a unique numerical ID to each place, transition and arc in the net

        Because the equivalence of places transitions and arcs is based on the built-in id function
        we cannot easily compare two places that represent the same "thing" without keeping references
        This is a workaround that fixes that
        """
        for place in self.net.places:
            place.properties["id"] = self.place_id_counter
            self.place_id_counter += 1

        for transition in self.net.transitions:
            transition.properties["id"] = self.transition_id_counter
            self.transition_id_counter += 1

        for arc in self.net.arcs:
            arc.properties["id"] = self.arc_id_counter
            self.arc_id_counter += 1

    def add_place_with_id(self, place: pm4py.PetriNet.Place):
        place.properties["id"] = self.place_id_counter
        self.place_id_counter += 1
        self.net.places.add(place)

    def add_transition_with_id(self, transition: pm4py.PetriNet.Transition):
        transition.properties["id"] = self.transition_id_counter
        self.transition_id_counter += 1
        self.net.transitions.add(transition)

    def add_arc_with_id(self, arc: pm4py.PetriNet.Arc):
        arc.properties["id"] = self.arc_id_counter
        self.arc_id_counter += 1
        self.net.arcs.add(arc)

    def get_postset(
        self, node: pm4py.PetriNet.Place | pm4py.PetriNet.Transition
    ) -> list[pm4py.PetriNet.Place | pm4py.PetriNet.Transition]:
        ret = []

        for arc in list(node.out_arcs):
            ret.append(arc.target)

        return ret

    def get_preset(
        self, node: pm4py.PetriNet.Place | pm4py.PetriNet.Transition
    ) -> list[pm4py.PetriNet.Place | pm4py.PetriNet.Transition]:
        ret = []

        for arc in list(node.in_arcs):
            ret.append(arc.source)

        return ret


class BranchingProcess():

    def __init__(self, bp: ExtendedPetriNet, net: ExtendedPetriNet):
        self.bp = bp
        self.underlying_net = net
        # # Keep a list of the ending nodes (no postset in the branching process), because we can only extend from those
        # self.ending_nodes = []

    def view(self):
        pm4py.view_petri_net(self.bp.net)

    def add_initial_place(self, place: pm4py.PetriNet.Place):
        """Add an intial node to the branching process

        Intial nodes, unlike other nodes, do not add any arcs, even if they have a preset

        Args:
            node (pm4py.PetriNet.Place | pm4py.PetriNet.Transition): The input node coming from the original net
        """
        new_properties = {}
        new_properties["origin_id"] = place.properties["id"]

        # TODO The name of the new node should be unique, now it is not
        new_place = pm4py.PetriNet.Place(place.name, properties=new_properties)

        self.bp.add_place_with_id(new_place)

    def get_possible_extensions(
        self,
        bp_places: list[pm4py.PetriNet.Place] = None
    ) -> list[pm4py.PetriNet.Transition]:
        """Gets the possible extensions to a branching process

        When bp_places is left as None, bp_places will be set to include all places currently in the branching process
        However, because the intent is to search in a best-first manner, where we want to extend from is influenced by the search strategy
        Therefore, we can supply the additional argument to adhere to our searching strategy by only expanding from a limited set of places

        Args:
            bp_places (list[pm4py.PetriNet.Place], optional): The places we wish to find the extension from. Defaults to None.

        Returns:
            list[pm4py.PetriNet.Transition]: _description_
        """
        if bp_places is None:
            bp_places = self.bp.net.places

        possible_extensions = []

        for transition in self.underlying_net.net.transitions:
            preset_places = self.underlying_net.get_preset(transition)

            # Get sets of places in the BP that correspond to the preset of the transition in the underlying net
            corresponding_places_combinations = self.get_corresponding_places_combinations(
                preset_places)
            
            # For each set of corresponding places, check if the combination of corresponding places->transition is in the BP already
            # If it is, it's not an extension. If not, it is a possible extension
            for cpc in corresponding_places_combinations:
                # Get the bp_transitions that correspond to the input transition corresponding to a place
                matching = {}
                for cp in cpc:
                    matching[cp] = False
                    postset_transitions = self.bp.get_postset(
                        corresponding_place)
                    for bp_transtion in postset_transitions:
                        if bp_transition.properties["origin_id"] == transition.properties["id"]:
                            matching[cp] = True    
                
                # If all places remain unmatched, then the cpc->transition combination doesn't exist yet
                no_matches = True
                for _,v in matching:
                    if v:
                        no_matches = False

                # For every bp_transition we want to know if it corresponds to our input transition (the one we are trying to extend)
                for bp_transition in bp_transitions_to_check:
                    # This cpc->transition combination is already present
                    if bp_transition.properties["origin_id"] == transition.properties["id"]:

                    for cp in cpc:
                        for arc in cp.out_arcs:
                            # It's already there, this combination is not an extension
                            if arc.target.properties["origin_id"] == transition.properties["id"]:
                                pass

            # For each possible extension, compute the downward closure
            # If we can't, it's not an extension

            # Then, check each downward closure for conflict-freeness
            # If it isn't, it's not an extension

            # Then, if we have anything remaining, it's an extension
            # We extend one at a time

            # We found a corresponding transition, this could be an extension
            # TODO question: what if we find multiple corresponding places?
            # TODO for now, we check if the preset places are contained in the corresponding places
            contained = set()
            for cp in corresponding_places:
                for pp in preset_places:
                    if cp.properties["origin_id"] == pp.properties["id"]:
                        contained.add(pp)

            if len(contained) == len(preset_places):
                # It's only an extension if the combination of this transition + the found places is not already in the BP
                extendable = True
                for corresponding_place in corresponding_places:
                    postset_transitions = self.bp.get_postset(
                        corresponding_place)
                    for bp_transition in postset_transitions:
                        # Check the target of all out_arcs of the corresponding place to see if these are connected
                        for arc in corresponding_place.out_arcs:
                            if arc.target.properties[
                                    "origin_id"] == transition.properties[
                                        "id"] and bp_transition.properties[
                                            "origin_id"] == transition.properties[
                                                "id"]:
                                # We can't add it
                                extendable = False

                if extendable:
                    possible_extensions.append(transition)

        return possible_extensions

    def extend_naive(self) -> bool:
        # Get possible extensions
        possible_extension = self.get_possible_extensions()

        # No possible extensions
        if len(possible_extension) == 0:
            return False

        possible_extension = possible_extension.pop()

        # Try to extend, keep track of what we tried to add
        extension_arcs = []

        preset_places = self.underlying_net.get_preset(possible_extension)

        # Get places in BP corresponding to transition preset so we know where to put the arcs
        corresponding_places = self.get_corresponding_places_combinations(
            preset_places)

        new_properties = {}
        new_properties["origin_id"] = possible_extension.properties["id"]

        # TODO The name of the new node should be unique, now it is not
        new_transition = pm4py.PetriNet.Transition(possible_extension.name,
                                                   properties=new_properties)

        #TODO Check here if order matters, aka can we create the arc before finishing the transition properties?
        for bp_place in corresponding_places:
            new_arc = pm4py.PetriNet.Arc(bp_place, new_transition)

            self.bp.add_arc_with_id(new_arc)
            extension_arcs.append(new_arc)
            # Mutate the bp_places arcs
            bp_place.out_arcs.add(new_arc)

            new_transition.in_arcs.add(new_arc)

        # Add the transition with all new arcs
        self.bp.add_transition_with_id(new_transition)

        # If it has a local configuration, then do the extension on the real BP
        has_local_config = self.has_local_config(new_transition)

        if has_local_config:
            # Add the transition postset to the bp
            postset = self.underlying_net.get_postset(possible_extension)
            for place in postset:
                new_place_properties = {}
                new_place_properties["origin_id"] = place.properties["id"]
                new_place = pm4py.PetriNet.Place(
                    place.name, properties=new_place_properties)
                new_postset_arc = pm4py.PetriNet.Arc(new_transition, new_place)
                new_transition.out_arcs.add(new_postset_arc)
                new_place.in_arcs.add(new_postset_arc)
                self.bp.add_place_with_id(new_place)
                self.bp.add_arc_with_id(new_postset_arc)
        else:
            # Undo the adding
            self.bp.net.transitions.remove(new_transition)
            for arc in extension_arcs:
                for bp_place in corresponding_places:
                    if arc.source == bp_place:
                        bp_place.out_arcs.remove(arc)
                self.bp.net.arcs.remove(arc)

        return True

    def get_corresponding_places_combinations(
            self,
            net_places: list[pm4py.PetriNet.Place],
            bp: ExtendedPetriNet = None):
        if bp is None:
            bp = self.bp

        bp_places_combinations = []

        for net_place in net_places:
            place_list = []
            for bp_place in bp.net.places:
                if net_place.properties["id"] == bp_place.properties[
                        "origin_id"]:
                    place_list.append(bp_place)
            bp_places_combinations.append(place_list)

        return product(*bp_places_combinations)

    def get_corresponding_transitions_combinations(
            self,
            net_transitions: list[pm4py.PetriNet.Transition],
            bp: ExtendedPetriNet = None):
        if bp is None:
            bp = self.bp

        bp_transitions_combinations = []

        for net_transition in net_transitions:
            transition_list = []
            for bp_transition in bp.net.transitions:
                if net_transition.properties["id"] == bp_transition.properties[
                        "origin_id"]:
                    transition_list.append(bp_transition)
            bp_transitions_combinations.append(transition_list)

        return product(*bp_transitions_combinations)

    def build_downward_closure(self, transition):
        # Downward closure rules, aka can we fire from  the IM to the transition to be added
        # It starts from the given transition
        # It ends at the IM (and EXACTLY THE IM)
        # A transition can input from multiple places
        #   This means we have to follow both  branches now to continue our downward closure
        # Because a place can only input from one transition in the BP, we are not concerned with what happens when getting the place preset
        # This means we can just recursively get the preset starting at a transition until  we reach the IM
        # TODO Question: is it true that the preset of a place in a BP will only ever be one transition?'
        # TODO Question: is it true that any given transition in the BP will only have one configuration?
        downward_closure = {transition}

        # TODO check this?
        # We can directly get the preset places because in the extend_naive function we made sure to add all input places
        # Therefore we don't have to validate wheter we have all input places for a transition in the BP
        preset_places = self.bp.get_preset(transition)

        for preset_place in preset_places:
            downward_closure.add(preset_place)
            self._build_downward_closure_helper(preset_place, downward_closure)

        # TODO how to check if the downward closure ends EXACTLY at the IM? For now I just find the places with no input arcs and check if that creates the IM
        # TODO is it possible that the possible_marking will include some place of the IM twice? If so, the represented counter should be a dict or something to count occurances
        possible_marking = []
        for node in downward_closure:
            if type(node) == pm4py.PetriNet.Place:
                if len(node.in_arcs) == 0:
                    possible_marking.append(node)

        represented_counter = 0
        for i in self.underlying_net.im:
            for n in possible_marking:
                if i.properties["id"] == n.properties["origin_id"]:
                    represented_counter += 1

        if represented_counter != len(self.underlying_net.im):
            # It's not a downward closure! Return empty set?
            return {}

        return downward_closure

    def _build_downward_closure_helper(self, place, downward_closure):
        # Get corresponding bp transitions
        bpts = self.bp.get_preset(place)
        if len(bpts) == 0:
            print("No preset found aka basecase, returning")
            return
        if len(bpts) > 1:
            print("Bad place preset lenght in BP?")
        for bpt in bpts:
            downward_closure.add(bpt)
            pr = self.bp.get_preset(bpt)
            downward_closure.update(pr)
            for p in pr:
                self._build_downward_closure_helper(p, downward_closure)

    def has_local_config(
        self,
        transition: pm4py.PetriNet.Transition,
    ) -> bool:
        # Build downward closure, if we can't then there is no local config
        downward_closure = self.build_downward_closure(transition)

        if len(downward_closure) == 0:
            # No downward closures, no local config
            return False

        # Check conflict-freeness in the downward closure
        # This mean that two transitions may not input from the same place in the downward closure
        for node in downward_closure:
            if type(node) == pm4py.PetriNet.Place:
                if len(node.out_arcs) > 1:
                    counter = 0
                    # Check if more than one of the targets of the out_arc are present in the downward closure
                    for out_arc in node.out_arcs:
                        if out_arc.target in downward_closure:
                            counter += 1
                    # More than one transition connected to a place, aka they input from the same place, no local config
                    if counter > 1:
                        return False

        return True


def initialize_unfolding(net: ExtendedPetriNet) -> BranchingProcess:
    # Initialize an empty branching process with an underlying Petri net
    bp = BranchingProcess(ExtendedPetriNet(), net)

    for m in net.im:
        bp.add_initial_place(m)

    return bp


def get_possible_extensions(net: ExtendedPetriNet, bp: BranchingProcess):
    pass


def build_petri_net(
        filepath: str) -> tuple[pm4py.PetriNet, pm4py.Marking, pm4py.Marking]:
    df = pd.read_csv(filepath, sep=",")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    net, im, fm = pm4py.discover_petri_net_inductive(df,
                                                     activity_key="Activity",
                                                     case_id_key="CaseID",
                                                     timestamp_key="Timestamp")
    return net, im, fm


def main():
    net, im, fm = build_petri_net("testnet_no_cycles.csv")
    extended_net = ExtendedPetriNet(net, im, fm)
    pm4py.view_petri_net(net)
    # for p in extended_net.net.places:
    #     print(extended_net.get_preset(p))
    #     print(extended_net.get_postset(p))

    # for t in extended_net.net.transitions:
    #     print(extended_net.get_preset(t))
    #     print(extended_net.get_postset(t))

    bp = initialize_unfolding(extended_net)
    extendable = True
    while extendable:
        extendable = bp.extend_naive()
        pm4py.view_petri_net(bp.bp.net)

    pm4py.view_petri_net(bp.bp.net)
    bp.extend_naive()
    pm4py.view_petri_net(bp.bp.net)
    bp.extend_naive()


if __name__ == "__main__":
    main()
