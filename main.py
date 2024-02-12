import pandas as pd
import pm4py


class ExtendedPetriNet():
    # The general idea here is to make the PetriNet more suitable for unfolding
    def __init__(self,
                 net: pm4py.PetriNet = pm4py.PetriNet(),
                 im: pm4py.Marking = None,
                 fm: pm4py.Marking = None):
        self.net = net
        self.im = im
        self.fm = fm

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
        self.net = net
        # Keep a list of the ending nodes (no postset in the branching process), because we can only extend from those
        self.ending_nodes = []

    def view(self):
        pm4py.view_petri_net(self.bp.net)

    def add_initial_place(self, node: pm4py.PetriNet.Place):
        """Add an intial node to the branching process

        Intial nodes, unlike other nodes, do not add any arcs, even if they have a preset

        Args:
            node (pm4py.PetriNet.Place | pm4py.PetriNet.Transition): The input node coming from the original net
        """
        new_properties = node.properties
        new_properties["origin_node"] = node

        # TODO The name of the new node should be unique, now it is not
        new_node = pm4py.PetriNet.Place(node.name, properties=new_properties)

        self.bp.net.places.add(new_node)

        # The added initial nodes will always be ending nodes, so we can blindly add them
        self.ending_nodes.append(new_node)

    def extend(self):
        # Extensions always happen from a ending node right?
        # Ending nodes should always be places too?
        new_ending_nodes = []
        for en in self.ending_nodes:
            # Find corresponding node in the original net
            for n in self.net.net.places:
                if en.properties["origin_node"] == n:
                    node_in_net = n
                    break
            # node_in_net = self.net.net.places.get(en.properties["origin_node"])

            # Find a transition of which the corresponding node is in the preset
            # Is this the same as finding the postset of the corresponding node? I think so
            postset = self.net.get_postset(node_in_net)

            # If our postset is empty we can skip this iteration
            if len(postset) == 0:
                continue

            # If we find the transition, we need to gets its preset and connect those places
            # If there is more than one transition in the postset we need to consider all those transitions
            for t in postset:
                preset = self.net.get_preset(t)
                # Get the equivalent places in the branching process
                # TODO check if we can optimize this
                bp_places = []
                for place in preset:
                    for bp_place in self.bp.net.places:
                        if bp_place.properties["origin_node"] == place:
                            bp_places.append(bp_place)

                # add an and connect the transition

                added_t = self.add_node(bp_places, t)
                # Check if the added transition has a local configuration
                # If not, revert the addition

                # Add the postset of the transition to the branching process if we added the transition
                postset_t = self.net.get_postset(t)

                # TODO will this work if a place has two input transitions? Then we'd end up adding duplicate places?
                for place_2 in postset_t:
                    added_place = self.add_node([added_t], place_2)
                    # Add the added places to the new_ending nodes
                    new_ending_nodes.append(added_place)

                # Remove any nodes from self.ending_nodes that are now no longer ending nodes
                # aka the nodes in the preset of the transition we just added
                # this will update the loop, that's okay (in fact we have to, otherwise we will end up with duplicate transitions etc)
                for place_3 in bp_places:
                    try:
                        self.ending_nodes.remove(place_3)
                    except:
                        print("failed to rm")

        # Update the ending nodes
        self.ending_nodes = new_ending_nodes

    def add_node(
        self, preset: list[pm4py.PetriNet.Place | pm4py.PetriNet.Transition],
        target_node: pm4py.PetriNet.Place
        | pm4py.PetriNet.Transition
    ) -> pm4py.PetriNet.Place | pm4py.PetriNet.Transition:
        """Add a target node to the branching process

        When adding a node to the branching process we want to store the name it had in the original net
        To do this, we store the input node in the properties of the new node

        Args:
            source_node (pm4py.PetriNet.Place | pm4py.PetriNet.Transition): The node we try to extend from the branching process
            target_node (pm4py.PetriNet.Place | pm4py.PetriNet.Transition): The node to add coming from the original net
        """
        new_properties = target_node.properties
        new_properties["origin_node"] = target_node

        # TODO The name of the new node should be unique, now it is not
        if type(target_node) == pm4py.PetriNet.Place:
            new_target_node = pm4py.PetriNet.Place(target_node.name,
                                                   properties=new_properties)
        if type(target_node) == pm4py.PetriNet.Transition:
            new_target_node = pm4py.PetriNet.Transition(
                target_node.name, properties=new_properties)

        # Create a new_arcs
        # TODO put a weight on this based on whether its a synchronous move or not
        # TODO put a weight on this based on whether or not the object synchronise
        for preset_node in preset:
            new_arc = pm4py.PetriNet.Arc(preset_node, new_target_node)

            # Add an arc to the source_node out_arcs
            preset_node.out_arcs.add(new_arc)
            # Add an arc to the new_target_node in_arcs
            new_target_node.in_arcs.add(new_arc)
            # Add the arc to the branching process arcs
            self.bp.net.arcs.add(new_arc)

        # Add to either transitions or places depending on the type of the new_target_node
        if type(target_node) == pm4py.PetriNet.Place:
            self.bp.net.places.add(new_target_node)
        if type(target_node) == pm4py.PetriNet.Transition:
            self.bp.net.transitions.add(new_target_node)

        # # We just extended the source_node, so it's no longer an ending node
        # self.ending_nodes.remove(source_node)

        # # The node we just added is a new ending node
        # self.ending_nodes.add(new_target_node)

        return new_target_node


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
    while len(bp.ending_nodes) != 0:
        bp.extend()
        bp.view()


if __name__ == "__main__":
    main()
