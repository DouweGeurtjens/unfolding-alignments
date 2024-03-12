import pandas as pd
from pm4py import PetriNet, Marking, discover_petri_net_inductive, view_petri_net
from typing import TypeAlias, Callable
import pm4py
from enum import Enum

PlaceID: TypeAlias = int
TransitionID: TypeAlias = int


class NetProperties(Enum):
    ID = 1
    MOVE_TYPE = 2


class MoveTypes(Enum):
    SYNC = 1
    LOG = 2
    MODEL = 3
    MODEL_SILENT = 4
    DUMMY = 5


class IDGenerator:

    def __init__(self) -> None:
        self.counter = 0

    def generate_id(self) -> int:
        self.counter += 1
        return self.counter


IDGEN = IDGenerator()


class ExtendedNet(PetriNet):

    def __init__(
        self,
        net: PetriNet,
        im: Marking,
        fm: Marking,
    ):
        super().__init__(net.name, net.places, net.transitions, net.arcs,
                         net.properties)
        self.im = im
        self.fm = fm
        self._extend_net()
        self.id_to_node_mapping = self._build_id_to_node_mapping()

    def _extend_net(self):
        """Extend a Petri Net with explicit IDs on all places and transitions

        Args:
            net (PetriNet): The Petri Net to extend
        """
        for p in self.places:
            p.properties[NetProperties.ID] = IDGEN.generate_id()
        for t in self.transitions:
            t.properties[NetProperties.ID] = IDGEN.generate_id()

    def _build_id_to_node_mapping(self):
        ret = {}

        for p in self.places:
            ret[p.properties[NetProperties.ID]] = p
        for t in self.transitions:
            ret[t.properties[NetProperties.ID]] = t

        return ret

    def check_or_branch(self, enabled_transition_ids: set[TransitionID],
                        marking: set[PlaceID]) -> set[TransitionID]:
        ret: set[TransitionID] = set()
        # TODO rewrite this so it takes into account mutually exclusive PAIRS of transitions instead of one big set
        # A set of transitions is mutually excluse if: for each transition there is one common input place

        for place_id in marking:
            postset_ids = get_postset_ids(self.get_net_node_by_id(place_id))
            if len(postset_ids) > 1:
                # This place can cause multiple transitions to be enabled
                # We note down which transitions by checking the enabled_transtions and marking each one that is present in the postset of the place
                mutually_exclusive_transition_ids = enabled_transition_ids.intersection(
                    postset_ids)
                ret = ret.union(mutually_exclusive_transition_ids)

        return ret

    def get_net_node_by_id(
        self, node_id: PlaceID | TransitionID
    ) -> PetriNet.Place | PetriNet.Transition:
        return self.id_to_node_mapping[node_id]

    def is_transition_enabled(self, transition_id: TransitionID,
                              marking_ids: set[PlaceID]) -> bool:
        # Checks if a given transition is enabled
        preset_ids = get_preset_ids(self.get_net_node_by_id(transition_id))
        return preset_ids.issubset(marking_ids)

    def get_enabled_net_transition_ids(
            self, marking_ids: set[PlaceID]) -> set[TransitionID]:
        # Given a marking in the underlying net, get all enabled transitions
        # For each place in the marking, get the postset transitions
        # Then, for each transition, check if its preset places are ALL present in the given marking
        # Only those transitions are enabled
        ret = set()
        transition_ids: list[TransitionID] = []
        for place in marking_ids:
            transition_ids.extend(
                get_postset_ids(self.get_net_node_by_id(place)))

        for transition_id in transition_ids:
            enabled = self.is_transition_enabled(transition_id, marking_ids)
            if enabled:
                ret.add(transition_id)

        return ret


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


def construct_net(df: pd.DataFrame) -> tuple[PetriNet, Marking, Marking]:
    # TODO don't discover trace nets lawl
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    net, im, fm = discover_petri_net_inductive(df,
                                               activity_key="Activity",
                                               case_id_key="CaseID",
                                               timestamp_key="Timestamp")
    return net, im, fm


def copy_into(source_net, target_net, upper, skip):
    t_map = {}
    p_map = {}
    for t in source_net.transitions:
        name = (t.name, skip) if upper else (skip, t.name)
        label = (t.label, skip) if upper else (skip, t.label)
        t_map[t] = PetriNet.Transition(name, label, properties=t.properties)
        target_net.transitions.add(t_map[t])

    for p in source_net.places:
        name = (p.name, skip) if upper else (skip, p.name)
        p_map[p] = PetriNet.Place(name, properties=p.properties)
        target_net.places.add(p_map[p])

    for t in source_net.transitions:
        for a in t.in_arcs:
            add_arc_from_to(p_map[a.source], t_map[t], target_net)
        for a in t.out_arcs:
            add_arc_from_to(t_map[t], p_map[a.target], target_net)

    return t_map, p_map


def add_arc_from_to(fr, to, net: PetriNet, weight=1):
    arc = PetriNet.Arc(fr, to, weight)
    net.arcs.add(arc)
    fr.out_arcs.add(arc)
    to.in_arcs.add(arc)


def construct_synchronous_product(net_1: PetriNet, net_1_im: Marking,
                                  net_1_fm: Marking, net_2: PetriNet,
                                  net_2_im: Marking, net_2_fm: Marking):
    sync_net = PetriNet()
    # First, set some properties
    for t in net_1.transitions:
        if t.label is None:
            t.properties[
                NetProperties.MOVE_TYPE.name] = MoveTypes.MODEL_SILENT.name
        else:
            t.properties[NetProperties.MOVE_TYPE.name] = MoveTypes.MODEL.name
    for t in net_2.transitions:
        t.properties[NetProperties.MOVE_TYPE.name] = MoveTypes.LOG.name

    t1_map, p1_map = copy_into(net_1, sync_net, True, ">>")
    t2_map, p2_map = copy_into(net_2, sync_net, False, ">>")

    # Then, create the sync moves
    for t1 in net_1.transitions:
        for t2 in net_2.transitions:
            if t1.label == t2.label:
                sync = PetriNet.Transition((t1.name, t2.name),
                                           (t1.label, t2.label),
                                           properties={
                                               NetProperties.MOVE_TYPE.name:
                                               MoveTypes.SYNC.name
                                           })
                sync_net.transitions.add(sync)
                # # copy the properties of the transitions inside the transition of the sync net
                # for p1 in t1.properties:
                #     sync.properties[p1] = t1.properties[p1]
                # for p2 in t2.properties:
                #     sync.properties[p2] = t2.properties[p2]
                for a in t1.in_arcs:
                    add_arc_from_to(p1_map[a.source], sync, sync_net)
                for a in t2.in_arcs:
                    add_arc_from_to(p2_map[a.source], sync, sync_net)
                for a in t1.out_arcs:
                    add_arc_from_to(sync, p1_map[a.target], sync_net)
                for a in t2.out_arcs:
                    add_arc_from_to(sync, p2_map[a.target], sync_net)

    # Make IM and FM
    combined_im = Marking()
    combined_fm = Marking()
    for p in net_1_im:
        combined_im[p1_map[p]] = net_1_im[p]
    for p in net_2_im:
        combined_im[p2_map[p]] = net_2_im[p]
    for p in net_1_fm:
        combined_fm[p1_map[p]] = net_1_fm[p]
    for p in net_2_fm:
        combined_fm[p2_map[p]] = net_2_fm[p]

    # TODO make dummy start and end as IM/FM
    dummy_start_place = PetriNet.Place(
        ("dummy_start_place", "dummy_start_place"))
    sync_net.places.add(dummy_start_place)

    dummy_start_transition = PetriNet.Transition(
        ("dummy_start_transition", "dummy start_transition"),
        ("dummy_start_transition", "dummy start_transition"),
        properties={NetProperties.MOVE_TYPE.name: MoveTypes.DUMMY.name})
    sync_net.transitions.add(dummy_start_transition)

    add_arc_from_to(dummy_start_place, dummy_start_transition, sync_net)

    for p in combined_im:
        add_arc_from_to(dummy_start_transition, p, sync_net)

    dummy_end_transition = PetriNet.Transition(
        ("dummy_end_transition", "dummy_end_transition_1"),
        ("dummy_end_transition", "dummy_end_transition_1"),
        properties={NetProperties.MOVE_TYPE.name: MoveTypes.DUMMY.name})
    sync_net.transitions.add(dummy_end_transition)

    dummy_end_place = PetriNet.Place(("dummy_end_place", "dummy_end_place_1"))
    sync_net.places.add(dummy_end_place)

    add_arc_from_to(dummy_end_transition, dummy_end_place, sync_net)

    for p in combined_fm:
        add_arc_from_to(p, dummy_end_transition, sync_net)

    sync_im = Marking()
    sync_im[dummy_start_place] = 1
    sync_fm = Marking()
    sync_fm[dummy_end_place] = 1

    return sync_net, sync_im, sync_fm


if __name__ == "__main__":
    df = pd.read_csv("testnet_no_cycles.csv", sep=",")
    df_trace = df.loc[df["CaseID"] == "c1"]
    print(df)
    print(df_trace)
    model, model_im, model_fm = construct_net(df)
    trace_net, trace_net_im, trace_net_fm = construct_net(df_trace)
    sync_net, sync_im, sync_fm = construct_synchronous_product(
        model, model_im, model_fm, trace_net, trace_net_im, trace_net_fm)
    view_petri_net(sync_net, sync_im, sync_fm)
