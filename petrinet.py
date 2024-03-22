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
            p.properties[NetProperties.ID.name] = IDGEN.generate_id()
        for t in self.transitions:
            t.properties[NetProperties.ID.name] = IDGEN.generate_id()

    def _build_id_to_node_mapping(self):
        ret = {}

        for p in self.places:
            ret[p.properties[NetProperties.ID.name]] = p
        for t in self.transitions:
            ret[t.properties[NetProperties.ID.name]] = t

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
        ret.add(arc.target.properties[NetProperties.ID.name])

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
        ret.add(arc.source.properties[NetProperties.ID.name])

    return ret


def construct_trace_net(trace, trace_name_key, activity_key):
    """
    Creates a trace net, i.e. a trace in Petri net form.

    Parameters
    ----------
    trace: :class:`list` input trace, assumed to be a list of events
    trace_name_key: :class:`str` key of the attribute that defines the name of the trace
    activity_key: :class:`str` key of the attribute of the events that defines the activity name

    Returns
    -------
    tuple: :class:`tuple` of the net, initial marking and the final marking

    """
    net = PetriNet('trace net of %s' %
                   trace.attributes[trace_name_key] if trace_name_key in
                   trace.attributes else ' ')
    place_map = {0: PetriNet.Place('p_0')}
    net.places.add(place_map[0])
    for i in range(0, len(trace)):
        t = PetriNet.Transition('t_' + trace[i][activity_key] + '_' + str(i),
                                trace[i][activity_key])
        net.transitions.add(t)
        place_map[i + 1] = PetriNet.Place('p_' + str(i + 1))
        net.places.add(place_map[i + 1])
        add_arc_from_to(place_map[i], t, net)
        add_arc_from_to(t, place_map[i + 1], net)
    return net, Marking({place_map[0]: 1}), Marking({place_map[len(trace)]: 1})


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


def construct_synchronous_product(model_net: PetriNet, model_net_im: Marking,
                                  model_net_fm: Marking, trace_net: PetriNet,
                                  trace_net_im: Marking,
                                  trace_net_fm: Marking):
    sync_net = PetriNet()
    # First, set some properties
    for t in model_net.transitions:
        if t.label is None:
            t.properties[
                NetProperties.MOVE_TYPE.name] = MoveTypes.MODEL_SILENT.name
        else:
            t.properties[NetProperties.MOVE_TYPE.name] = MoveTypes.MODEL.name
    for t in trace_net.transitions:
        t.properties[NetProperties.MOVE_TYPE.name] = MoveTypes.LOG.name

    t1_map, p1_map = copy_into(model_net, sync_net, False, ">>")
    t2_map, p2_map = copy_into(trace_net, sync_net, True, ">>")

    # Then, create the sync moves
    for t1 in model_net.transitions:
        for t2 in trace_net.transitions:
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
    for p in model_net_im:
        combined_im[p1_map[p]] = model_net_im[p]
    for p in trace_net_im:
        combined_im[p2_map[p]] = trace_net_im[p]
    for p in model_net_fm:
        combined_fm[p1_map[p]] = model_net_fm[p]
    for p in trace_net_fm:
        combined_fm[p2_map[p]] = trace_net_fm[p]

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
    pass
