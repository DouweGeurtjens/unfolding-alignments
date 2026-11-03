import pandas as pd
from pm4py import PetriNet, Marking, discover_petri_net_inductive, view_petri_net, write_pnml
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
    LOG_SILENT = 6


class IDGenerator:

    def __init__(self) -> None:
        self.counter = 0

    def generate_id(self) -> int:
        self.counter += 1
        return self.counter


IDGEN = IDGenerator()


class ExtendedSyncNet(PetriNet):

    def __init__(
        self,
        sync_net: PetriNet,
        sync_im: Marking,
        sync_fm: Marking,
        cost_function: dict,
    ):
        super().__init__(sync_net.name, sync_net.places, sync_net.transitions,
                         sync_net.arcs, sync_net.properties)
        self.im = sync_im
        self.fm = sync_fm
        self.cost_function = cost_function
        self._extend_net()
        self.id_to_node_mapping = self._build_id_to_node_mapping()

    def _extend_net(self):
        """Extend a Petri Net with explicit IDs on all places and transitions

        Args:
            net (PetriNet): The Petri Net to extend
        """
        for p in self.places:
            if NetProperties.ID.name not in p.properties:
                p.properties[NetProperties.ID.name] = IDGEN.generate_id()
        for t in self.transitions:
            if NetProperties.ID.name not in t.properties:
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


class ExtendedSyncNetStreaming(ExtendedSyncNet):

    def __init__(self, sync_net: PetriNet, sync_im: Marking, sync_fm: Marking,
                 trace_fm: Marking, cost_function: dict):
        super().__init__(sync_net, sync_im, sync_fm, cost_function)
        # To find the endpoint of the perfix alignment
        self.trace_fm = Marking()
        for p1 in trace_fm:
            for p2 in sync_net.places:
                if p2.name == (p1.name, ">>"):
                    self.trace_fm[p2] = 1


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


def construct_po_trace_net(trace, trace_name_key, activity_key):
    net = PetriNet('trace net of %s' %
                   trace.attributes[trace_name_key] if trace_name_key in
                   trace.attributes else ' ')
    p0 = PetriNet.Place('p_0')
    net.places.add(p0)

    # Group transitions by day, in order
    groups = {}
    for i in range(0, len(trace)):
        t = PetriNet.Transition('t_' + trace[i][activity_key] + '_' + str(i),
                                trace[i][activity_key])
        net.transitions.add(t)
        if trace[i]["Timestamp"].day in groups:
            groups[trace[i]["Timestamp"].day].append(t)
        else:
            groups[trace[i]["Timestamp"].day] = [t]
    keys = list(groups.keys())

    place_map = {0: p0}
    for i in range(0, len(keys)):
        k = keys[i]
        v = groups[k]
        if len(v) == 1:
            t = v[0]
            p = PetriNet.Place('p_' + str(i + 1))
            place_map[i + 1] = p
            net.places.add(p)
            # add_arc_from_to(place_map[i], t, net)
            add_arc_from_to(t, p, net)
            add_arc_from_to(place_map[i], t, net)
        else:
            # silent start
            sts = PetriNet.Transition("tau_log_start" + str(i))
            net.transitions.add(sts)
            add_arc_from_to(place_map[i], sts, net)

            #silent end
            ste = PetriNet.Transition("tau_log_end" + str(i))
            net.transitions.add(ste)
            pste = PetriNet.Place("p_ste" + str(i))
            place_map[i + 1] = pste
            net.places.add(pste)
            add_arc_from_to(ste, pste, net)

            for j in range(0, len(v)):
                t = v[j]

                pin = PetriNet.Place(f'p_in' + str(i + 1) + '_' + str(j + 1))
                net.places.add(pin)
                add_arc_from_to(sts, pin, net)
                add_arc_from_to(pin, t, net)

                pout = PetriNet.Place(f'p_out' + str(i + 1) + '_' + str(j + 1))
                net.places.add(pout)
                add_arc_from_to(t, pout, net)
                add_arc_from_to(pout, ste, net)

    return net, Marking({p0: 1}), Marking({place_map[len(keys)]: 1})


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


# trace_event_number is the index of the trace_event in the total trace
def update_new_synchronous_product_streaming(old_sync_net, new_sync_net):
    # Copy over IDs
    copied_transition_ids = set()
    for t1 in old_sync_net.transitions:
        for t2 in new_sync_net.transitions:
            if t1.name == t2.name:
                t2.properties[NetProperties.ID.name] = t1.properties[
                    NetProperties.ID.name]
                copied_transition_ids.add(t2.properties[NetProperties.ID.name])

    for p1 in old_sync_net.places:
        for p2 in new_sync_net.places:
            if p1.name == p2.name:
                p2.properties[NetProperties.ID.name] = p1.properties[
                    NetProperties.ID.name]


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
        if t.label is None:
            t.properties[
                NetProperties.MOVE_TYPE.name] = MoveTypes.LOG_SILENT.name
        else:
            t.properties[NetProperties.MOVE_TYPE.name] = MoveTypes.LOG.name

    for p in model_net.places:
        p.properties[NetProperties.MOVE_TYPE.name] = MoveTypes.MODEL.name

    for p in trace_net.places:
        p.properties[NetProperties.MOVE_TYPE.name] = MoveTypes.LOG.name

    t1_map, p1_map = copy_into(model_net, sync_net, False, ">>")
    t2_map, p2_map = copy_into(trace_net, sync_net, True, ">>")

    # Then, create the sync moves
    for t1 in model_net.transitions:
        for t2 in trace_net.transitions:
            if t1.label == t2.label and (
                    t1.properties[NetProperties.MOVE_TYPE.name]
                    != MoveTypes.MODEL_SILENT.name
                    or t2.properties[NetProperties.MOVE_TYPE.name]
                    != MoveTypes.LOG_SILENT.name):
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
        ("dummy_start_place", "dummy_start_place"),
        properties={NetProperties.MOVE_TYPE.name: MoveTypes.DUMMY.name})
    sync_net.places.add(dummy_start_place)

    dummy_start_transition = PetriNet.Transition(
        ("dummy_start_transition", "dummy start_transition"),
        ("START", "START"),
        properties={NetProperties.MOVE_TYPE.name: MoveTypes.DUMMY.name})
    sync_net.transitions.add(dummy_start_transition)

    add_arc_from_to(dummy_start_place, dummy_start_transition, sync_net)

    for p in combined_im:
        add_arc_from_to(dummy_start_transition, p, sync_net)

    dummy_end_transition = PetriNet.Transition(
        ("dummy_end_transition", "dummy_end_transition_1"), ("END", "END"),
        properties={NetProperties.MOVE_TYPE.name: MoveTypes.DUMMY.name})
    sync_net.transitions.add(dummy_end_transition)

    dummy_end_place = PetriNet.Place(
        ("dummy_end_place", "dummy_end_place_1"),
        properties={NetProperties.MOVE_TYPE.name: MoveTypes.DUMMY.name})
    sync_net.places.add(dummy_end_place)

    add_arc_from_to(dummy_end_transition, dummy_end_place, sync_net)

    for p in combined_fm:
        add_arc_from_to(p, dummy_end_transition, sync_net)

    sync_im = Marking()
    sync_im[dummy_start_place] = 1
    sync_fm = Marking()
    sync_fm[dummy_end_place] = 1

    # Make cost mapping
    cost_mapping = {}
    for t in sync_net.transitions:
        if t.properties[NetProperties.MOVE_TYPE.name] == MoveTypes.LOG.name:
            cost_mapping[t] = 10000
        if t.properties[NetProperties.MOVE_TYPE.name] == MoveTypes.MODEL.name:
            cost_mapping[t] = 10000
        if t.properties[NetProperties.MOVE_TYPE.name] == MoveTypes.DUMMY.name:
            cost_mapping[t] = 0
        if t.properties[NetProperties.MOVE_TYPE.name] == MoveTypes.SYNC.name:
            cost_mapping[t] = 0
        if t.properties[
                NetProperties.MOVE_TYPE.name] == MoveTypes.MODEL_SILENT.name:
            cost_mapping[t] = 1
        if t.properties[
                NetProperties.MOVE_TYPE.name] == MoveTypes.LOG_SILENT.name:
            cost_mapping[t] = 1

    return sync_net, sync_im, sync_fm, cost_mapping


def decorate_net(n: PetriNet, color: bool = True):
    # Will only work for nets with move type properties
    dec = {}
    for t in n.transitions:
        dec[t] = {}
        # Log skip
        if t.properties[NetProperties.MOVE_TYPE.name] == MoveTypes.LOG.name:
            dec[t]["label"] = t.label[0] + ", >>"
            if color:
                dec[t]["color"] = "lightgoldenrod"
        # Model skip
        elif t.properties[
                NetProperties.MOVE_TYPE.name] == MoveTypes.MODEL.name:
            dec[t]["label"] = ">>, " + t.label[1]
            if color:
                dec[t]["color"] = "lightblue"
        # Silent
        elif t.properties[
                NetProperties.MOVE_TYPE.
                name] == MoveTypes.MODEL_SILENT.name or t.properties[
                    NetProperties.MOVE_TYPE.name] == MoveTypes.LOG_SILENT.name:
            # Workaround to not need to modify underlying label
            dec[t]["label"] = " "
            if color:
                dec[t]["color"] = "darkgray"
        else:
            dec[t]["label"] = t.label[0]
            if color:
                dec[t]["color"] = "lightgreen"

    for p in n.places:
        dec[p] = {}
        # Log skip
        if p.properties[NetProperties.MOVE_TYPE.name] == MoveTypes.LOG.name:
            if color:
                dec[p]["color"] = "lightgoldenrod"
        # Model skip
        elif p.properties[
                NetProperties.MOVE_TYPE.name] == MoveTypes.MODEL.name:
            if color:
                dec[p]["color"] = "lightblue"
        else:
            if color:
                dec[p]["color"] = "lightgreen"

    return dec


def import_from_tpn(path):
    pn = PetriNet()
    im = Marking()
    fm = Marking()
    with open(path) as f:
        ls = f.readlines()
        for l in ls:
            sp = l.strip().strip(";").split(" ")
            if sp[0] == "place":
                name = sp[1].strip("\"")
                p = PetriNet.Place(name)
                pn.places.add(p)
                if "init" in sp:
                    im[p] = int(sp[3])
                if name == "end":
                    fm[p] = 1
            if sp[0] == "trans":
                sp_name_label = sp[1].split("~")
                name = sp_name_label[0].strip("\"")
                label = sp_name_label[1].split("+")[0].strip("\"")
                t = PetriNet.Transition(name, label)
                pn.transitions.add(t)

                in_index = sp.index("in")
                out_index = sp.index("out")

                preset = sp[in_index + 1:out_index]
                preset = [x.strip("\"") for x in preset]

                postset = sp[out_index + 1:]
                postset = [x.strip("\"") for x in postset]
                for p in pn.places:
                    if p.name in preset:
                        add_arc_from_to(p, t, pn)
                    if p.name in postset:
                        add_arc_from_to(t, p, pn)
    return pn, im, fm


if __name__ == "__main__":
    pass
