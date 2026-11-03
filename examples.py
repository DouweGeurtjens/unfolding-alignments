from pm4py import PetriNet, Marking, view_petri_net, save_vis_petri_net
from petrinet import add_arc_from_to, IDGEN
from coset_standard import *
from coset_streaming import *
from settings import *

makebk_label = "MakeBk"
useprd_label = "UsePrD"
awaitc_label = "AwaitC"
sign_label = "Sign"
retractd_label = "RetractD"
submitpd_label = "SubmitPD"
submitpoe_label = "SubmitPoE"
filetypes = ["png", "svg"]


def create_example_unbounded_net():
    n = PetriNet()
    p0 = PetriNet.Place('p_0')
    n.places.add(p0)

    make_booking = PetriNet.Transition('t_1', makebk_label)
    n.transitions.add(make_booking)

    add_arc_from_to(p0, make_booking, n)
    add_arc_from_to(make_booking, p0, n)

    p1 = PetriNet.Place('p_1')
    n.places.add(p1)

    add_arc_from_to(make_booking, p1, n)

    a2 = PetriNet.Transition('t_2', submitpd_label)
    n.transitions.add(a2)

    add_arc_from_to(p1, a2, n)

    p2 = PetriNet.Place('p_2')
    n.places.add(p2)

    add_arc_from_to(a2, p2, n)

    im = Marking()
    im[p0] = 1
    fm = Marking()
    fm[p2] = 1
    fm[p0] = 1

    return n, im, fm


def example_unfolding():
    n = PetriNet()
    pi = PetriNet.Place(IDGEN.generate_id())
    n.places.add(pi)

    make_booking = PetriNet.Transition(IDGEN.generate_id(), makebk_label)
    n.transitions.add(make_booking)

    add_arc_from_to(pi, make_booking, n)

    p0 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p0)

    add_arc_from_to(make_booking, p0, n)

    # Branch use previous
    use_previous_data = PetriNet.Transition(IDGEN.generate_id(), useprd_label)
    n.transitions.add(use_previous_data)
    add_arc_from_to(p0, use_previous_data, n)

    p5 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p5)
    add_arc_from_to(use_previous_data, p5, n)

    await_contract = PetriNet.Transition(IDGEN.generate_id(), awaitc_label)
    n.transitions.add(await_contract)
    add_arc_from_to(p5, await_contract, n)

    p6 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p6)
    add_arc_from_to(await_contract, p6, n)

    countersign = PetriNet.Transition(IDGEN.generate_id(), sign_label)
    n.transitions.add(countersign)
    add_arc_from_to(p6, countersign, n)

    pf = PetriNet.Place(IDGEN.generate_id())
    n.places.add(pf)
    add_arc_from_to(countersign, pf, n)

    # Branch use previous repeat
    retract_data = PetriNet.Transition(IDGEN.generate_id(), retractd_label)
    n.transitions.add(retract_data)
    add_arc_from_to(p5, retract_data, n)

    p0_1 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p0_1)
    add_arc_from_to(retract_data, p0_1, n)

    # Branch use previous repeat use previous
    use_previous_data_1 = PetriNet.Transition(IDGEN.generate_id(),
                                              useprd_label)
    n.transitions.add(use_previous_data_1)
    add_arc_from_to(p0_1, use_previous_data_1, n)

    p5_2 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p5_2)
    add_arc_from_to(use_previous_data_1, p5_2, n)

    # Branch use previous repeat concurrent
    tau_3 = PetriNet.Transition(IDGEN.generate_id())
    n.transitions.add(tau_3)
    add_arc_from_to(p0_1, tau_3, n)

    p1_1 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p1_1)
    add_arc_from_to(tau_3, p1_1, n)

    p2_1 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p2_1)
    add_arc_from_to(tau_3, p2_1, n)

    # Branch concurrent
    tau_1 = PetriNet.Transition(IDGEN.generate_id())
    n.transitions.add(tau_1)
    add_arc_from_to(p0, tau_1, n)

    p1 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p1)
    add_arc_from_to(tau_1, p1, n)

    p2 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p2)
    add_arc_from_to(tau_1, p2, n)

    submit_payment_details = PetriNet.Transition(IDGEN.generate_id(),
                                                 submitpd_label)
    n.transitions.add(submit_payment_details)
    add_arc_from_to(p2, submit_payment_details, n)

    submit_proof_of_enrollment = PetriNet.Transition(IDGEN.generate_id(),
                                                     submitpoe_label)
    n.transitions.add(submit_proof_of_enrollment)
    add_arc_from_to(p1, submit_proof_of_enrollment, n)

    p3 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p3)
    add_arc_from_to(submit_payment_details, p3, n)

    p4 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p4)
    add_arc_from_to(submit_proof_of_enrollment, p4, n)

    tau_2 = PetriNet.Transition(IDGEN.generate_id())
    n.transitions.add(tau_2)
    add_arc_from_to(p3, tau_2, n)
    add_arc_from_to(p4, tau_2, n)

    p5_1 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p5_1)
    add_arc_from_to(tau_2, p5_1, n)

    await_contract_1 = PetriNet.Transition(IDGEN.generate_id(), awaitc_label)
    n.transitions.add(await_contract_1)
    add_arc_from_to(p5_1, await_contract_1, n)

    p6_1 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p6_1)
    add_arc_from_to(await_contract_1, p6_1, n)

    countersign_1 = PetriNet.Transition(IDGEN.generate_id(), sign_label)
    n.transitions.add(countersign_1)
    add_arc_from_to(p6_1, countersign_1, n)

    pf_1 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(pf_1)
    add_arc_from_to(countersign_1, pf_1, n)

    # Branch concurrent repeat
    retract_data_1 = PetriNet.Transition(IDGEN.generate_id(), retractd_label)
    n.transitions.add(retract_data_1)
    add_arc_from_to(p5_1, retract_data_1, n)

    p0_2 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p0_2)
    add_arc_from_to(retract_data_1, p0_2, n)

    # Branch concurrent repeat use previous
    use_previous_data_2 = PetriNet.Transition(IDGEN.generate_id(),
                                              useprd_label)
    n.transitions.add(use_previous_data_2)
    add_arc_from_to(p0_2, use_previous_data_2, n)

    p5_3 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p5_3)
    add_arc_from_to(use_previous_data_2, p5_3, n)

    # Branch concurrent repeat concurrent
    tau_4 = PetriNet.Transition(IDGEN.generate_id())
    n.transitions.add(tau_4)
    add_arc_from_to(p0_2, tau_4, n)

    p1_2 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p1_2)
    add_arc_from_to(tau_4, p1_2, n)

    p2_2 = PetriNet.Place(IDGEN.generate_id())
    n.places.add(p2_2)
    add_arc_from_to(tau_4, p2_2, n)

    dec_conf = {
        make_booking: {
            "color": "lightgreen"
        },
        use_previous_data: {
            "color": "lightcoral"
        },
        submit_proof_of_enrollment: {
            "color": "lightgreen"
        },
        p4: {
            "color": "lightblue"
        },
        submit_payment_details: {
            "color": "lightgreen"
        },
        p3: {
            "color": "lightblue"
        },
        tau_1: {
            "color": "lightgreen"
        },
    }

    dec_local_conf = {
        make_booking: {
            "color": "lightgreen"
        },
        submit_proof_of_enrollment: {
            "color": "lightgreen"
        },
        p4: {
            "color": "lightblue"
        },
        tau_1: {
            "color": "lightgreen"
        },
        p2: {
            "color": "lightblue"
        },
    }

    view_petri_net(n)
    view_petri_net(n, decorations=dec_conf)
    view_petri_net(n, decorations=dec_local_conf)
    for s in filetypes:
        save_vis_petri_net(n,
                           Marking(),
                           Marking(),
                           file_path=f"figures/examples/unf.{s}")

        save_vis_petri_net(n,
                           Marking(),
                           Marking(),
                           decorations=dec_conf,
                           file_path=f"figures/examples/unfconf.{s}")

        save_vis_petri_net(n,
                           Marking(),
                           Marking(),
                           decorations=dec_local_conf,
                           file_path=f"figures/examples/unfloconf.{s}")


def example_unbounded():
    mn, mi, mf = create_example_unbounded_net()
    view_petri_net(mn, mi, mf)
    for s in filetypes:
        save_vis_petri_net(mn,
                           mi,
                           mf,
                           file_path=f"figures/examples/unbounded_net.{s}")

    df = pd.read_csv("data/testnet_unbounded.csv", sep=",")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = format_dataframe(df,
                          activity_key="Activity",
                          case_id="CaseID",
                          timestamp_key="Timestamp")

    el = convert_to_event_log(df)

    tn, ti, tf = construct_trace_net(el[0], "Activity", "Activity")
    sp, spi, spf, cf = construct_synchronous_product(mn, mi, mf, tn, ti, tf)
    dec_sp = decorate_net(sp)
    view_petri_net(sp, spi, spf)
    for s in filetypes:
        save_vis_petri_net(sp,
                           spi,
                           spf,
                           decorations=dec_sp,
                           file_path=f"figures/examples/unbounded_sp.{s}")
    sync_net_extended = ExtendedSyncNet(sp, spi, spf, cf)

    bp = BranchingProcessStandard(sync_net_extended)

    bp.initialise_from_initial_marking()
    alignment, cost = bp.astar()
    conf = bp.get_full_configuration_from_marking(alignment)
    n = bp.convert_nodes_to_net(conf.nodes)
    dec = decorate_net(n)
    view_petri_net(n, decorations=dec)
    for s in filetypes:
        save_vis_petri_net(
            n,
            Marking(),
            Marking(),
            decorations=dec,
            file_path=f"figures/examples/unbounded_alignment.{s}")

    nodes = set()
    for _, v in bp.conditions.items():
        nodes.update(v)
    for _, v in bp.events.items():
        nodes.update(v)
    n_unf = bp.convert_nodes_to_net(nodes)
    dec_unf = decorate_net(n_unf)

    view_petri_net(n_unf, decorations=dec_unf)
    for s in filetypes:
        save_vis_petri_net(n_unf,
                           Marking(),
                           Marking(),
                           decorations=dec_unf,
                           file_path=f"figures/examples/unbound_unf.{s}")


def example_po(deviated: str = ""):
    df = pd.read_csv("data/testnet_po.csv", sep=",")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = format_dataframe(df,
                          activity_key="Activity",
                          case_id="CaseID",
                          timestamp_key="Timestamp")
    mn, mi, mf = pm4py.discover_petri_net_inductive(df,
                                                    activity_key="Activity",
                                                    case_id_key="CaseID",
                                                    timestamp_key="Timestamp")
    view_petri_net(mn, mi, mf)
    el = convert_to_event_log(df)
    trace = el[0]
    if deviated:
        rm = None
        for e in trace:
            if e["Activity"] == submitpoe_label:
                rm = e
        trace._list.remove(rm)
    tn, ti, tf = construct_po_trace_net(trace, "Activity", "Activity")

    view_petri_net(tn, ti, tf)
    for s in filetypes:
        save_vis_petri_net(tn,
                           ti,
                           tf,
                           file_path=f"figures/examples/po_en{deviated}.{s}")

    sp, spi, spf, cf = construct_synchronous_product(mn, mi, mf, tn, ti, tf)

    dec_sp = decorate_net(sp)
    view_petri_net(sp, spi, spf, decorations=dec_sp)
    for s in filetypes:
        save_vis_petri_net(sp,
                           spi,
                           spf,
                           decorations=dec_sp,
                           file_path=f"figures/examples/po_sp{deviated}.{s}")
    sync_net_extended = ExtendedSyncNet(sp, spi, spf, cf)

    bp = BranchingProcessStandard(sync_net_extended)

    bp.initialise_from_initial_marking()
    alignment, cost = bp.astar()
    conf = bp.get_full_configuration_from_marking(alignment)
    n = bp.convert_nodes_to_net(conf.nodes)
    dec = decorate_net(n)
    view_petri_net(n, decorations=dec)
    for s in filetypes:
        save_vis_petri_net(
            n,
            Marking(),
            Marking(),
            decorations=dec,
            file_path=f"figures/examples/po_alignment{deviated}.{s}")


def example(undeviated: str = ""):
    df = pd.read_csv("data/testnet_cycles.csv", sep=",")
    df["Timestamp"] = pd.to_datetime(df["Timestamp"])
    df = format_dataframe(df,
                          activity_key="Activity",
                          case_id="CaseID",
                          timestamp_key="Timestamp")
    mn, mi, mf = pm4py.discover_petri_net_inductive(df,
                                                    activity_key="Activity",
                                                    case_id_key="CaseID",
                                                    timestamp_key="Timestamp")
    for s in filetypes:
        save_vis_petri_net(mn,
                           mi,
                           mf,
                           file_path=f"figures/examples/model{undeviated}.{s}")

    el = convert_to_event_log(df)
    trace = el[0]
    if not undeviated:
        rm = None
        for e in trace:
            if e["Activity"] == submitpoe_label:
                rm = e
        trace._list.remove(rm)
    tn, ti, tf = construct_trace_net(el[0], "Activity", "Activity")
    view_petri_net(tn, ti, tf)
    for s in filetypes:
        save_vis_petri_net(tn,
                           ti,
                           tf,
                           file_path=f"figures/examples/en{undeviated}.{s}")

    sp, spi, spf, cf = construct_synchronous_product(mn, mi, mf, tn, ti, tf)

    dec_sp = decorate_net(sp)
    view_petri_net(sp, spi, spf, decorations=dec_sp)
    for s in filetypes:
        save_vis_petri_net(sp,
                           spi,
                           spf,
                           decorations=dec_sp,
                           file_path=f"figures/examples/sp{undeviated}.{s}")
    sync_net_extended = ExtendedSyncNet(sp, spi, spf, cf)

    bp = BranchingProcessStandard(sync_net_extended)

    bp.initialise_from_initial_marking()
    alignment, cost = bp.astar()
    conf = bp.get_full_configuration_from_marking(alignment)
    alignment_net = bp.convert_nodes_to_net(conf.nodes)
    dec_alignment = decorate_net(alignment_net)
    for s in filetypes:
        save_vis_petri_net(
            alignment_net,
            Marking(),
            Marking(),
            decorations=dec_alignment,
            file_path=f"figures/examples/alignment{undeviated}.{s}")
    nodes = set()
    for _, v in bp.conditions.items():
        nodes.update(v)
    for _, v in bp.events.items():
        nodes.update(v)
    n = bp.convert_nodes_to_net(nodes)

    dec = decorate_net(n, color=False)

    for t in n.transitions:
        for t2 in alignment_net.transitions:
            if t.name == t2.name:
                dec[t]["color"] = "lightgreen"
    for p in n.places:
        for p2 in alignment_net.places:
            if p.name == p2.name:
                dec[p]["color"] = "lightgreen"

    view_petri_net(n, decorations=dec)
    for s in filetypes:
        save_vis_petri_net(
            n,
            Marking(),
            Marking(),
            decorations=dec,
            file_path=f"figures/examples/bplabeled{undeviated}.{s}")


if __name__ == "__main__":
    # # example_unfolding()
    # example("_undeviated")
    # example()
    # example_unbounded()
    example_po("_deviated")
