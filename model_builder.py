from pm4py import PetriNet, Marking, ProcessTree, parse_process_tree, view_process_tree, convert_to_petri_net, view_petri_net
from pm4py.objects.process_tree.semantics import generate_log
from petrinet import IDGEN
from enum import Enum
import pm4py
import random
from itertools import product
from settings import *
from copy import copy


class ActivityGenerator:

    def __init__(self) -> None:
        self.verbs = [
            "actualize",
            "administrate",
            "aggregate",
            "architect",
            "benchmark",
            "brand",
            "build",
            "cloudify",
            "communicate",
            "conceptualize",
            "coordinate",
            "create",
            "cultivate",
            "customize",
            "deliver",
            "deploy",
            "develop",
            "drive",
            "embrace",
            "empower",
            "enable",
            "engage",
            "engineer",
            "enhance",
            "envisioneer",
            "evisculate",
            "evolve",
            "expedite",
            "exploit",
            "extend",
            "fabricate",
            "facilitate",
            "fashion",
            "formulate",
            "foster",
            "generate",
            "grow",
            "harness",
            "impact",
            "implement",
            "incentivize",
            "incept",
            "incubate",
            "initiate",
            "innovate",
            "integrate",
            "iterate",
            "maintain",
            "matrix",
            "maximize",
            "mesh",
            "monetize",
            "morph",
            "myocardinate",
            "negotiate",
            "network",
            "optimize",
            "onboard",
            "orchestrate",
            "plagiarize",
            "pontificate",
            "predominate",
            "procrastinate",
            "productivate",
            "productize",
            "promote",
            "pursue",
            "recaptiualize",
            "reconceptualize",
            "redefine",
            "reintermediate",
            "reinvent",
            "repurpose",
            "restore",
            "revolutionize",
            "scale",
            "seize",
            "simplify",
            "strategize",
            "streamline",
            "supply",
            "syndicate",
            "synergize",
            "synthesize",
            "target",
            "transform",
            "transition",
            "underwhelm",
            "unleash",
            "utilize",
            "visualize",
            "whiteboard",
        ]

        self.adjectives = [
            "accurate",
            "adaptive",
            "agile",
            "alternative",
            "an expanded array of",
            "B2B",
            "B2C",
            "backend",
            "backward-compatible",
            "best-of-breed",
            "bleeding-edge",
            "bricks-and-clicks",
            "business",
            "clicks-and-mortar",
            "client-based",
            "client-centered",
            "client-centric",
            "client-focused",
            "cloud-based",
            "cloud-centric",
            "cloudified",
            "collaborative",
            "compelling",
            "competitive",
            "cooperative",
            "corporate",
            "cost effective",
            "covalent",
            "cross functional",
            "cross-media",
            "cross-platform",
            "cross-unit",
            "customer directed",
            "customized",
            "cutting-edge",
            "distinctive",
            "distributed",
            "diverse",
            "dynamic",
            "e-business",
            "economically sound",
            "effective",
            "efficient",
            "elastic",
            "emerging",
            "empowered",
            "enabled",
            "end-to-end",
            "enterprise",
            "enterprise-wide",
            "equity invested",
            "error-free",
            "ethical",
            "excellent",
            "exceptional",
            "extensible",
            "extensive",
            "flexible",
            "focused",
            "frictionless",
            "front-end",
            "fully researched",
            "fully tested",
            "functional",
            "functionalized",
            "fungible",
            "future-proof",
            "global",
            "go forward",
            "goal-oriented",
            "granular",
            "high standards in",
            "high-payoff",
            "hyperscale",
            "high-quality",
            "highly efficient",
            "holistic",
            "impactful",
            "inexpensive",
            "innovative",
            "installed base",
            "integrated",
            "interactive",
            "interdependent",
            "intermandated",
            "interoperable",
            "intuitive",
            "just in time",
            "leading-edge",
            "leveraged",
            "long-term high-impact",
            "low-risk high-yield",
            "magnetic",
            "maintainable",
            "market positioning",
            "market-driven",
            "mission-critical",
            "multidisciplinary",
            "multifunctional",
            "multimedia based",
            "next-generation",
            "on-demand",
            "one-to-one",
            "open-source",
            "optimal",
            "orthogonal",
            "out-of-the-box",
            "pandemic",
            "parallel",
            "performance based",
            "plug-and-play",
            "premier",
            "premium",
            "principle-centered",
            "proactive",
            "process-centric",
            "professional",
            "progressive",
            "prospective",
            "quality",
            "real-time",
            "reliable",
            "resource-sucking",
            "resource-maximizing",
            "resource-leveling",
            "revolutionary",
            "robust",
            "scalable",
            "seamless",
            "stand-alone",
            "standardized",
            "standards compliant",
            "state of the art",
            "sticky",
            "strategic",
            "superior",
            "sustainable",
            "synergistic",
            "tactical",
            "team building",
            "team driven",
            "technically sound",
            "timely",
            "top-line",
            "transparent",
            "turnkey",
            "ubiquitous",
            "unique",
            "user-centric",
            "user friendly",
            "value-added",
            "vertical",
            "viral",
            "virtual",
            "visionary",
            "web-enabled",
            "wireless",
            "world-class",
            "worldwide",
        ]

        self.nouns = [
            "action items",
            "adoption",
            "alignments",
            "applications",
            "architectures",
            "bandwidth",
            "benefits",
            "best practices",
            "catalysts for change",
            "channels",
            "clouds",
            "collaboration and idea-sharing",
            "communities",
            "content",
            "convergence",
            "core competencies",
            "customer service",
            "data",
            "deliverables",
            "e-business",
            "e-commerce",
            "e-markets",
            "e-tailers",
            "e-services",
            "experiences",
            "expertise",
            "functionalities",
            "fungibility",
            "growth strategies",
            "human capital",
            "ideas",
            "imperatives",
            "infomediaries",
            "information",
            "infrastructures",
            "initiatives",
            "innovation",
            "intellectual capital",
            "interfaces",
            "leadership",
            "leadership skills",
            "manufactured products",
            "markets",
            "materials",
            "meta-services",
            "methodologies",
            "methods of empowerment",
            "metrics",
            "mindshare",
            "models",
            "networks",
            "niches",
            "niche markets",
            "nosql",
            "opportunities",
            "outsourcing",
            "paradigms",
            "partnerships",
            "platforms",
            "portals",
            "potentialities",
            "rocess improvements",
            "processes",
            "products",
            "quality vectors",
            "relationships",
            "resources",
            "results",
            "ROI",
            "scenarios",
            "schemas",
            "scrums",
            "services",
            "solutions",
            "sources",
            "sprints",
            "strategic theme areas",
            "storage",
            "supply chains",
            "synergy",
            "systems",
            "technologies",
            "technology",
            "testing procedures",
            "total linkage",
            "users",
            "value",
            "vortals",
            "web-readiness",
            "web services",
            "wins",
            "virtualization",
        ]

        self.combinations = list(
            product(self.verbs, self.adjectives, self.nouns))
        self.rand = random.Random()
        self.multiplier = 0

    def generate_activity(self):
        if len(self.combinations) == 0:
            self.combinations = list(
                product(self.verbs, self.adjectives, self.nouns))
            self.multiplier += 1

        comb = self.rand.choice(self.combinations)
        self.combinations.remove(comb)

        activity = f"{comb[0]} {comb[1]} {comb[2]}"

        if self.multiplier > 0:
            activity += " repeat"

        return activity


class Model:

    def __init__(self, starting_block) -> None:
        self.starting_block: Block = starting_block

    def build(self, activity_generator) -> ProcessTree:
        pt_string = ""
        pt_string = self.starting_block.build(pt_string, activity_generator)
        pt = parse_process_tree(pt_string)

        return pt


class Block:

    def __init__(self,
                 breadth: int,
                 depth: int,
                 children,
                 parent=None) -> None:
        self.breadth: int = breadth
        self.depth: int = depth
        self.children: list[Block] = children
        self.parent: Block = parent

    def build(self, pt: str, operator,
              activity_generator: ActivityGenerator) -> str:
        pt += operator.value + "("

        for i in range(self.breadth):
            # Always follow with a sequence to build the depth
            pt += pm4py.objects.process_tree.obj.Operator.SEQUENCE.value + "("

            # Build depth, each activity followed by a comma
            for _ in range(self.depth):
                pt += f"'{activity_generator.generate_activity()}',"

            # If there are children, build them
            if i < len(self.children):
                pt = self.children[i].build(pt, activity_generator)

            # If there are no children, remove the trailing comma
            else:
                pt = pt.strip(",")

            # Add closing bracket
            pt += "),"

        # Remove trailing comma from last closing bracket
        pt = pt.strip(",")

        # Add final closing bracket
        pt += ")"

        return pt


class ConcurrentBlock(Block):

    def __init__(self,
                 breadth: int,
                 depth: int,
                 children,
                 parent=None) -> None:
        super().__init__(breadth, depth, children, parent)

    def build(self, pt: str, activity_generator) -> str:
        return super().build(pt,
                             pm4py.objects.process_tree.obj.Operator.PARALLEL,
                             activity_generator)


class ExclusiveBlock(Block):

    def __init__(self,
                 breadth: int,
                 depth: int,
                 children,
                 parent=None) -> None:
        super().__init__(breadth, depth, children, parent)

    def build(self, pt: str, activity_generator):
        return super().build(pt, pm4py.objects.process_tree.obj.Operator.XOR,
                             activity_generator)


class LoopBlock(Block):

    def __init__(self,
                 breadth: int,
                 depth: int,
                 children,
                 parent=None) -> None:
        super().__init__(breadth, depth, children, parent)

    def build(self, pt: str, activity_generator) -> str:
        return super().build(pt, pm4py.objects.process_tree.obj.Operator.LOOP,
                             activity_generator)


class SequenceBlock(Block):

    def __init__(self,
                 breadth: int,
                 depth: int,
                 children,
                 parent=None) -> None:
        super().__init__(breadth, depth, children, parent)

    def build(self, pt: str, activity_generator) -> str:
        return super().build(pt,
                             pm4py.objects.process_tree.obj.Operator.SEQUENCE,
                             activity_generator)


def main():
    # Basic concurrent and exclusive models
    for breadth in range(2, 16):
        for depth in range(1, 16):
            activity_generator_c = ActivityGenerator()
            c = ConcurrentBlock(breadth, depth, [])
            m = Model(c)
            pt = m.build(activity_generator_c)
            pm4py.write_ptml(pt, f"{CONCURRENT_MODEL_DIR}/b{breadth}_d{depth}")
            log = generate_log(pt, 50)
            pm4py.write_xes(log, f"{CONCURRENT_MODEL_DIR}/b{breadth}_d{depth}")

            activity_generator_e = ActivityGenerator()
            e = ExclusiveBlock(breadth, depth, [])
            me = Model(e)
            pt = me.build(activity_generator_e)
            pm4py.write_ptml(pt, f"{EXCLUSIVE_MODEL_DIR}/b{breadth}_d{depth}")
            log = generate_log(pt, 50)
            pm4py.write_xes(log, f"{EXCLUSIVE_MODEL_DIR}/b{breadth}_d{depth}")

    # Nested concurrency in concurrency
    for breadth in range(2, 3):
        for depth in range(1, 6):
            nesting_block_base = ConcurrentBlock(breadth, depth, [])
            for nest in range(1, 6):
                if nest == 1:
                    nesting_block = nesting_block_base
                else:
                    nesting_block = ConcurrentBlock(
                        breadth, depth, [nesting_block_base] * breadth)
                    nesting_block_base = nesting_block

                # Nested once
                activity_generator = ActivityGenerator()
                c = ConcurrentBlock(breadth, depth, [nesting_block] * breadth)
                m = Model(c)
                pt = m.build(activity_generator)
                pm4py.write_ptml(
                    pt,
                    f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/b{breadth}_d{depth}_n{nest}_bn{nesting_block.breadth}_dn{nesting_block.depth}"
                )
                log = generate_log(pt, 50)
                pm4py.write_xes(
                    log,
                    f"{CONCURRENT_CONCURRENT_NESTED_MODEL_DIR}/b{breadth}_d{depth}_n{nest}_bn{nesting_block.breadth}_dn{nesting_block.depth}"
                )
                # pm4py.view_petri_net(*pm4py.convert_to_petri_net(pt))

    # Nested exclusive in exclusive
    for breadth in range(2, 3):
        for depth in range(1, 6):
            nesting_block_base = ExclusiveBlock(breadth, depth, [])
            for nest in range(1, 6):
                if nest == 1:
                    nesting_block = nesting_block_base
                else:
                    nesting_block = ConcurrentBlock(
                        breadth, depth, [nesting_block_base] * breadth)
                    nesting_block_base = nesting_block

                # Nested once
                activity_generator = ActivityGenerator()
                c = ExclusiveBlock(breadth, depth, [nesting_block] * breadth)
                m = Model(c)
                pt = m.build(activity_generator)
                pm4py.write_ptml(
                    pt,
                    f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/b{breadth}_d{depth}_n{nest}_bn{nesting_block.breadth}_dn{nesting_block.depth}"
                )
                log = generate_log(pt, 50)
                pm4py.write_xes(
                    log,
                    f"{EXCLUSIVE_EXCLUSIVE_NESTED_MODEL_DIR}/b{breadth}_d{depth}_n{nest}_bn{nesting_block.breadth}_dn{nesting_block.depth}"
                )


if __name__ == "__main__":
    total_length = 20
    loop_counter = 5
    for loop_length in range(4, total_length):
        tail_length = total_length - loop_length - 1
        tail = SequenceBlock(1, tail_length, [])
        loop = LoopBlock(1, loop_length, [])
        start = SequenceBlock(2, 1, [loop, tail])
        activity_generator = ActivityGenerator()
        m = Model(start)
        pt = m.build(activity_generator)
        pm4py.write_ptml(
            pt, f"{LOOP_MODEL_DIR}/b1_d{total_length}_l{loop_length}")

        # Flatten PT
        q = []
        q.extend(pt.children)
        loop_part = None
        while loop_part is None:
            v = q.pop(-1)
            if v.operator and v.operator == pm4py.objects.process_tree.obj.Operator.LOOP:
                loop_part = v
            else:
                q.extend(v.children)
        loop_part.operator = pm4py.objects.process_tree.obj.Operator.SEQUENCE
        c = copy(loop_part.children[0].children)

        full_log = pm4py.objects.log.obj.EventLog()
        for i in range(loop_counter):
            if i > 0:
                loop_part.children[0].children.extend(c)
            view_petri_net(*convert_to_petri_net(pt))
            log = generate_log(pt, 10)
            for t in log:
                t.attributes[pm4py.util.xes_constants.DEFAULT_NAME_KEY] = str(
                    int(t.attributes[
                        pm4py.util.xes_constants.DEFAULT_NAME_KEY]) + 10 * i)
                full_log.append(t)
        pm4py.write_xes(full_log,
                        f"{LOOP_MODEL_DIR}/b1_d{total_length}_l{loop_length}")
    # main()
