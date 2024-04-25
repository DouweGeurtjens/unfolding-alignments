from pm4py import PetriNet, Marking, ProcessTree, parse_process_tree, view_process_tree, convert_to_petri_net, view_petri_net
from pm4py.objects.process_tree.semantics import generate_log
from petrinet import IDGEN
from enum import Enum
import pm4py
import random
from itertools import product


class Operators(Enum):
    SEQUENCE = "->"
    PARALLEL = "+"
    XOR = "X"
    CHOICE = "O"
    BINARY_LOOP = "*"


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


ACTIVITY_GENERATOR = ActivityGenerator()


class Model:

    def __init__(self, starting_block) -> None:
        self.starting_block: Block = starting_block

    def build(self) -> ProcessTree:
        pt_string = ""
        pt_string = self.starting_block.build(pt_string)
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

    def build(self, pt: str, operator: Operators) -> str:
        pt += operator.value + "("

        for i in range(self.breadth):
            # Always follow with a sequence to build the depth
            pt += Operators.SEQUENCE.value + "("

            # Build depth, each activity followed by a comma
            for _ in range(self.depth):
                pt += f"'{ACTIVITY_GENERATOR.generate_activity()}',"

            # If there are children, build them
            if i < len(self.children):
                pt = self.children[i].build(pt)

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

    def build(self, pt: str) -> str:
        return super().build(pt, Operators.PARALLEL)


class ExclusiveBlock(Block):

    def __init__(self,
                 breadth: int,
                 depth: int,
                 children,
                 parent=None) -> None:
        super().__init__(breadth, depth, children, parent)

    def build(self, pt: str):
        return super().build(pt, Operators.XOR)


class LoopBlock(Block):

    def __init__(self,
                 breadth: int,
                 depth: int,
                 children,
                 parent=None) -> None:
        super().__init__(breadth, depth, children, parent)

    def build(self, pt: str) -> str:
        return super().build(pt, Operators.BINARY_LOOP)


def main():
    l = LoopBlock(2, 3, [])
    e = ExclusiveBlock(2, 1, [l])
    c = ConcurrentBlock(4, 2, [e, e])
    m = Model(c)
    pt = m.build()
    # log = generate_log(pt, 10000)
    # for trace in log:
    #     print(trace)
    pn, im, fm = convert_to_petri_net(pt)
    view_petri_net(pn, im, fm)


if __name__ == "__main__":
    main()
