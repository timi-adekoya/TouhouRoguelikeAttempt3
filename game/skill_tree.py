from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class TreeNodeDef:
    """One node in a skill tree. Nodes grant skills and/or apply upgrades to
    skills already in the book. Class trees may branch; species/innate trees
    are mostly linear (a node chain) — same structure, fewer branches."""

    node_id: str
    name: str
    description: str = ""
    requires_level: int = 1  # level of the tree's owner (class or species)
    prereqs: List[str] = field(default_factory=list)  # other node_ids in this tree
    grants_skills: List[str] = field(default_factory=list)
    grants_upgrades: List[str] = field(default_factory=list)
    # Skill points needed to claim this node (see the owning tree/skill's own
    # `points_per_level`). 0 (default) = free, same as every pre-existing
    # node — starter/base-kit nodes should stay 0. Deliberately per-node
    # rather than tied to any global tier concept, since different trees
    # (a shallow 2-branch class fork vs. a deep 4-branch one) need very
    # different cost scales.
    cost: int = 0
    # Mutually-exclusive alternatives (e.g. a Berserker vs. Sentinel fork):
    # claiming one permanently locks the other out for this character.
    # TODO(over-leveling): once a class can exceed its current max_level,
    # a character who's earned that much extra investment should be able
    # to go back and claim the path not taken too — this list is what an
    # "unlock the other branch" mechanic would need to check against, but
    # that unlock isn't built yet.
    excludes: List[str] = field(default_factory=list)


@dataclass
class SkillTreeDef:
    tree_id: str
    nodes: Dict[str, TreeNodeDef] = field(default_factory=dict)

    def available_nodes(self, level: int, claimed: set) -> List[TreeNodeDef]:
        """Nodes claimable right now: level met, prereqs claimed, not claimed,
        and no excluded alternative already taken."""
        return [
            node
            for node in self.nodes.values()
            if node.node_id not in claimed
            and node.requires_level <= level
            and all(p in claimed for p in node.prereqs)
            and not any(x in claimed for x in node.excludes)
        ]


def parse_tree(tree_id: str, raw_nodes: List[Dict[str, Any]]) -> SkillTreeDef:
    tree = SkillTreeDef(tree_id=tree_id)
    for raw in raw_nodes:
        node = TreeNodeDef(
            node_id=raw["node_id"],
            name=raw["name"],
            description=raw.get("description", ""),
            requires_level=raw.get("requires_level", 1),
            prereqs=raw.get("prereqs", []),
            grants_skills=raw.get("grants_skills", []),
            grants_upgrades=raw.get("grants_upgrades", []),
            excludes=raw.get("excludes", []),
            cost=raw.get("cost", 0),
        )
        tree.nodes[node.node_id] = node
    return tree
