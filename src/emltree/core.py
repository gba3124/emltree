"""EML tree data structure.

Grammar:  S -> 1 | x | eml(S, S)

Free variables are allowed as leaves for expressions with inputs. Pure
closed-form EML (as in the paper) uses only `1` and `eml`.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator


class EMLNode:
    """Base class for EML tree nodes."""

    def to_nested(self) -> str:
        raise NotImplementedError

    def to_rpn(self) -> str:
        """Reverse Polish notation; EML is encoded as 'E'."""
        raise NotImplementedError

    def leaf_count(self) -> int:
        raise NotImplementedError

    def node_count(self) -> int:
        raise NotImplementedError

    def depth(self) -> int:
        raise NotImplementedError

    def walk(self) -> Iterator["EMLNode"]:
        raise NotImplementedError


@dataclass(frozen=True)
class One(EMLNode):
    def to_nested(self) -> str:
        return "1"

    def to_rpn(self) -> str:
        return "1"

    def leaf_count(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1

    def depth(self) -> int:
        return 0

    def walk(self) -> Iterator[EMLNode]:
        yield self


@dataclass(frozen=True)
class Var(EMLNode):
    name: str

    def to_nested(self) -> str:
        return self.name

    def to_rpn(self) -> str:
        return self.name

    def leaf_count(self) -> int:
        return 1

    def node_count(self) -> int:
        return 1

    def depth(self) -> int:
        return 0

    def walk(self) -> Iterator[EMLNode]:
        yield self


@dataclass(frozen=True)
class Eml(EMLNode):
    left: EMLNode
    right: EMLNode

    def to_nested(self) -> str:
        return f"eml({self.left.to_nested()}, {self.right.to_nested()})"

    def to_rpn(self) -> str:
        return f"{self.left.to_rpn()} {self.right.to_rpn()} E"

    def leaf_count(self) -> int:
        return self.left.leaf_count() + self.right.leaf_count()

    def node_count(self) -> int:
        return 1 + self.left.node_count() + self.right.node_count()

    def depth(self) -> int:
        return 1 + max(self.left.depth(), self.right.depth())

    def walk(self) -> Iterator[EMLNode]:
        yield self
        yield from self.left.walk()
        yield from self.right.walk()


_ONE_SINGLETON = One()


def one() -> EMLNode:
    return _ONE_SINGLETON


def var(name: str) -> EMLNode:
    return Var(name)


def eml(x: EMLNode, y: EMLNode) -> EMLNode:
    return Eml(x, y)


def ascii_tree(node: EMLNode, prefix: str = "", is_last: bool = True) -> str:
    """Render an EML tree as ASCII. Left child drawn above right child."""
    connector = "└── " if is_last else "├── "
    if isinstance(node, One):
        return f"{prefix}{connector}1\n"
    if isinstance(node, Var):
        return f"{prefix}{connector}{node.name}\n"
    assert isinstance(node, Eml)
    line = f"{prefix}{connector}eml\n"
    child_prefix = prefix + ("    " if is_last else "│   ")
    line += ascii_tree(node.left, child_prefix, is_last=False)
    line += ascii_tree(node.right, child_prefix, is_last=True)
    return line
