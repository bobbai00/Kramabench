#!/usr/bin/env python3
"""
Application History Graph - Tracks variable snapshots across agent actions.

Definitions:
- Variable: A named entity (just a string).
- VariableSnapshot: A pair (variable_name, action_id) representing a variable at a point in time.
- Action: A single execution step identified by an ID.

Graph Structure (NetworkX DiGraph):
- Nodes: Action nodes and VariableSnapshot nodes
- Edges:
  - Action -> VariableSnapshot: Action creates the snapshot (only for successful actions)
  - VariableSnapshot -> Action: Snapshot is referenced by the action

Error Handling:
- Failed actions do NOT create variable snapshots
- Failed actions still reference existing variable snapshots
"""

import ast
import json
import sys
from typing import Dict, Set, List, Tuple, Any, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import networkx as nx


# =============================================================================
# Core Data Types
# =============================================================================

@dataclass(frozen=True)
class Action:
    """An execution step identified by an ID."""
    id: int
    success: bool = True
    error_msg: Optional[str] = None

    def __str__(self) -> str:
        status = "ok" if self.success else "err"
        return f"Action({self.id},{status})"

    def label(self, show_error: bool = True) -> str:
        """Label for visualization."""
        if show_error and not self.success:
            return f"Step {self.id} ❌"
        return f"Step {self.id}"


@dataclass(frozen=True)
class VariableSnapshot:
    """A variable at a specific point in time: (variable_name, action_id)."""
    variable: str
    created_at: int

    def __str__(self) -> str:
        return f"{self.created_at}-{self.variable}"

    @classmethod
    def from_string(cls, s: str) -> 'VariableSnapshot':
        parts = s.split("-", 1)
        return cls(variable=parts[1], created_at=int(parts[0]))


# =============================================================================
# Application History Graph
# =============================================================================

class ApplicationHistoryGraph:
    """
    A directed graph tracking actions and variable snapshots.

    Nodes: Action, VariableSnapshot
    Edges:
      - Action -> VariableSnapshot (creates)
      - VariableSnapshot -> Action (referenced_by)
    """

    def __init__(self):
        self.graph = nx.DiGraph()
        self._active_snapshot: Dict[str, VariableSnapshot] = {}  # var_name -> current snapshot

    def add_action(self, action: Action, defined: Set[str], referenced: Set[str],
                   visible_vars: Set[str] = None, invalid_vars: Set[str] = None) -> None:
        """Add an action and its variable definitions/references to the graph.

        Args:
            action: The action to add
            defined: Variables defined by this action (before error if failed)
            referenced: Variables referenced by this action
            visible_vars: Variables that are printed/displayed (shown with eye icon)
            invalid_vars: Variables defined AFTER an error line (shown in red)
        """
        visible_vars = visible_vars or set()
        invalid_vars = invalid_vars or set()

        self.graph.add_node(action, node_type='action', error_msg=action.error_msg)

        # FIRST: Check read dependencies BEFORE updating active snapshots
        # This ensures we capture cross-action reads even when a variable is redefined
        for var in referenced:
            if var in self._active_snapshot:
                snapshot = self._active_snapshot[var]
                # Only track cross-action reads (reader != writer)
                if snapshot.created_at != action.id:
                    self.graph.add_edge(snapshot, action, edge_type='read_dependency')

        # THEN: Create snapshots for all defined variables (including partial success)
        all_defined = defined | invalid_vars
        for var in all_defined:
            snapshot = VariableSnapshot(variable=var, created_at=action.id)
            is_visible = var in visible_vars
            is_valid = var not in invalid_vars
            self.graph.add_node(snapshot, node_type='snapshot',
                              visible=is_visible, valid=is_valid)
            self.graph.add_edge(action, snapshot, edge_type='write_dependency')
            # Only update active snapshot for valid variables
            if is_valid:
                self._active_snapshot[var] = snapshot

    # === Query Methods ===

    def actions(self) -> Iterator[Action]:
        """Iterate over all actions."""
        for node in self.graph.nodes:
            if isinstance(node, Action):
                yield node

    def snapshots(self) -> Iterator[VariableSnapshot]:
        """Iterate over all snapshots."""
        for node in self.graph.nodes:
            if isinstance(node, VariableSnapshot):
                yield node

    def write_dependencies_of(self, action: Action) -> List[VariableSnapshot]:
        """Get write_dependency edges: snapshots created by an action."""
        return [n for n in self.graph.successors(action) if isinstance(n, VariableSnapshot)]

    def writer_of(self, snapshot: VariableSnapshot) -> Action:
        """Get the action that has write_dependency to this snapshot."""
        for n in self.graph.predecessors(snapshot):
            if isinstance(n, Action):
                return n
        return None

    def read_dependencies_of(self, snapshot: VariableSnapshot) -> List[Action]:
        """Get read_dependency edges: actions that read this snapshot."""
        return [n for n in self.graph.successors(snapshot) if isinstance(n, Action)]

    def read_dependencies_to(self, action: Action) -> List[VariableSnapshot]:
        """Get read_dependency edges: snapshots read by this action."""
        return [n for n in self.graph.predecessors(action) if isinstance(n, VariableSnapshot)]

    def state_at(self, action_id: int) -> Dict[str, VariableSnapshot]:
        """Get the active snapshot for each variable at a given action."""
        state = {}
        for snapshot in self.snapshots():
            if snapshot.created_at <= action_id:
                # Check if there's a later snapshot for this variable
                var = snapshot.variable
                if var not in state or state[var].created_at < snapshot.created_at:
                    state[var] = snapshot
        return state

    # === Analysis Methods ===

    def total_write_dependencies(self) -> int:
        """Count total write_dependency edges (Action -> VariableSnapshot)."""
        return sum(1 for _ in self.snapshots())

    def total_read_dependencies(self) -> int:
        """Count total read_dependency edges (cross-action only)."""
        return sum(len(self.read_dependencies_of(s)) for s in self.snapshots())

    def snapshots_with_read_dependencies(self) -> Dict[VariableSnapshot, List[Action]]:
        """Get all snapshots that have read_dependency edges."""
        result = {}
        for snapshot in self.snapshots():
            readers = self.read_dependencies_of(snapshot)
            if readers:
                result[snapshot] = readers
        return result

    def snapshots_without_read_dependencies(self) -> List[VariableSnapshot]:
        """Find snapshots with no read_dependency edges (never read by other actions)."""
        return [s for s in self.snapshots() if not self.read_dependencies_of(s)]

    def redefined_variables(self) -> Dict[str, List[VariableSnapshot]]:
        """Find variables with multiple snapshots (multiple write_dependencies)."""
        var_snapshots: Dict[str, List[VariableSnapshot]] = {}
        for s in self.snapshots():
            var_snapshots.setdefault(s.variable, []).append(s)
        return {v: sorted(ss, key=lambda x: x.created_at)
                for v, ss in var_snapshots.items() if len(ss) > 1}

    def stats(self) -> Dict[str, any]:
        """Compute summary statistics."""
        total_snapshots = sum(1 for _ in self.snapshots())
        read_deps = self.total_read_dependencies()
        unused = len(self.snapshots_without_read_dependencies())

        return {
            'total_actions': sum(1 for _ in self.actions()),
            'successful_actions': sum(1 for a in self.actions() if a.success),
            'failed_actions': sum(1 for a in self.actions() if not a.success),
            'total_variables': len(set(s.variable for s in self.snapshots())),
            'total_snapshots': total_snapshots,
            'write_dependency_edges': self.total_write_dependencies(),
            'read_dependency_edges': read_deps,
            'snapshots_without_read_dependencies': unused,
            # New metrics for reuse analysis
            'avg_reads_per_snapshot': read_deps / total_snapshots if total_snapshots > 0 else 0,
            'snapshot_utilization_rate': 1 - unused / total_snapshots if total_snapshots > 0 else 0,
        }

    def print_summary(self) -> None:
        """Print analysis summary."""
        print("\n" + "=" * 80)
        print("APPLICATION HISTORY GRAPH")
        print("=" * 80)

        # Actions with their dependencies
        print("\n--- Actions ---")
        for action in sorted(self.actions(), key=lambda a: a.id):
            writes = self.write_dependencies_of(action)
            reads = self.read_dependencies_to(action)
            status = "OK" if action.success else "ERR"
            writes_str = [str(s) for s in writes]
            reads_str = [str(s) for s in reads]
            print(f"  {action.id} [{status}]: writes={writes_str}, reads={reads_str}")

        # State at each timestamp
        action_ids = sorted(a.id for a in self.actions())
        if action_ids:
            print("\n--- State at Each Timestamp ---")
            for aid in action_ids:
                state = self.state_at(aid)
                vars_list = sorted(state.keys())
                snaps_list = [str(state[v]) for v in vars_list]
                print(f"  After {aid}: vars={vars_list}, snapshots={snaps_list}")

        # Read dependencies (all are cross-action by design)
        reads = self.snapshots_with_read_dependencies()
        if reads:
            print("\n--- Read Dependencies ---")
            for snap, actions in sorted(reads.items(), key=lambda x: str(x[0])):
                action_ids = [a.id for a in actions]
                print(f"  {snap} -> read by {action_ids}")
        else:
            print("\n--- No Read Dependencies (Monolithic Pattern) ---")

        # Snapshots without read dependencies
        unread = self.snapshots_without_read_dependencies()
        if unread:
            print(f"\n--- Snapshots Without Read Dependencies ({len(unread)}) ---")
            for s in sorted(unread, key=str):
                print(f"  {s}")

        # Stats
        print("\n--- Stats ---")
        for k, v in self.stats().items():
            print(f"  {k}: {v}")

        print("\n" + "=" * 80)

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        action_ids = sorted(a.id for a in self.actions())
        return {
            'actions': [
                {
                    'id': a.id,
                    'success': a.success,
                    'error_msg': a.error_msg,
                    'write_dependencies': [str(s) for s in self.write_dependencies_of(a)],
                    'read_dependencies': [str(s) for s in self.read_dependencies_to(a)],
                }
                for a in sorted(self.actions(), key=lambda x: x.id)
            ],
            'snapshots': [str(s) for s in sorted(self.snapshots(), key=str)],
            'timestamp_state': {
                aid: {
                    'variables': sorted(self.state_at(aid).keys()),
                    'snapshots': {v: str(s) for v, s in self.state_at(aid).items()}
                }
                for aid in action_ids
            },
            'cross_action_read_dependencies': {
                str(s): [a.id for a in actions]
                for s, actions in self.snapshots_with_read_dependencies().items()
            },
            'snapshots_without_read_dependencies': [str(s) for s in self.snapshots_without_read_dependencies()],
            'redefined_variables': {
                v: [str(s) for s in ss]
                for v, ss in self.redefined_variables().items()
            },
            'stats': self.stats(),
        }

    # === Visualization ===

    def visualize(self, output_path: str = "graph.png", rankdir: str = "TB",
                  show_legend: bool = True) -> str:
        """
        Render the graph to an image file using Graphviz.

        Args:
            output_path: Output file path (supports .png, .svg, .pdf)
            rankdir: Layout direction - "TB" (top-bottom) or "LR" (left-right)
            show_legend: Whether to include a legend

        Returns:
            Path to the generated image file
        """
        try:
            from graphviz import Digraph
        except ImportError:
            raise ImportError("graphviz package required. Install with: pip install graphviz")

        # Determine output format from extension
        path = Path(output_path)
        fmt = path.suffix[1:] if path.suffix else 'png'
        base_name = str(path.with_suffix(''))

        # Create the graph
        dot = Digraph(format=fmt)
        dot.attr(rankdir=rankdir, splines='spline', nodesep='0.5', ranksep='0.8')

        # Node colors
        COLOR_BORDER = 'black'           # Black border for all nodes

        # Edge color
        COLOR_EDGE = 'black'

        # Icons
        EYE_ICON = ' 👁'
        CROSS_ICON = ' ❌'

        # Group actions by rank (same action ID = same rank for layout)
        actions_sorted = sorted(self.actions(), key=lambda a: a.id)

        # Add action nodes (rectangles) - all transparent, cross in label for errors
        for action in actions_sorted:
            dot.node(
                f"action_{action.id}",
                label=action.label(show_error=True),
                shape='box',
                style='rounded',
                color=COLOR_BORDER,
                fontname='Helvetica',
                fontsize='10'
            )

        # Add snapshot nodes (ellipses) - all transparent, cross in label for invalid
        for snapshot in self.snapshots():
            # Get visibility and validity flags from node attributes
            node_data = self.graph.nodes[snapshot]
            is_visible = node_data.get('visible', False)
            is_valid = node_data.get('valid', True)

            # Truncate long variable names
            var_name = snapshot.variable
            if len(var_name) > 20:
                var_name = var_name[:17] + "..."

            # Build label with icons
            label = var_name
            if is_visible:
                label += EYE_ICON
            if not is_valid:
                label += CROSS_ICON

            dot.node(
                f"snap_{snapshot}",
                label=label,
                shape='ellipse',
                style='',
                color=COLOR_BORDER,
                fontname='Helvetica',
                fontsize='9'
            )

        # Add edges (all black, no labels)
        for snapshot in self.snapshots():
            # Write dependency: Action -> Snapshot (solid)
            writer = self.writer_of(snapshot)
            if writer:
                dot.edge(
                    f"action_{writer.id}",
                    f"snap_{snapshot}",
                    color=COLOR_EDGE,
                    penwidth='1.5'
                )

            # Read dependencies: Snapshot -> Action (dashed)
            for reader in self.read_dependencies_of(snapshot):
                dot.edge(
                    f"snap_{snapshot}",
                    f"action_{reader.id}",
                    color=COLOR_EDGE,
                    style='dashed',
                    penwidth='1.5'
                )

        # Add invisible edges between consecutive actions for layout ordering
        for i in range(len(actions_sorted) - 1):
            dot.edge(
                f"action_{actions_sorted[i].id}",
                f"action_{actions_sorted[i+1].id}",
                style='invis',
                weight='10'
            )

        # Add legend as a subgraph
        if show_legend:
            with dot.subgraph(name='cluster_legend') as legend:
                legend.attr(label='Legend', fontname='Helvetica', fontsize='10',
                           style='rounded', color='gray')

                # Legend nodes - Actions
                legend.node('leg_action_ok', 'Step N', shape='box',
                           style='rounded', color=COLOR_BORDER,
                           fontname='Helvetica', fontsize='8')
                legend.node('leg_action_err', 'Step N ❌', shape='box',
                           style='rounded', color=COLOR_BORDER,
                           fontname='Helvetica', fontsize='8')

                # Legend nodes - Variables
                legend.node('leg_snapshot', 'var', shape='ellipse',
                           style='', color=COLOR_BORDER,
                           fontname='Helvetica', fontsize='8')
                legend.node('leg_snapshot_vis', 'var 👁', shape='ellipse',
                           style='', color=COLOR_BORDER,
                           fontname='Helvetica', fontsize='8')
                legend.node('leg_snapshot_inv', 'var ❌', shape='ellipse',
                           style='', color=COLOR_BORDER,
                           fontname='Helvetica', fontsize='8')

                # Legend edges (use xlabel to avoid spline warning)
                legend.node('leg_src1', '', shape='point', width='0')
                legend.node('leg_dst1', '', shape='point', width='0')
                legend.edge('leg_src1', 'leg_dst1', xlabel='writes', color=COLOR_EDGE,
                           fontname='Helvetica', fontsize='8', penwidth='1.5')

                legend.node('leg_src2', '', shape='point', width='0')
                legend.node('leg_dst2', '', shape='point', width='0')
                legend.edge('leg_src2', 'leg_dst2', xlabel='reads', color=COLOR_EDGE,
                           style='dashed', fontname='Helvetica', fontsize='8', penwidth='1.5')

                # Arrange legend items vertically
                legend.attr(rank='sink')

        # Render
        dot.render(base_name, cleanup=True)
        final_path = f"{base_name}.{fmt}"
        print(f"Graph saved to: {final_path}")
        return final_path


# =============================================================================
# Code Analysis (for CodeAgent traces)
# =============================================================================

class GlobalScopeAnalyzer(ast.NodeVisitor):
    """AST visitor that extracts only module-level (global) variable definitions and references.

    Only tracks variables at the top level of the module. Ignores:
    - Variables inside function/class bodies
    - Loop iteration variables (for x in ...)
    - Comprehension variables
    - Exception handler variables
    - With statement variables

    Tracks:
    - Direct assignments at module level: x = value
    - Subscript mutations: df['col'] = value (mutates df)
    - Augmented assignments: x += value
    - In-place method calls: df.dropna(inplace=True)
    - Imports
    """

    INPLACE_METHODS = {
        'drop', 'dropna', 'fillna', 'replace', 'rename', 'reset_index',
        'set_index', 'sort_values', 'sort_index', 'drop_duplicates',
        'update', 'clip', 'interpolate', 'bfill', 'ffill', 'where', 'mask',
    }

    def __init__(self):
        self.defined: Set[str] = set()
        self.referenced: Set[str] = set()

    def analyze(self, code: str) -> Tuple[Set[str], Set[str]]:
        """Analyze code and return (defined, referenced) global variables."""
        try:
            tree = ast.parse(code)
        except SyntaxError:
            return set(), set()

        # Only process top-level statements
        for node in tree.body:
            self._process_statement(node)

        return self.defined, self.referenced

    def _process_statement(self, node: ast.AST) -> None:
        """Process a single top-level statement."""
        if isinstance(node, ast.Assign):
            # x = value or x, y = values
            for target in node.targets:
                self._extract_defined(target)
            self._extract_referenced(node.value)

        elif isinstance(node, ast.AugAssign):
            # x += value
            self._extract_defined(node.target)
            self._extract_referenced(node.value)

        elif isinstance(node, ast.Expr):
            # Expression statement (e.g., function call, method call)
            self._process_expr_statement(node.value)

        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.asname or alias.name.split('.')[0]
                self.defined.add(name)

        elif isinstance(node, ast.ImportFrom):
            for alias in node.names:
                name = alias.asname or alias.name
                if name != '*':
                    self.defined.add(name)

        elif isinstance(node, (ast.For, ast.While)):
            # Process loop body but NOT the iteration variable
            # The iteration variable is local to the loop conceptually
            if isinstance(node, ast.For):
                self._extract_referenced(node.iter)
            elif isinstance(node, ast.While):
                self._extract_referenced(node.test)
            for stmt in node.body:
                self._process_statement(stmt)
            for stmt in node.orelse:
                self._process_statement(stmt)

        elif isinstance(node, ast.If):
            self._extract_referenced(node.test)
            for stmt in node.body:
                self._process_statement(stmt)
            for stmt in node.orelse:
                self._process_statement(stmt)

        elif isinstance(node, ast.With):
            for item in node.items:
                self._extract_referenced(item.context_expr)
            for stmt in node.body:
                self._process_statement(stmt)

        elif isinstance(node, ast.Try):
            for stmt in node.body:
                self._process_statement(stmt)
            for stmt in node.orelse:
                self._process_statement(stmt)
            for stmt in node.finalbody:
                self._process_statement(stmt)
            for handler in node.handlers:
                for stmt in handler.body:
                    self._process_statement(stmt)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            # Function/class definitions create a name but we don't recurse into body
            self.defined.add(node.name)

    def _process_expr_statement(self, node: ast.AST) -> None:
        """Process an expression statement, checking for in-place mutations."""
        if isinstance(node, ast.Call):
            # Check for inplace=True method calls
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in self.INPLACE_METHODS:
                    for kw in node.keywords:
                        if kw.arg == 'inplace' and self._is_true(kw.value):
                            base = self._get_base_name(node.func.value)
                            if base:
                                self.defined.add(base)
                # Also track the object being called on as referenced
                self._extract_referenced(node.func.value)
            # Track all arguments as referenced
            for arg in node.args:
                self._extract_referenced(arg)
        else:
            self._extract_referenced(node)

    def _extract_defined(self, node: ast.AST) -> None:
        """Extract variable names being assigned to."""
        if isinstance(node, ast.Name):
            self.defined.add(node.id)
        elif isinstance(node, ast.Tuple) or isinstance(node, ast.List):
            for elt in node.elts:
                self._extract_defined(elt)
        elif isinstance(node, ast.Subscript):
            # df['col'] = value -> df is mutated
            base = self._get_base_name(node.value)
            if base:
                self.defined.add(base)
        elif isinstance(node, ast.Attribute):
            # obj.attr = value -> obj is mutated
            base = self._get_base_name(node.value)
            if base:
                self.defined.add(base)

    def _extract_referenced(self, node: ast.AST) -> None:
        """Extract variable names being read."""
        if node is None:
            return
        if isinstance(node, ast.Name):
            self.referenced.add(node.id)
        elif isinstance(node, ast.Attribute):
            self._extract_referenced(node.value)
        elif isinstance(node, ast.Subscript):
            self._extract_referenced(node.value)
            self._extract_referenced(node.slice)
        elif isinstance(node, ast.Call):
            self._extract_referenced(node.func)
            for arg in node.args:
                self._extract_referenced(arg)
            for kw in node.keywords:
                self._extract_referenced(kw.value)
        elif isinstance(node, ast.BinOp):
            self._extract_referenced(node.left)
            self._extract_referenced(node.right)
        elif isinstance(node, ast.UnaryOp):
            self._extract_referenced(node.operand)
        elif isinstance(node, ast.Compare):
            self._extract_referenced(node.left)
            for comp in node.comparators:
                self._extract_referenced(comp)
        elif isinstance(node, ast.BoolOp):
            for val in node.values:
                self._extract_referenced(val)
        elif isinstance(node, ast.IfExp):
            self._extract_referenced(node.test)
            self._extract_referenced(node.body)
            self._extract_referenced(node.orelse)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                self._extract_referenced(elt)
        elif isinstance(node, ast.Dict):
            for k in node.keys:
                self._extract_referenced(k)
            for v in node.values:
                self._extract_referenced(v)
        elif isinstance(node, ast.Index):  # Python 3.8 compatibility
            self._extract_referenced(node.value)
        elif isinstance(node, ast.Slice):
            self._extract_referenced(node.lower)
            self._extract_referenced(node.upper)
            self._extract_referenced(node.step)
        elif isinstance(node, ast.JoinedStr):  # f-strings
            for val in node.values:
                if isinstance(val, ast.FormattedValue):
                    self._extract_referenced(val.value)
        elif isinstance(node, (ast.ListComp, ast.SetComp, ast.GeneratorExp)):
            # For comprehensions, only track the iterable, not the loop variable
            self._extract_referenced(node.elt)
            for gen in node.generators:
                self._extract_referenced(gen.iter)
        elif isinstance(node, ast.DictComp):
            self._extract_referenced(node.key)
            self._extract_referenced(node.value)
            for gen in node.generators:
                self._extract_referenced(gen.iter)

    def _get_base_name(self, node: ast.AST) -> Optional[str]:
        """Get the root variable name from a chain like df['col'].method()."""
        if isinstance(node, ast.Name):
            return node.id
        elif isinstance(node, ast.Subscript):
            return self._get_base_name(node.value)
        elif isinstance(node, ast.Attribute):
            return self._get_base_name(node.value)
        return None

    def _is_true(self, node: ast.AST) -> bool:
        """Check if a node represents the value True."""
        if isinstance(node, ast.Constant):
            return node.value is True
        if isinstance(node, ast.NameConstant):  # Python 3.7
            return node.value is True
        return False


class PrintedVarsAnalyzer(ast.NodeVisitor):
    """AST visitor to extract variables that are printed."""

    def __init__(self):
        self.printed: Set[str] = set()

    def visit_Call(self, node: ast.Call) -> None:
        # Check if this is a print() call
        if isinstance(node.func, ast.Name) and node.func.id == 'print':
            # Extract all variable names from print arguments
            for arg in node.args:
                self._extract_names(arg)
        self.generic_visit(node)

    def _extract_names(self, node: ast.AST) -> None:
        """Recursively extract variable names from an expression."""
        if isinstance(node, ast.Name):
            self.printed.add(node.id)
        elif isinstance(node, ast.Attribute):
            # For obj.attr, add the base object
            self._extract_names(node.value)
        elif isinstance(node, ast.Subscript):
            # For obj[key], add the base object
            self._extract_names(node.value)
        elif isinstance(node, ast.BinOp):
            self._extract_names(node.left)
            self._extract_names(node.right)
        elif isinstance(node, ast.Compare):
            self._extract_names(node.left)
            for comp in node.comparators:
                self._extract_names(comp)
        elif isinstance(node, ast.Call):
            for arg in node.args:
                self._extract_names(arg)
        elif isinstance(node, (ast.List, ast.Tuple, ast.Set)):
            for elt in node.elts:
                self._extract_names(elt)
        elif isinstance(node, ast.Dict):
            for key in node.keys:
                if key:
                    self._extract_names(key)
            for val in node.values:
                self._extract_names(val)
        elif isinstance(node, ast.JoinedStr):  # f-strings
            for val in node.values:
                if isinstance(val, ast.FormattedValue):
                    self._extract_names(val.value)


def extract_printed_vars(code: str) -> Set[str]:
    """Extract variables that are printed in the code."""
    try:
        tree = ast.parse(code)
        analyzer = PrintedVarsAnalyzer()
        analyzer.visit(tree)
        return analyzer.printed - BUILTINS
    except SyntaxError:
        return set()


BUILTINS = {
    'print', 'len', 'range', 'str', 'int', 'float', 'bool', 'list', 'dict',
    'set', 'tuple', 'open', 'type', 'isinstance', 'sum', 'min', 'max', 'abs',
    'round', 'sorted', 'enumerate', 'zip', 'map', 'filter', 'any', 'all',
    'next', 'iter', 'True', 'False', 'None', 'Exception', 'ValueError',
    'KeyError', 'TypeError', 'IndexError', 'AttributeError', 'RuntimeError',
    'format', 'repr', 'id', 'dir', 'vars', 'globals', 'locals', 'hasattr',
    'getattr', 'setattr', 'super', 'input', 'exit', 'hash', 'ord', 'chr',
    'bytes', 'object', 'slice', 'reversed', 'compile', 'exec', 'eval',
    '__name__', '__file__', 'final_answer', 'np', 'pd',
}


def analyze_code_regex_fallback(code: str) -> Tuple[Set[str], Set[str]]:
    """Regex-based fallback for extracting variables from malformed code.

    Used when AST parsing fails (e.g., truncated code in traces).
    Less accurate than AST but better than nothing.
    """
    import re

    defined = set()
    referenced = set()

    # === DEFINED VARIABLES ===

    # Pattern 1: Simple assignments - "variable = ..."
    # Matches: x = ..., df = ..., msa_reports_filtered = ...
    assignment_pattern = r'^[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)\s*='
    for match in re.finditer(assignment_pattern, code, re.MULTILINE):
        var = match.group(1)
        if var not in BUILTINS and var not in {'np', 'pd'}:
            defined.add(var)

    # Pattern 2: In-place mutations - "variable.method(...inplace=True...)"
    inplace_pattern = r'^[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*\w+\s*\([^)]*inplace\s*=\s*True'
    for match in re.finditer(inplace_pattern, code, re.MULTILINE):
        var = match.group(1)
        if var not in BUILTINS and var not in {'np', 'pd'}:
            defined.add(var)

    # Pattern 3: Subscript in-place mutations - "df['col'].method(...inplace=True...)"
    subscript_inplace_pattern = r'^[ \t]*([a-zA-Z_][a-zA-Z0-9_]*)\s*\[[^\]]+\]\s*\.\s*\w+\s*\([^)]*inplace\s*=\s*True'
    for match in re.finditer(subscript_inplace_pattern, code, re.MULTILINE):
        var = match.group(1)
        if var not in BUILTINS and var not in {'np', 'pd'}:
            defined.add(var)

    # NOTE: We intentionally do NOT track for-loop iteration variables as global definitions.
    # Loop variables like `for f in files:` are local to the loop scope conceptually.

    # Pattern 4: Import statements
    import_pattern = r'^[ \t]*import\s+([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(import_pattern, code, re.MULTILINE):
        var = match.group(1)
        defined.add(var)

    from_import_pattern = r'^[ \t]*from\s+\S+\s+import\s+([a-zA-Z_][a-zA-Z0-9_,\s]+)'
    for match in re.finditer(from_import_pattern, code, re.MULTILINE):
        imports = match.group(1)
        for var in re.findall(r'([a-zA-Z_][a-zA-Z0-9_]*)', imports):
            if var not in BUILTINS:
                defined.add(var)

    # === REFERENCED VARIABLES ===

    # Pattern R1: Method calls - "variable.method(...)"
    method_call_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\.\s*[a-zA-Z_]\w*\s*\('
    for match in re.finditer(method_call_pattern, code):
        var = match.group(1)
        if var not in BUILTINS and var not in {'np', 'pd', 'os', 'sys', 're', 'json', 'glob'}:
            referenced.add(var)

    # Pattern R2: Subscript access - "variable[...]" or "variable['key']"
    subscript_pattern = r'([a-zA-Z_][a-zA-Z0-9_]*)\s*\['
    for match in re.finditer(subscript_pattern, code):
        var = match.group(1)
        if var not in BUILTINS and var not in {'np', 'pd'}:
            referenced.add(var)

    # Pattern R3: Right-hand side of assignments - "x = variable" or "x = variable.method()"
    # This catches cases like: msa_reports = df.groupby(...)
    rhs_pattern = r'=\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[\.\[\(]'
    for match in re.finditer(rhs_pattern, code):
        var = match.group(1)
        if var not in BUILTINS and var not in {'np', 'pd'}:
            referenced.add(var)

    # Pattern R4: Function arguments - "func(variable, ...)" or "func(x, variable)"
    # Look for identifiers inside parentheses
    func_args_pattern = r'\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*[,\)]'
    for match in re.finditer(func_args_pattern, code):
        var = match.group(1)
        if var not in BUILTINS and var not in {'np', 'pd'}:
            referenced.add(var)

    # Pattern R5: Variables in expressions - "variable + ..." or "... + variable"
    expr_pattern = r'(?:[\+\-\*/\%\<\>\=\!]\s*)([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(expr_pattern, code):
        var = match.group(1)
        if var not in BUILTINS and var not in {'np', 'pd', 'True', 'False', 'None'}:
            referenced.add(var)

    # Pattern R6: isin() and similar methods - ".isin(variable['col'])"
    isin_pattern = r'\.isin\s*\(\s*([a-zA-Z_][a-zA-Z0-9_]*)'
    for match in re.finditer(isin_pattern, code):
        var = match.group(1)
        if var not in BUILTINS and var not in {'np', 'pd'}:
            referenced.add(var)

    # Remove self-references (if a variable is both defined and referenced in same code,
    # it's often a mutation like df['col'] = df['col'].method())
    # We still want to capture these as references for cross-action dependencies

    return defined, referenced


def analyze_code(code: str) -> Tuple[Set[str], Set[str]]:
    """Extract defined and referenced global variables from Python code.

    Only tracks variables at module/global scope. Ignores:
    - Loop iteration variables (for x in ...)
    - Comprehension variables
    - Function parameters
    - Variables inside function/class bodies

    Returns:
        (defined, referenced) where defined includes both new assignments
        and in-place mutations at global scope.

    Uses AST parsing first, falls back to regex for malformed code.
    """
    analyzer = GlobalScopeAnalyzer()
    defined, referenced = analyzer.analyze(code)

    if not defined and not referenced:
        # AST parsing might have failed (SyntaxError), try regex fallback
        defined, referenced = analyze_code_regex_fallback(code)

    # Filter out common imports/builtins
    defined = defined - BUILTINS - {'np', 'pd'}
    referenced = referenced - BUILTINS

    return defined, referenced


def extract_failing_line(error_msg: str) -> Optional[str]:
    """Extract the failing line of code from an error message.

    Handles formats like:
    - "Code execution failed at line 'x = foo()' due to: ..."
    - "Code execution failed at line 'x = foo()' with error: ..."
    """
    import re

    if not error_msg:
        return None

    # Pattern: "at line '...' due to" or "at line '...' with"
    patterns = [
        r"at line ['\"](.+?)['\"] (?:due to|with)",
        r"at line ['\"](.+?)['\"]$",
        r"failed at line ['\"](.+?)['\"]",
    ]

    for pattern in patterns:
        match = re.search(pattern, error_msg, re.DOTALL)
        if match:
            return match.group(1).strip()

    return None


def find_line_number(code: str, failing_line: str) -> Optional[int]:
    """Find the line number (1-indexed) of the failing line in the code."""
    if not failing_line:
        return None

    lines = code.split('\n')
    failing_line_stripped = failing_line.strip()

    for i, line in enumerate(lines):
        line_stripped = line.strip()
        # Skip empty lines
        if not line_stripped:
            continue
        # Exact match
        if failing_line_stripped == line_stripped:
            return i + 1
        # Failing line is contained in this line
        if failing_line_stripped in line_stripped:
            return i + 1
        # This line is a significant part of the failing line (for multi-line statements)
        if len(line_stripped) > 10 and line_stripped in failing_line_stripped:
            return i + 1
        # Check if it starts with the same significant prefix (for truncated matches)
        if len(failing_line_stripped) > 20:
            prefix = failing_line_stripped[:30]
            if line_stripped.startswith(prefix):
                return i + 1

    return None


def analyze_code_until_line(code: str, until_line: int) -> Tuple[Set[str], Set[str]]:
    """Analyze code only up to (but not including) a specific line number.

    Args:
        code: The full code
        until_line: The 1-indexed line number to stop before

    Returns:
        (defined, referenced) variables from the partial code
    """
    if until_line is None or until_line <= 1:
        return set(), set()

    lines = code.split('\n')
    partial_code = '\n'.join(lines[:until_line - 1])

    if not partial_code.strip():
        return set(), set()

    return analyze_code(partial_code)


def analyze_code_from_line(code: str, from_line: int) -> Tuple[Set[str], Set[str]]:
    """Analyze code starting from a specific line number (inclusive).

    Args:
        code: The full code
        from_line: The 1-indexed line number to start from

    Returns:
        (defined, referenced) variables from the code starting at from_line
    """
    if from_line is None or from_line < 1:
        return set(), set()

    lines = code.split('\n')
    if from_line > len(lines):
        return set(), set()

    partial_code = '\n'.join(lines[from_line - 1:])

    if not partial_code.strip():
        return set(), set()

    return analyze_code(partial_code)


def extract_printed_vars_until_line(code: str, until_line: int) -> Set[str]:
    """Extract printed variables from code up to a specific line."""
    if until_line is None or until_line <= 1:
        return set()

    lines = code.split('\n')
    partial_code = '\n'.join(lines[:until_line - 1])

    if not partial_code.strip():
        return set()

    return extract_printed_vars(partial_code)


# =============================================================================
# Trace Parsers
# =============================================================================

def parse_code_agent_trace(path: str) -> ApplicationHistoryGraph:
    """Parse a CodeAgent trace and build the graph.

    For failed actions, attempts to determine which variables were successfully
    defined before the failing line (partial execution state preservation).
    Also tracks:
    - visible_vars: Variables that are printed (for eye icon display)
    - invalid_vars: Variables defined AFTER the error line (for red display)
    """
    with open(path) as f:
        trace = json.load(f)

    graph = ApplicationHistoryGraph()
    for step in trace:
        action_id = step.get('step', 0)
        code = step.get('code', '')
        success = 'error' not in step

        # Extract error message if present
        error_msg = None
        if not success:
            error_data = step.get('error', {})
            if isinstance(error_data, dict):
                error_msg = error_data.get('message', str(error_data))
            elif isinstance(error_data, str):
                # Try to parse string representation of dict
                try:
                    parsed = ast.literal_eval(error_data)
                    if isinstance(parsed, dict):
                        error_msg = parsed.get('message', error_data)
                    else:
                        error_msg = error_data
                except (ValueError, SyntaxError):
                    error_msg = error_data
            else:
                error_msg = str(error_data)

        # Initialize variables
        defined = set()
        referenced = set()
        visible_vars = set()
        invalid_vars = set()

        if code:
            if success:
                # Successful action: analyze all code
                defined, referenced = analyze_code(code)
                visible_vars = extract_printed_vars(code)
            else:
                # Failed action: try to determine partial state
                failing_line = extract_failing_line(error_msg)
                line_num = find_line_number(code, failing_line) if failing_line else None

                # For read dependencies, we need ALL referenced variables from the full code
                # because reads are attempted regardless of where the error occurs
                _, all_referenced = analyze_code(code)

                if line_num and line_num > 1:
                    # Analyze code before the failing line (valid variables)
                    defined, _ = analyze_code_until_line(code, line_num)
                    visible_vars = extract_printed_vars_until_line(code, line_num)

                    # Analyze code from the failing line onwards (invalid variables)
                    invalid_defined, _ = analyze_code_from_line(code, line_num)
                    invalid_vars = invalid_defined - defined  # Only vars not already defined

                    # Use all referenced variables for read dependencies
                    referenced = all_referenced
                else:
                    # Couldn't determine failing line, assume no variables were defined
                    # All defined variables are invalid
                    all_defined, _ = analyze_code(code)
                    invalid_vars = all_defined
                    visible_vars = set()  # Can't determine visibility
                    referenced = all_referenced

        action = Action(id=action_id, success=success, error_msg=error_msg)
        graph.add_action(action, defined, referenced,
                        visible_vars=visible_vars, invalid_vars=invalid_vars)

    return graph


def extract_function_params(code: str) -> List[str]:
    """Extract parameter names from a function definition."""
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                return [arg.arg for arg in node.args.args if arg.arg != 'self']
    except SyntaxError:
        pass
    return []


def parse_dataflow_trace(path: str) -> ApplicationHistoryGraph:
    """Parse a Dataflow messages trace and build the graph.

    For Dataflow, all variables (operators) are visible by default since
    their outputs are displayed in the execution results.
    """
    with open(path) as f:
        messages = json.load(f)

    # First pass: collect tool results and check for errors
    tool_results: Dict[str, Tuple[bool, Optional[str]]] = {}  # toolCallId -> (success, error_msg)
    for msg in messages:
        if msg.get('role') != 'tool':
            continue
        content = msg.get('content', [])
        if isinstance(content, str):
            continue
        for item in content:
            if item.get('type') != 'tool-result':
                continue
            tool_call_id = item.get('toolCallId', '')
            output = item.get('output', {})
            value = output.get('value', '') if isinstance(output, dict) else ''
            # Check for error markers in the output
            success = '[ERROR]' not in value
            error_msg = None
            if not success:
                # Extract error message from the output
                lines = value.split('\n')
                for line in lines:
                    if '[ERROR]' in line:
                        error_msg = line.replace('[ERROR]', '').strip()[:60]
                        break
            tool_results[tool_call_id] = (success, error_msg)

    # Second pass: process tool calls
    graph = ApplicationHistoryGraph()
    action_id = 0

    for msg in messages:
        if msg.get('role') != 'assistant':
            continue
        content = msg.get('content', [])
        if isinstance(content, str):
            continue

        for item in content:
            if item.get('type') != 'tool-call':
                continue
            if item.get('toolName') not in ('addOperator', 'modifyOperator'):
                continue

            action_id += 1
            tool_call_id = item.get('toolCallId', '')
            input_data = item.get('input', {})
            operator_id = input_data.get('operatorId', '')
            code = input_data.get('code', '')

            defined = {operator_id} if operator_id else set()
            referenced = set(extract_function_params(code))

            # Look up success and error_msg from tool result
            success, error_msg = tool_results.get(tool_call_id, (True, None))

            # For Dataflow, all operators are visible (their output is shown)
            # Invalid vars only apply to failed operators
            visible_vars = defined  # All defined operators are visible
            invalid_vars = set() if success else defined  # If failed, the operator output is invalid

            action = Action(id=action_id, success=success, error_msg=error_msg)
            graph.add_action(action, defined if success else set(), referenced,
                           visible_vars=visible_vars, invalid_vars=invalid_vars)

    return graph


# =============================================================================
# Main
# =============================================================================

def analyze(path: str, trace_type: str = 'auto', verbose: bool = True) -> ApplicationHistoryGraph:
    """Analyze a trace file and return the graph."""
    # Auto-detect trace type based on filename
    if trace_type == 'auto':
        filename = Path(path).name.lower()
        if 'code_agent' in filename or 'codeagent' in filename:
            trace_type = 'codeagent'
        elif 'dataflow' in filename or 'messages' in filename:
            trace_type = 'dataflow'
        else:
            trace_type = 'codeagent'  # default

    print(f"\nAnalyzing ({trace_type}): {path}")

    if trace_type == 'dataflow':
        graph = parse_dataflow_trace(path)
    else:
        graph = parse_code_agent_trace(path)

    if verbose:
        graph.print_summary()

    return graph


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Analyze agent traces and build application history graphs.')
    parser.add_argument('path', nargs='?',
                        default="result-analysis/o4mini-oracle/2_dataflow_succeed/legal-easy-19/code_agent_trace.json",
                        help='Path to trace file')
    parser.add_argument('--visualize', '-v', action='store_true',
                        help='Generate visualization image')
    parser.add_argument('--output', '-o', type=str, default=None,
                        help='Output path for visualization (default: <input>.png)')
    parser.add_argument('--format', '-f', type=str, default='png',
                        choices=['png', 'svg', 'pdf'],
                        help='Output format for visualization')
    parser.add_argument('--no-legend', action='store_true',
                        help='Hide legend in visualization')
    parser.add_argument('--quiet', '-q', action='store_true',
                        help='Suppress verbose output')

    args = parser.parse_args()

    if not Path(args.path).exists():
        print(f"Error: File not found: {args.path}")
        sys.exit(1)

    graph = analyze(args.path, verbose=not args.quiet)

    # Save JSON analysis
    output_path = Path(args.path).with_suffix('.graph_analysis.json')
    with open(output_path, 'w') as f:
        json.dump(graph.to_dict(), f, indent=2)
    print(f"\nSaved JSON to: {output_path}")

    # Generate visualization if requested
    if args.visualize:
        vis_output = args.output or str(Path(args.path).with_suffix(f'.{args.format}'))
        graph.visualize(vis_output, show_legend=not args.no_legend)


if __name__ == "__main__":
    main()
