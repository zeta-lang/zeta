use std::num::NonZero;

use crate::nodes::{DepNode, Edge, NodeIdx};

struct WorkItem {
    current_vertex: NodeIdx,
    next_neighbor: usize,
}

/// Partitions a directed graph into its strongly connected components using
/// Tarjan's algorithm (a depth-first search), implemented iteratively.
///
/// # What's a Strongly Connected Component?
///
/// Vertex: a single node (such as `C`)
///
/// Edge: a direct connection (such as `A -> C`)
///
/// Path: a sequence of directed edges from one vertex to another (such as `A -> C -> B`)
///
/// A strongly connected component is a maximal subset of vertices such that every
/// vertex in the subset is reachable from every other vertex in it, following edge
/// direction. For any two vertices `u` and `v` there must be a path from `u` to `v`
/// and another from `v` to `u`.
///
///       ┌─────────┐
///       ▼         │
///      (A) ───► (B)
///       ▲         │
///       │         ▼
///       └─────── (C)
///
/// Here all three nodes are in one component: you can reach B and C from A,
/// C and A from B, and A and B from C.
///
/// # Applications
/// Used in the Zeta compiler to:
/// - detect circular imports
/// - detect recursive cycles
/// - detect files in the same SCC
/// - and most importantly: to figure out the order in which files should be compiled.
pub fn find_sccs_of_nodes(nodes: &[DepNode]) -> Vec<Vec<NodeIdx>> {
    const DONE: NonZero<usize> = NonZero::<usize>::MAX;

    let n = nodes.len();
    // discovery order of each vertex; None = not yet visited
    let mut index: Vec<Option<NonZero<usize>>> = vec![None; n];
    // vertices visited but not yet assigned to a component
    let mut stack: Vec<NodeIdx> = Vec::with_capacity(n);
    // lowlink[v.0] is the smallest index reachable from v's DFS subtree
    let mut lowlink: Vec<usize> = vec![0; n];
    // heap-allocated call stack instead of recursion
    let mut call_stack: Vec<WorkItem> = Vec::with_capacity(n);

    let mut sccs: Vec<Vec<NodeIdx>> = Vec::with_capacity(n);

    let mut counter = NonZero::<usize>::MIN;

    for r in 0..n {
        let root = NodeIdx(r);
        if index[root.0].is_some() {
            continue;
        }

        index[root.0] = Some(counter);
        lowlink[root.0] = counter.get();
        stack.push(root);
        counter = counter.saturating_add(1);
        call_stack.push(WorkItem {
            current_vertex: root,
            next_neighbor: 0,
        });

        while let Some(frame) = call_stack.last_mut() {
            let v = frame.current_vertex;

            if let Some(&edge) = nodes[v.0].deps.get(frame.next_neighbor) {
                let w = edge.to;
                frame.next_neighbor += 1;

                if index[w.0].is_none() {
                    index[w.0] = Some(counter);
                    lowlink[w.0] = counter.get();
                    stack.push(w);
                    counter = counter.saturating_add(1);
                    call_stack.push(WorkItem {
                        current_vertex: w,
                        next_neighbor: 0,
                    });
                } else {
                    // propagate index[w.0] to the current vertex
                    lowlink[v.0] = lowlink[v.0].min(index[w.0].unwrap().get());
                }
            } else {
                call_stack.pop();

                if lowlink[v.0] == index[v.0].unwrap().get() {
                    let mut component: Vec<NodeIdx> = Vec::new();
                    loop {
                        let w = stack.pop().expect("tarjan: stack underflow");
                        index[w.0] = Some(DONE);
                        component.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(component);
                }

                if let Some(parent) = call_stack.last() {
                    let p = parent.current_vertex;
                    // propagate lowlink[v.0] to its parent
                    lowlink[p.0] = lowlink[p.0].min(lowlink[v.0]);
                }
            }
        }
    }
    sccs
}

/// Exactly [`find_sccs_of_nodes`], but only follows edges for which `keep` returns true.
pub fn find_sccs_of_nodes_filtered(
    nodes: &[DepNode],
    keep: impl Fn(&Edge) -> bool,
) -> Vec<Vec<NodeIdx>> {
    const DONE: NonZero<usize> = NonZero::<usize>::MAX;

    let n = nodes.len();
    let mut index: Vec<Option<NonZero<usize>>> = vec![None; n];
    let mut stack: Vec<NodeIdx> = Vec::with_capacity(n);
    let mut lowlink: Vec<usize> = vec![0; n];
    let mut call_stack: Vec<WorkItem> = Vec::with_capacity(n);

    let mut sccs: Vec<Vec<NodeIdx>> = Vec::with_capacity(n);

    let mut counter = NonZero::<usize>::MIN;

    for r in 0..n {
        let root = NodeIdx(r);
        if index[root.0].is_some() {
            continue;
        }

        index[root.0] = Some(counter);
        lowlink[root.0] = counter.get();
        stack.push(root);
        counter = counter.saturating_add(1);
        call_stack.push(WorkItem {
            current_vertex: root,
            next_neighbor: 0,
        });

        while let Some(frame) = call_stack.last_mut() {
            let v = frame.current_vertex;

            let mut next = None;
            while let Some(e) = nodes[v.0].deps.get(frame.next_neighbor) {
                frame.next_neighbor += 1;
                if keep(e) {
                    next = Some(e.to);
                    break;
                }
            }

            if let Some(w) = next {
                if index[w.0].is_none() {
                    index[w.0] = Some(counter);
                    lowlink[w.0] = counter.get();
                    stack.push(w);
                    counter = counter.saturating_add(1);
                    call_stack.push(WorkItem {
                        current_vertex: w,
                        next_neighbor: 0,
                    });
                } else {
                    lowlink[v.0] = lowlink[v.0].min(index[w.0].unwrap().get());
                }
            } else {
                call_stack.pop();

                if lowlink[v.0] == index[v.0].unwrap().get() {
                    let mut component: Vec<NodeIdx> = Vec::new();
                    loop {
                        let w = stack.pop().expect("tarjan: stack underflow");
                        index[w.0] = Some(DONE);
                        component.push(w);
                        if w == v {
                            break;
                        }
                    }
                    sccs.push(component);
                }

                if let Some(parent) = call_stack.last() {
                    let p = parent.current_vertex;
                    lowlink[p.0] = lowlink[p.0].min(lowlink[v.0]);
                }
            }
        }
    }
    sccs
}

#[cfg(test)]
mod tests {
    use ir::span::SourceSpan;

    use super::*;
    use crate::nodes::{EdgeKind, ModuleIdx, NodeKind};

    fn graph<'a>(n: usize, edges: &[(usize, usize)]) -> Vec<DepNode<'a>> {
        let mut nodes: Vec<DepNode> = (0..n)
            .map(|i| {
                DepNode::new(
                    NodeIdx(i),
                    NodeKind::Module {
                        module_idx: ModuleIdx(usize::MAX),
                    },
                    None,
                )
            })
            .collect();
        for &(from, to) in edges {
            let (from, to) = (NodeIdx(from), NodeIdx(to));
            nodes[from].deps.push(Edge {
                to,
                kind: EdgeKind::Use,
                span: SourceSpan::default(),
            });
            nodes[to].rev_deps.push(from);
        }
        nodes
    }

    fn ids(xs: &[usize]) -> Vec<NodeIdx> {
        xs.iter().copied().map(NodeIdx).collect()
    }

    fn groups(xs: &[&[usize]]) -> Vec<Vec<NodeIdx>> {
        xs.iter().map(|x| ids(x)).collect()
    }

    fn normalized(mut sccs: Vec<Vec<NodeIdx>>) -> Vec<Vec<NodeIdx>> {
        for c in &mut sccs {
            c.sort_unstable();
        }
        sccs.sort();
        sccs
    }

    fn assert_partition(n: usize, sccs: &[Vec<NodeIdx>]) {
        let mut seen = vec![false; n];
        for &v in sccs.iter().flatten() {
            assert!(!seen[v.0], "node {v} appears in more than one SCC");
            seen[v.0] = true;
        }
        assert!(
            seen.iter().all(|&s| s),
            "some node is missing from the SCCs"
        );
    }

    fn assert_deps_first(nodes: &[DepNode], sccs: &[Vec<NodeIdx>]) {
        let mut comp_of = vec![usize::MAX; nodes.len()];
        for (i, c) in sccs.iter().enumerate() {
            for &v in c {
                comp_of[v.0] = i;
            }
        }
        for node in nodes {
            for &dep in &node.deps {
                assert!(
                    comp_of[dep.to.0] <= comp_of[node.idx.0],
                    "edge {} -> {} is emitted in the wrong order",
                    node.idx,
                    dep.to
                );
            }
        }
    }

    #[test]
    fn working_tarjan_scc_algorithm() {
        let nodes = graph(6, &[(0, 1), (1, 2), (2, 0), (2, 3), (3, 4), (4, 3), (4, 5)]);
        let sccs = find_sccs_of_nodes(&nodes);

        assert_eq!(
            normalized(sccs.clone()),
            groups(&[&[0, 1, 2], &[3, 4], &[5]])
        );
        assert_partition(6, &sccs);
        assert_deps_first(&nodes, &sccs);

        assert_eq!(sccs.len(), 3);
        assert_eq!(sccs[0], ids(&[5]));
        assert_eq!(normalized(vec![sccs[1].clone()]), groups(&[&[3, 4]]));
        assert_eq!(normalized(vec![sccs[2].clone()]), groups(&[&[0, 1, 2]]));
    }

    #[test]
    fn empty_graph() {
        assert!(find_sccs_of_nodes(&graph(0, &[])).is_empty());
    }

    #[test]
    fn single_node_no_edges() {
        assert_eq!(find_sccs_of_nodes(&graph(1, &[])), groups(&[&[0]]));
    }

    #[test]
    fn self_loop_is_a_single_component() {
        assert_eq!(find_sccs_of_nodes(&graph(1, &[(0, 0)])), groups(&[&[0]]));
    }

    #[test]
    fn two_cycle() {
        let sccs = find_sccs_of_nodes(&graph(2, &[(0, 1), (1, 0)]));
        assert_eq!(normalized(sccs), groups(&[&[0, 1]]));
    }

    #[test]
    fn simple_cycle_of_five() {
        let nodes = graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(normalized(sccs), groups(&[&[0, 1, 2, 3, 4]]));
    }

    #[test]
    fn chain_dag_emits_dependencies_first() {
        let nodes = graph(4, &[(0, 1), (1, 2), (2, 3)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(sccs, groups(&[&[3], &[2], &[1], &[0]]));
    }

    #[test]
    fn diamond_dag() {
        //   0
        //  / \
        // 1   2
        //  \ /
        //   3
        let nodes = graph(4, &[(0, 1), (0, 2), (1, 3), (2, 3)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(sccs.len(), 4);
        assert_partition(4, &sccs);
        assert_deps_first(&nodes, &sccs);
        assert_eq!(sccs[0], ids(&[3]));
        assert_eq!(sccs[3], ids(&[0]));
    }

    #[test]
    fn disconnected_components() {
        let nodes = graph(6, &[(0, 1), (1, 0), (3, 4), (4, 5), (5, 3)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(normalized(sccs), groups(&[&[0, 1], &[2], &[3, 4, 5]]));
    }

    #[test]
    fn cross_edge_into_finished_scc_is_ignored() {
        let nodes = graph(3, &[(0, 1), (0, 2), (2, 1)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(sccs, groups(&[&[1], &[2], &[0]]));
    }

    #[test]
    fn edge_into_earlier_root_scc() {
        let nodes = graph(2, &[(1, 0)]);
        assert_eq!(find_sccs_of_nodes(&nodes), groups(&[&[0], &[1]]));
    }

    #[test]
    fn duplicate_edges_are_harmless() {
        let nodes = graph(3, &[(0, 1), (0, 1), (1, 2), (1, 2), (2, 0), (2, 0)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(normalized(sccs), groups(&[&[0, 1, 2]]));
    }

    #[test]
    fn cycle_nested_with_tail() {
        let nodes = graph(3, &[(0, 1), (1, 2), (2, 1)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(sccs.len(), 2);
        assert_eq!(normalized(vec![sccs[0].clone()]), groups(&[&[1, 2]]));
        assert_eq!(sccs[1], ids(&[0]));
    }

    #[test]
    fn filtered_with_keep_all_matches_unfiltered() {
        let nodes = graph(3, &[(0, 1), (0, 2), (2, 0)]);
        let plain = find_sccs_of_nodes(&nodes);
        let filtered = find_sccs_of_nodes_filtered(&nodes, |_| true);
        assert_eq!(normalized(filtered), normalized(plain));
    }

    #[test]
    fn filtered_ignores_rejected_edges() {
        let nodes = graph(2, &[(0, 1), (1, 0)]);
        let sccs = find_sccs_of_nodes_filtered(&nodes, |_| false);
        assert_eq!(normalized(sccs), groups(&[&[0], &[1]]));
    }

    #[test]
    fn deep_chain_does_not_overflow_the_stack() {
        let n = 200_000;
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let nodes = graph(n, &edges);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(sccs.len(), n);
        assert_eq!(sccs[0], ids(&[n - 1]));
        assert_eq!(sccs[n - 1], ids(&[0]));
    }

    #[test]
    fn deep_cycle_does_not_overflow_the_stack() {
        let n = 200_000;
        let mut edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        edges.push((n - 1, 0));
        let nodes = graph(n, &edges);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(sccs.len(), 1);
        assert_eq!(sccs[0].len(), n);
    }
}
