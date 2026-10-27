use std::num::NonZero;

use crate::{NodeIdx, nodes::DepNode};

struct WorkItem {
    current_vertex: NodeIdx,
    next_neighbor: usize,
}

/// Find a partitioned directed graph into its strongly connected components using
/// Tarjan's Algorithm for creating this directed graph (Depth-First Search algorithm) but as an iterative implementation.
///
/// # What's a Strongly Connected Component?
///
/// Vertice: A single node (such as `C`)
///
/// Edge: A direct connection (such as `A -> C`)
///
/// Path: A sequence of directed edges to travel from one vertex to another indirectly (such as getting `B` from `A` like: `A -> C -> B`)
///
/// A Strongly Connected Component of a directed graph is a maximal subset of vertices such that every vertex in the subset is
/// reachable from every other vertex in that same subset, following the direction of the edges -- Meaning, for any two vertices (one is `u` and one is `v`),
/// there must be a direct path from `u` to `v`, and another from `u` to `v`
///
///       ┌─────────┐
///       ▼         │
///      (A) ───► (B)
///       ▲         │
///       │         ▼
///       └─────── (C)
///
/// Here, all nodes belong to the same vertex, you can access B and C through A; C and A through B; and A and B through C
///
/// # Applications
/// This is used in the Zeta compiler to:
/// - detect circular imports
/// - detect recursive cycles
/// - detect files in the same SCC
/// - and most importantly: to figure out the order in which files should be properly compiled.
pub fn find_sccs_of_nodes(nodes: &[DepNode]) -> Vec<Vec<NodeIdx>> {
    const DONE: NonZero<usize> = NonZero::<usize>::MAX;

    let n = nodes.len();
    // amount of discoveries of visited vertices
    let mut index: Vec<Option<NonZero<usize>>> = vec![None; n]; // completely unvisited
    // what is still unassigned?
    let mut stack: Vec<NodeIdx> = Vec::with_capacity(n);
    // lowlink[v] is the smallest index that v’s DFS subtree can reach, where v = current vertex
    let mut lowlink: Vec<usize> = vec![0; n];
    // heap-allocated call stack instead of stack-allocated call stack (aka, recursion)
    let mut call_stack: Vec<WorkItem> = Vec::with_capacity(n);

    let mut sccs: Vec<Vec<usize>> = Vec::with_capacity(n);

    let mut counter = NonZero::<usize>::MIN;

    for root in 0..n {
        if index[root].is_some() {
            continue;
        }

        index[root] = Some(counter);
        lowlink[root] = counter.get();
        stack.push(root);
        counter = counter.saturating_add(1);
        call_stack.push(WorkItem {
            current_vertex: root,
            next_neighbor: 0,
        });

        while let Some(frame) = call_stack.last_mut() {
            let v = frame.current_vertex;

            if let Some(&w) = nodes[v].deps.get(frame.next_neighbor) {
                frame.next_neighbor += 1;

                if index[w].is_none() {
                    index[w] = Some(counter);
                    lowlink[w] = counter.get();
                    stack.push(w);
                    counter = counter.saturating_add(1);
                    call_stack.push(WorkItem {
                        current_vertex: w,
                        next_neighbor: 0,
                    });
                } else {
                    // propagate index[w] to the current vertex
                    lowlink[v] = lowlink[v].min(index[w].unwrap().get());
                }
            } else {
                call_stack.pop();

                if lowlink[v] == index[v].unwrap().get() {
                    let mut component: Vec<usize> = Vec::new();
                    loop {
                        let w = stack.pop().expect("tarjan: stack underflow");
                        index[w] = Some(DONE);
                        component.push(w);
                        if w == v {
                            break;
                        }
                    }

                    sccs.push(component);
                }

                if let Some(parent) = call_stack.last() {
                    let p = parent.current_vertex;
                    // propagate index[v] to its parent
                    lowlink[p] = lowlink[p].min(lowlink[v]);
                }
            }
        }
    }
    sccs
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::nodes::NodeKind;

    fn graph(n: usize, edges: &[(usize, usize)]) -> Vec<DepNode> {
        let mut nodes: Vec<DepNode> = (0..n)
            .map(|i| DepNode {
                idx: i,
                kind: NodeKind::Module {
                    module_idx: usize::MAX,
                },
                hint: None,
                deps: Vec::new(),
                rev_deps: Vec::new(),
            })
            .collect();
        for &(from, to) in edges {
            nodes[from].deps.push(to);
            nodes[to].rev_deps.push(from);
        }
        nodes
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
            assert!(!seen[v], "node {v} appears in more than one SCC");
            seen[v] = true;
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
                comp_of[v] = i;
            }
        }
        for node in nodes {
            for &dep in &node.deps {
                assert!(
                    comp_of[dep] <= comp_of[node.idx],
                    "edge {} -> {dep} is emitted in the wrong order",
                    node.idx
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
            vec![vec![0, 1, 2], vec![3, 4], vec![5]]
        );
        assert_partition(6, &sccs);
        assert_deps_first(&nodes, &sccs);

        assert_eq!(sccs.len(), 3);
        assert_eq!(sccs[0], vec![5]);
        assert_eq!(normalized(vec![sccs[1].clone()]), vec![vec![3, 4]]);
        assert_eq!(normalized(vec![sccs[2].clone()]), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn empty_graph() {
        assert!(find_sccs_of_nodes(&graph(0, &[])).is_empty());
    }

    #[test]
    fn single_node_no_edges() {
        assert_eq!(find_sccs_of_nodes(&graph(1, &[])), vec![vec![0]]);
    }

    #[test]
    fn self_loop_is_a_single_component() {
        assert_eq!(find_sccs_of_nodes(&graph(1, &[(0, 0)])), vec![vec![0]]);
    }

    #[test]
    fn two_cycle() {
        let sccs = find_sccs_of_nodes(&graph(2, &[(0, 1), (1, 0)]));
        assert_eq!(normalized(sccs), vec![vec![0, 1]]);
    }

    #[test]
    fn simple_cycle_of_five() {
        let nodes = graph(5, &[(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(normalized(sccs), vec![vec![0, 1, 2, 3, 4]]);
    }

    #[test]
    fn chain_dag_emits_dependencies_first() {
        let nodes = graph(4, &[(0, 1), (1, 2), (2, 3)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(sccs, vec![vec![3], vec![2], vec![1], vec![0]]);
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
        assert_eq!(sccs[0], vec![3]);
        assert_eq!(sccs[3], vec![0]);
    }

    #[test]
    fn disconnected_components() {
        let nodes = graph(6, &[(0, 1), (1, 0), (3, 4), (4, 5), (5, 3)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(normalized(sccs), vec![vec![0, 1], vec![2], vec![3, 4, 5]]);
    }

    #[test]
    fn cross_edge_into_finished_scc_is_ignored() {
        let nodes = graph(3, &[(0, 1), (0, 2), (2, 1)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(sccs, vec![vec![1], vec![2], vec![0]]);
    }

    #[test]
    fn edge_into_earlier_root_scc() {
        let nodes = graph(2, &[(1, 0)]);
        assert_eq!(find_sccs_of_nodes(&nodes), vec![vec![0], vec![1]]);
    }

    #[test]
    fn duplicate_edges_are_harmless() {
        let nodes = graph(3, &[(0, 1), (0, 1), (1, 2), (1, 2), (2, 0), (2, 0)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(normalized(sccs), vec![vec![0, 1, 2]]);
    }

    #[test]
    fn cycle_nested_with_tail() {
        let nodes = graph(3, &[(0, 1), (1, 2), (2, 1)]);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(sccs.len(), 2);
        assert_eq!(normalized(vec![sccs[0].clone()]), vec![vec![1, 2]]);
        assert_eq!(sccs[1], vec![0]);
    }

    #[test]
    fn deep_chain_does_not_overflow_the_stack() {
        let n = 200_000;
        let edges: Vec<(usize, usize)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        let nodes = graph(n, &edges);
        let sccs = find_sccs_of_nodes(&nodes);
        assert_eq!(sccs.len(), n);
        assert_eq!(sccs[0], vec![n - 1]);
        assert_eq!(sccs[n - 1], vec![0]);
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
