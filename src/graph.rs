use petgraph::{
    dot::Dot,
    prelude::{DiGraph, NodeIndex},
};
use std::collections::HashMap;
use uuid::Uuid;

use crate::engine::*;

#[allow(dead_code)]
type Graph = DiGraph<String, String>;
#[allow(dead_code)]
type NodeHashMap = HashMap<Uuid, (NodeIndex, Value)>;

#[allow(dead_code)]
fn recursive_build(
    graph: &mut Graph,
    node_hash_map: &mut NodeHashMap,
    // op_hash_set: &mut OpHashSet,
    node: &Value,
) -> NodeIndex {
    if let Some(ref node_props) = node_hash_map.get(&node.borrow().id) {
        return node_props.0;
    }

    let node_index = graph.add_node(format!(
        " {} | data: {:.4} | grad: {:.4}",
        node.borrow().label,
        node.borrow().data,
        node.borrow().grad
    ));

    node_hash_map.insert(node.borrow().id, (node_index, node.clone()));

    for child in node.borrow().prev.iter() {
        let child_index = recursive_build(graph, node_hash_map, child);

        let op = match child.borrow().op {
            Some(Op::Add) => "+",
            Some(Op::Mul) => "*",
            Some(Op::Tanh) => "tanh",
            Some(Op::Exp) => "exp",
            None => "",
        };

        graph.add_edge(child_index, node_index, op.to_string());
    }
    node_index
}

#[allow(dead_code)]
fn build_graph(root_node: &Value) -> Graph {
    let mut graph = DiGraph::<String, String>::new();
    let mut node_hash_map = HashMap::<Uuid, (NodeIndex, Value)>::new();

    let mut op_hash_map = HashMap::<NodeIndex, NodeIndex>::default();

    recursive_build(&mut graph, &mut node_hash_map, root_node);

    let mut cloned_node_hash_map = node_hash_map.clone();

    cloned_node_hash_map
        .iter_mut()
        .for_each(|(_id, node_index)| {
            // let node = graph.node_weight(node_index.0).unwrap();
            let node = node_index.1.borrow();

            if let Some(op) = &node.op {
                let op_str = match op {
                    Op::Add => "+",
                    Op::Mul => "*",
                    Op::Tanh => "tanh",
                    Op::Exp => "exp",
                };

                let op_index = graph.add_node(op_str.to_string());

                node_hash_map.insert(Uuid::new_v4(), (op_index, Value::new(0.0, op_str)));

                graph.add_edge(op_index, node_index.0, op_str.to_string());

                op_hash_map.insert(node_index.0, op_index);
            }
        });

    graph
        .edge_indices()
        .step_by(2)
        .zip(graph.edge_indices().skip(1).step_by(2))
        .for_each(|(first, second)| {
            // Now you have pairs of consecutive edge indices
            let edge1 = graph.edge_endpoints(first).unwrap();
            let edge2 = graph.edge_endpoints(second).unwrap();

            let (_source1, target1) = (edge1.clone().0.index(), edge1.clone().1.index());
            let (_source2, target2) = (edge2.clone().0.index(), edge2.clone().1.index());

            if target1 == target2 {
                let op_index = op_hash_map.get(&edge1.1).unwrap();

                graph.update_edge(edge1.0, *op_index, " ".to_string());
                graph.update_edge(edge2.0, *op_index, " ".to_string());

                graph.remove_edge(first);
                graph.remove_edge(second);

                graph.update_edge(*op_index, edge1.1, " ".to_string());
            }
        });

    graph
}

#[allow(dead_code)]
pub fn create_graphviz(root_node: &Value, file_path: &str) {
    let graph = build_graph(root_node);

    let mut dot = format!("{:?}", Dot::new(&graph));

    // Hacky way to adjust graphviz output
    dot = dot.replace("\\\"", "");
    dot.insert_str(10, "    rankdir=\"LR\"");
    dot.insert_str(10, "    node [shape=box]\n");

    std::fs::write(file_path, dot.as_bytes()).unwrap();
}
