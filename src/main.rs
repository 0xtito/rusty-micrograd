mod engine;
mod graph;
mod neuron;

use engine::Value;

// use ndarray::prelude::*;
// use ndarray::{arr1, Array};
use plotters::prelude::*;

use petgraph::algo::{dijkstra, min_spanning_tree};
use petgraph::data::FromElements;
use petgraph::dot::{Config, Dot};
use petgraph::graph::{NodeIndex, UnGraph};

use crate::graph::create_graphviz;

fn main() {
    let mut args = std::env::args();

    println!("args: {:?} ", args);

    if args.len() > 1 {
        let arg = args.nth(1).unwrap();
        match arg.as_str() {
            "first" => first_example(),
            "second" => second_example(),
            "impl_ops" => impl_ops_example(),
            "petgraph" => petgraph_example(),
            _ => println!("Invalid argument"),
        }
    } else {
        first_example();
    }
}

fn impl_ops_example() {
    println!("impl_ops_example");
    let x1 = Value::new(2.0, "x1");
    let x2 = Value::new(0.0, "x2");

    let w1 = Value::new(-3.0, "w1");
    let w2 = Value::new(1.0, "w2");

    let b = Value::new(6.8813735870195432, "b");

    let x1w1 = x1 * w1;
    x1w1.set_label("x1*w1");

    let x2w2 = x2 * w2;
    x2w2.set_label("x2*w2");

    let x1w1x2w2 = x1w1 + x2w2;
    x1w1x2w2.set_label("x1w1 + x2w2");

    let n = x1w1x2w2 + b;
    n.set_label("n");

    let o = n._tanh("o");

    o.backward();

    create_graphviz(&o, "./plots/backprop_ops.dot");
}

fn second_example() {
    let mut x1 = Value::new(2.0, "x1");
    let mut x2 = Value::new(0.0, "x2");

    let w1 = Value::new(-3.0, "w1");
    let w2 = Value::new(1.0, "w2");

    let b = Value::new(6.8813735870195432, "b");

    let x1w1 = x1.mul(&w1, "x1*w1");
    let x2w2 = x2.mul(&w2, "x2*w2");

    let x1w1x2w2 = x1w1.add(&x2w2, "x1w1 + x2w2");

    let n = x1w1x2w2.add(&b, "n");

    let o = n._tanh("o");

    o.backward();

    create_graphviz(&o, "./plots/backprop.dot");
}

fn first_example() {
    let mut a = Value::new(2.0, "a");
    let b = Value::new(-3.0, "b");
    let c = Value::new(10.0, "c");

    let d = a.mul(&b, "d");

    let mut e = d.add(&c, "e");

    let f = Value::new(-2.0, "f");

    let g = e.mul(&f, "g");

    let h = g._tanh("h");

    h.backward();

    create_graphviz(&h, "./plots/graph.dot");
}

#[warn(dead_code)]
pub fn test_function(x: i32) -> i32 {
    let x = 3 * x.pow(2) + 4 * x + 5;
    x
}

#[warn(dead_code)]
pub fn plotters_example() -> Result<(), Box<dyn std::error::Error>> {
    let root = BitMapBackend::new("plots/example_1.png", (640, 480)).into_drawing_area();
    let _ = root.fill(&WHITE);
    let root = root.margin(10, 10, 10, 10);
    // After this point, we should be able to construct a chart context
    let mut chart = ChartBuilder::on(&root)
        // Set the caption of the chart
        .caption("Plotters Test 2 ", ("sans-serif", 40).into_font())
        // Set the size of the label region
        .x_label_area_size(20)
        .y_label_area_size(40)
        // Finally attach a coordinate on the drawing area and make a chart context
        .build_cartesian_2d(0f32..10f32, 0f32..10f32)?;

    // Then we can draw a mesh
    chart
        .configure_mesh()
        // We can customize the maximum number of labels allowed for each axis
        .x_labels(5)
        .y_labels(5)
        // We can also change the format of the label text
        .y_label_formatter(&|x| format!("{:.3}", x))
        .draw()?;

    // And we can draw something in the drawing area
    chart.draw_series(LineSeries::new(
        vec![(0.0, 0.0), (5.0, 5.0), (8.0, 7.0)],
        &RED,
    ))?;
    // Similarly, we can draw point series
    chart.draw_series(PointSeries::of_element(
        vec![(0.0, 0.0), (5.0, 5.0), (8.0, 7.0)],
        5,
        &RED,
        &|c, s, st| {
            return EmptyElement::at(c)    // We want to construct a composed element on-the-fly
            + Circle::new((0,0),s,st.filled()) // At this point, the new pixel coordinate is established
            + Text::new(format!("{:?}", c), (10, 0), ("sans-serif", 10).into_font());
        },
    ))?;
    root.present()?;
    Ok(())
}

#[warn(dead_code)]
pub fn petgraph_example() {
    let g = UnGraph::<i32, ()>::from_edges(&[(1, 2), (2, 3), (3, 4), (1, 4)]);

    // Find the shortest path from `1` to `4` using `1` as the cost for every edge.
    let node_map = dijkstra(&g, 1.into(), Some(4.into()), |_| 1);
    assert_eq!(&1i32, node_map.get(&NodeIndex::new(4)).unwrap());

    // Get the minimum spanning tree of the graph as a new graph, and check that
    // one edge was trimmed.
    let mst = UnGraph::<_, _>::from_elements(min_spanning_tree(&g));
    assert_eq!(g.raw_edges().len() - 1, mst.raw_edges().len());

    // Output the tree to `graphviz` `DOT` format
    println!("{:?}", Dot::with_config(&mst, &[Config::EdgeNoLabel]));
}
