use crate::engine::*;
use rand::{
    self,
    distributions::{Distribution, Uniform},
};

#[derive(Debug)]
pub struct Neuron(pub Vec<Value>, pub Value, pub bool);

impl Neuron {
    pub fn new(nin: i32, nonlin: bool) -> Self {
        let mut weights = Vec::new();
        let between = Uniform::from(-1.0..1.0);
        let mut rng = rand::thread_rng();
        for _ in 0..nin {
            weights.push(Value::new(between.sample(&mut rng), "weight"));
        }
        // let weight = Value::new(rand::random::<f64>(), "weight");
        let bias = Value::new(between.sample(&mut rng), "bias");

        Neuron(weights, bias, nonlin)
    }

    pub fn from(nin: i32) -> Neuron {
        Neuron::new(nin, true)
    }

    pub fn call(&self, inputs: &Vec<Value>) -> Value {
        let bias = self.1.clone();

        let sum = self
            .0
            .iter()
            .zip(inputs)
            .map(|(wi, xi)| xi.to_owned() * wi.to_owned())
            .sum::<Value>();

        let out = sum + bias;

        if self.2 {
            out._tanh("output")
        } else {
            out
        }
    }

    pub fn parameters(&self) -> Vec<Value> {
        let mut out = self.0.clone();
        out.push(self.1.clone());
        out
    }
}
