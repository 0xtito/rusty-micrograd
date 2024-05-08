use crate::{engine::*, neuron::Neuron};

pub struct Layer(pub Vec<Neuron>);

impl Layer {
    pub fn new(nin: i32, nout: i32) -> Self {
        let mut neurons = Vec::new();
        for _ in 0..nout {
            neurons.push(Neuron::new(nin, true));
        }
        Layer(neurons)
    }

    pub fn call(&self, inputs: &Vec<Value>) -> Vec<Value> {
        self.0.iter().map(|neuron| neuron.call(inputs)).collect()
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.0
            .iter()
            .flat_map(|neuron| neuron.parameters())
            .collect()
    }
}

pub struct MLP(pub Vec<Layer>);

impl MLP {
    pub fn new(nin: i32, nouts: Vec<i32>) -> Self {
        let mut layers = Vec::new();
        let mut prev_nout = nin;
        for nout in nouts {
            layers.push(Layer::new(prev_nout, nout));
            prev_nout = nout;
        }
        MLP(layers)
    }

    pub fn call(&self, inputs: &Vec<Value>) -> Vec<Value> {
        let mut outputs = inputs.clone();
        for layer in self.0.iter() {
            outputs = layer.call(&outputs);
        }
        outputs
    }

    pub fn parameters(&self) -> Vec<Value> {
        self.0.iter().flat_map(|layer| layer.parameters()).collect()
    }
}
