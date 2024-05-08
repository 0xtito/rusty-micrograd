use std::{cell::RefCell, collections::HashSet, fmt, ops, rc::Rc};
use uuid::Uuid;

#[derive(Debug)]
pub enum Op {
    Add,
    Mul,
    Tanh,
    Exp,
}

pub struct ValueInfo {
    pub id: Uuid,
    pub label: String,
    pub data: f64,
    pub grad: f64,
    pub prev: Vec<Value>,
    pub _backward: Option<Box<fn(value: &ValueInfo)>>,
    pub op: Option<Op>,
}

#[derive(Clone)]
pub struct Value(pub Rc<RefCell<ValueInfo>>);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().id == other.0.borrow().id
    }
}

impl Eq for Value {}

impl fmt::Debug for ValueInfo {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        f.debug_struct("ValueInfo")
            .field("id", &self.id)
            .field("label", &self.label)
            .field("value", &self.data)
            .field("grad", &self.grad)
            .field("prev", &self.prev)
            .field("op", &self.op)
            .finish() // `_backward` is not included
    }
}

impl ops::Deref for Value {
    type Target = Rc<RefCell<ValueInfo>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl fmt::Debug for Value {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Value: {}", self.0.borrow())
    }
}

impl fmt::Display for ValueInfo {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "(value: {}, prev: {:?}, op: {:?})",
            self.data, self.prev, self.op
        )
    }
}

impl ops::Add for Value {
    type Output = Value;

    fn add(self, other: Value) -> Value {
        let new_value = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: "".to_string(),
            grad: 0.0,
            data: self.0.borrow().data + other.0.borrow().data,
            prev: vec![self.clone(), other.clone()],
            _backward: None,
            op: Some(Op::Add),
        })));

        new_value.borrow_mut()._backward = Some(Box::new(|value: &ValueInfo| {
            value.prev[0].borrow_mut().grad += 1.0 * value.grad;
            value.prev[1].borrow_mut().grad += 1.0 * value.grad;
        }));

        new_value
    }
}

impl ops::Neg for Value {
    type Output = Value;

    fn neg(self) -> Value {
        let b_self = self.0.borrow();
        return Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: b_self.label.clone(),
            grad: b_self.grad,
            data: -b_self.data,
            prev: vec![self.clone()],
            _backward: None,
            op: None,
        })));
    }
}

impl ops::Sub for Value {
    type Output = Value;

    fn sub(self, other: Value) -> Value {
        return self + (-other);
    }
}

impl ops::Mul for Value {
    type Output = Value;

    fn mul(self, other: Value) -> Value {
        let new_value = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: "".to_string(),
            grad: 0.0,
            data: self.0.borrow().data * other.0.borrow().data,
            prev: vec![self.clone(), other.clone()],
            _backward: None,
            op: Some(Op::Mul),
        })));

        new_value.borrow_mut()._backward = Some(Box::new(|value: &ValueInfo| {
            let data_1 = value.prev[0].borrow().data;
            let data_2 = value.prev[1].borrow().data;

            value.prev[0].borrow_mut().grad += data_2 * value.grad;
            value.prev[1].borrow_mut().grad += data_1 * value.grad;
        }));

        new_value
    }
}

impl ops::Div for Value {
    type Output = Value;

    fn div(self, other: Value) -> Value {
        let new_value = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: "".to_string(),
            grad: 0.0,
            data: self.0.borrow().data * other.0.borrow().data.powi(-1),
            prev: vec![self.clone(), other.clone()],
            _backward: None,
            op: None,
        })));

        new_value.borrow_mut()._backward = Some(Box::new(|value: &ValueInfo| {
            let data_1 = value.prev[0].borrow().data;
            let data_2 = value.prev[1].borrow().data;

            value.prev[0].borrow_mut().grad += data_2 * value.grad;
            value.prev[1].borrow_mut().grad += data_1 * value.grad;
        }));

        new_value
    }
}

impl Value {
    pub fn new(value: f64, label: &str) -> Value {
        Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: label.to_string(),
            grad: 0.0,
            data: value,
            prev: Vec::new(),
            _backward: None,
            op: None,
        })))
    }

    pub fn set_label(&self, label: &str) {
        self.0.borrow_mut().label = label.to_string();
    }

    pub fn add(&self, other: &Value, label: &str) -> Value {
        let new_value = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: label.to_string(),
            grad: 0.0,
            data: self.0.borrow().data + other.0.borrow().data,
            prev: vec![self.clone(), other.clone()],
            _backward: None,
            op: Some(Op::Add),
        })));

        new_value.borrow_mut()._backward = Some(Box::new(|value: &ValueInfo| {
            value.prev[0].borrow_mut().grad += 1.0 * value.grad;
            value.prev[1].borrow_mut().grad += 1.0 * value.grad;
        }));

        new_value
    }

    pub fn mul(&mut self, other: &Value, label: &str) -> Value {
        let new_value = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: label.to_string(),
            grad: 0.0,
            data: self.0.borrow().data * other.0.borrow().data,
            prev: vec![self.clone(), other.clone()],
            _backward: None,
            op: Some(Op::Mul),
        })));

        new_value.borrow_mut()._backward = Some(Box::new(|value: &ValueInfo| {
            let data_1 = value.prev[0].borrow().data;
            let data_2 = value.prev[1].borrow().data;

            value.prev[0].borrow_mut().grad += data_2 * value.grad;
            value.prev[1].borrow_mut().grad += data_1 * value.grad;
        }));

        new_value
    }

    pub fn div(&self, other: &Value, label: &str) -> Value {
        let new_value = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: label.to_string(),
            grad: 0.0,
            data: self.0.borrow().data * other.0.borrow().data.powi(-1),
            prev: vec![self.clone(), other.clone()],
            _backward: None,
            op: None,
        })));

        new_value.borrow_mut()._backward = Some(Box::new(|value: &ValueInfo| {
            let data_1 = value.prev[0].borrow().data;
            let data_2 = value.prev[1].borrow().data;

            value.prev[0].borrow_mut().grad += data_2 * value.grad;
            value.prev[1].borrow_mut().grad += data_1 * value.grad;
        }));

        new_value
    }

    pub fn _pow(&self, other: &Value, label: &str) -> Value {
        let new_value = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: label.to_string(),
            grad: 0.0,
            data: self.0.borrow().data.powf(other.0.borrow().data),
            prev: vec![self.clone(), other.clone()],
            _backward: None,
            op: None,
        })));

        new_value.borrow_mut()._backward = Some(Box::new(|value: &ValueInfo| {
            let data_1 = value.prev[0].borrow().data;
            let data_2 = value.prev[1].borrow().data;

            value.prev[0].borrow_mut().grad += data_2 * data_1.powf(data_2 - 1.0) * value.grad;
        }));

        new_value
    }

    pub fn _exp(&self, label: &str) -> Value {
        let new_value = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: label.to_string(),
            grad: 0.0,
            data: self.0.borrow().data.exp(),
            prev: vec![self.clone()],
            _backward: None,
            op: Some(Op::Exp),
        })));

        new_value.borrow_mut()._backward = Some(Box::new(|value: &ValueInfo| {
            value.prev[0].borrow_mut().grad += value.data * value.grad;
        }));

        new_value
    }

    pub fn _tanh(&self, label: &str) -> Value {
        let new_value = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: label.to_string(),
            grad: 0.0,
            data: self.0.borrow().data.tanh(),
            prev: vec![self.clone()],
            _backward: None,
            op: Some(Op::Tanh),
        })));

        new_value.borrow_mut()._backward = Some(Box::new(|value: &ValueInfo| {
            value.prev[0].borrow_mut().grad += (1.0 - value.data.powi(2)) * value.grad;
        }));

        new_value
    }

    pub fn backward(&self) {
        let mut stack = Vec::<Value>::new();
        let mut visited = HashSet::<Uuid>::new();

        fn build_topo(value: &Value, stack: &mut Vec<Value>, visited: &mut HashSet<Uuid>) {
            if !visited.contains(&value.borrow().id) {
                visited.insert(value.borrow().id);

                for prev in value.borrow().prev.iter() {
                    build_topo(prev, stack, visited)
                }
                stack.push(value.clone());
            } else {
                return;
            }
        }

        build_topo(self, &mut stack, &mut visited);

        stack.reverse();

        self.0.borrow_mut().grad = 1.0;

        for stack_node in stack.iter() {
            let node = stack_node.0.borrow();

            if let Some(backward) = &node._backward {
                backward(&node);
            }
        }
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, ValueInfo> {
        self.0.borrow()
    }
}
