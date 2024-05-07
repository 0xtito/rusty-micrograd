use std::{cell::RefCell, fmt, ops, rc::Rc};
use uuid::Uuid;

#[derive(Debug)]
pub enum Op {
    Add,
    Mul,
}

#[derive(fmt::Debug)]
pub struct ValueInfo {
    pub id: Uuid,
    pub label: String,
    pub value: f64,
    pub grad: f64,
    pub prev: Vec<Value>,
    pub op: Option<Op>,
}

#[derive(Clone)]
pub struct Value(Rc<RefCell<ValueInfo>>);

impl PartialEq for Value {
    fn eq(&self, other: &Self) -> bool {
        self.0.borrow().id == other.0.borrow().id
    }
}

impl Eq for Value {}

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
            self.value, self.prev, self.op
        )
    }
}

impl Value {
    pub fn new(value: f64, label: &str) -> Value {
        Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: label.to_string(),
            grad: 0.0,
            value,
            prev: Vec::new(),
            op: None,
        })))
    }

    pub fn mul(&self, other: &Value, label: &str) -> Value {
        let new = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: label.to_string(),
            grad: 0.0,
            value: self.0.borrow().value * other.0.borrow().value,
            prev: vec![self.clone(), other.clone()],
            op: Some(Op::Mul),
        })));
        new
    }

    pub fn add(&self, other: &Value, label: &str) -> Value {
        let new = Value(Rc::new(RefCell::new(ValueInfo {
            id: Uuid::new_v4(),
            label: label.to_string(),
            grad: 0.0,
            value: self.0.borrow().value + other.0.borrow().value,
            prev: vec![self.clone(), other.clone()],
            op: Some(Op::Add),
        })));
        new
    }

    pub fn borrow(&self) -> std::cell::Ref<'_, ValueInfo> {
        self.0.borrow()
    }
}
