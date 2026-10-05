use crate::{
    hir::{HirAttrArg, HirAttribute, StrId},
    ir_hasher::HashMap,
};

#[derive(Clone, Copy, PartialEq, Eq)]
pub struct Targets(u32);
impl Targets {
    pub const FUNC: Targets = Targets(1 << 0);
    pub const STRUCT: Targets = Targets(1 << 1);
    pub const ENUM: Targets = Targets(1 << 2);
    pub const IFACE: Targets = Targets(1 << 3);
    pub const IMPL: Targets = Targets(1 << 4);
    pub const CONST: Targets = Targets(1 << 5);
    pub const LET: Targets = Targets(1 << 6);
    pub const STMT: Targets = Targets(1 << 7);
    pub const EXPR: Targets = Targets(1 << 8);
    pub const ALL: Targets = Targets(!0);
    pub const TYPE_DECL: Targets = Targets((1 << 1) | (1 << 2) | (1 << 3));
    pub const fn or(self, o: Targets) -> Targets {
        Targets(self.0 | o.0)
    }
    pub fn contains(self, t: Targets) -> bool {
        self.0 & t.0 != 0
    }
}

#[derive(Clone, Copy)]
pub enum Arity {
    None,
    Exactly(usize),
    AtMost(usize),
}

pub struct AttrSpec {
    pub name: &'static str,
    pub targets: Targets,
    pub arity: Arity,
    pub repeatable: bool,
    /// Extra argument checks beyond arity. `Err(msg)` becomes a type error.
    pub validate: Option<fn(&HirAttribute<'_, '_>) -> Result<(), String>>,
}

fn validate_known(a: &HirAttribute<'_, '_>) -> Result<(), String> {
    match a.args.first() {
        None | Some(HirAttrArg::Str(_)) => Ok(()),
        Some(_) => {
            Err("`#[known]` expects a string literal, e.g. `#[known(\"string_new\")]`".into())
        }
    }
}

fn validate_packed(a: &HirAttribute<'_, '_>) -> Result<(), String> {
    match a.args.first() {
        None => Ok(()),
        Some(HirAttrArg::Number(n)) if *n > 0 && (*n as u64).is_power_of_two() => Ok(()),
        Some(_) => Err("`#[packed(N)]` requires N to be a positive power of two".into()),
    }
}

pub static ATTRIBUTE_SPECS: &[AttrSpec] = &[
    AttrSpec {
        name: "known",
        targets: Targets::FUNC.or(Targets::TYPE_DECL),
        arity: Arity::AtMost(1),
        repeatable: false,
        validate: Some(validate_known),
    },
    AttrSpec {
        name: "packed",
        targets: Targets::STRUCT,
        arity: Arity::AtMost(1),
        repeatable: false,
        validate: Some(validate_packed),
    },
];

impl AttrTarget {
    /// Which syntactic sites this target can correspond to.
    pub fn site(&self) -> Targets {
        match self {
            AttrTarget::Func(_) => Targets::FUNC,
            AttrTarget::Struct(_) => Targets::STRUCT,
            AttrTarget::Enum(_) => Targets::ENUM,
            AttrTarget::Interface(_) => Targets::IFACE,
            AttrTarget::Node(_) => Targets::STMT
                .or(Targets::EXPR)
                .or(Targets::LET)
                .or(Targets::CONST),
        }
    }
}

pub fn lookup(name: &str) -> Option<&'static AttrSpec> {
    ATTRIBUTE_SPECS.iter().find(|s| s.name == name)
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum AttrTarget {
    Func(StrId),
    Struct(StrId),
    Enum(StrId),
    Interface(StrId),
    /// Statements / expressions: keyed by address of the bump node
    Node(usize),
}

#[derive(Default)]
pub struct AttrTable<'a, 'bump> {
    pub map: HashMap<AttrTarget, Vec<HirAttribute<'a, 'bump>>>,
    pub order: Vec<AttrTarget>, // pushed on first insert in record_attrs
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum KnownKind {
    Func,
    Type,
}
