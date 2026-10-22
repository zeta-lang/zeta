pub mod auto_traits;
pub mod bitset;
pub mod borrow_lifetime;
pub mod borrowck;
pub mod closures;
pub mod concurrency;
pub mod control_flow;
pub mod dataflow;
pub mod ensures;
pub mod expressions;
pub mod initialization;
pub mod initialization_bitset;
pub mod instantiations;
pub mod local_table;
pub mod move_state;
pub mod naming;
pub mod nullables;
pub mod ref_effects;
pub mod ref_provenance;
pub mod references;
pub mod statements;
pub mod statics;
pub mod type_checker;
pub mod type_context;
pub mod types;

pub use type_checker::TypeChecker;
pub use type_context::{
    AsStrId, Binding, Globals, GlobalsBuilder, Scope, TypeContext, TypeMethodTable,
};
