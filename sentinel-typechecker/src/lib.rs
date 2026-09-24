pub mod borrowck;
pub mod closures;
pub mod control_flow;
pub mod expressions;
pub mod initialization;
pub mod instantiations;
pub mod move_state;
pub mod naming;
pub mod nullables;
pub mod ref_effects;
pub mod ref_provenance;
pub mod references;
pub mod statements;
pub mod type_checker;
pub mod type_context;
pub mod types;

pub use type_checker::TypeChecker;
pub use type_context::TypeContext;

pub use crate::naming::str_id_to_string;
