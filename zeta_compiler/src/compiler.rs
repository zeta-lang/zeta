use cranelift::prelude::*;
use cranelift_jit::{JITBuilder, JITModule};
use cranelift_module::{FuncId, Linkage, Module};
use crate::ast::*;

pub struct Codegen<'a> {
    builder_context: FunctionBuilderContext,
    ctx: codegen::Context,
    module: &'a mut JITModule,
    variables: VariableMap,
}

use std::collections::HashMap;
type VariableMap = HashMap<String, Variable>;

impl<'a> Codegen<'a> {
    pub fn new(module: &'a mut JITModule) -> Self {
        Self {
            builder_context: FunctionBuilderContext::new(),
            ctx: module.make_context(),
            module,
            variables: HashMap::new(),
        }
    }

    pub fn compile_func(&mut self, func: &FuncDecl) -> anyhow::Result<FuncId> {
        let sig = self.module.make_signature();
        self.ctx.func.signature = sig;

        {
            let ctx_func = &mut self.ctx.func; // first mutable borrow
            let mut builder = FunctionBuilder::new(ctx_func, &mut self.builder_context); // the cause is here at &mut self.ctx.func
            let block = builder.create_block();
            builder.switch_to_block(block);
            builder.seal_block(block);

            let mutable_builder = &mut builder;
            for stmt in &func.body.block {
                // borrows as mutable more than once at a time here
                self.compile_stmt(mutable_builder, stmt)?;
            }

            builder.ins().return_(&[]);
            builder.finalize();
        }

        let func_id = self.module.declare_function(&func.name, Linkage::Export, &self.ctx.func.signature)?;
        self.module.define_function(func_id, &mut self.ctx)?;
        self.module.clear_context(&mut self.ctx);

        Ok(func_id)
    }

    fn compile_stmt(&mut self, builder: &mut FunctionBuilder<'_>, stmt: &Stmt) -> anyhow::Result<()> {
        match stmt {
            Stmt::ExprStmt(expr_stmt) => {
                let _ = self.compile_expr(builder, &expr_stmt.expr)?;
            }
            Stmt::Let(let_stmt) => {
                let val = self.compile_expr(builder, &let_stmt.value)?;
                let var = Variable::new(self.variables.len());
                builder.declare_var(var, types::I64);
                builder.def_var(var, val);
                self.variables.insert(let_stmt.ident.clone(), var);
            }
            Stmt::Return(ret) => {
                if let Some(expr) = &ret.value {
                    let val = self.compile_expr(builder, expr)?;
                    builder.ins().return_(&[val]);
                } else {
                    builder.ins().return_(&[]);
                }
            }
            _ => {
                // Extend for if, while, match, etc.
                unimplemented!("Stmt not supported yet: {:?}", stmt);
            }
        }

        Ok(())
    }

    fn compile_expr(&mut self, builder: &mut FunctionBuilder<'_>, expr: &Expr) -> anyhow::Result<Value> {
        match expr {
            Expr::Number(n) => Ok(builder.ins().iconst(types::I64, *n)),
            Expr::Ident(name) => {
                if let Some(var) = self.variables.get(name) {
                    Ok(builder.use_var(*var))
                } else {
                    anyhow::bail!("Undefined variable: {}", name)
                }
            }
            Expr::Binary { left, op, right } => {
                let l = self.compile_expr(builder, left)?;
                let r = self.compile_expr(builder, right)?;

                let v = match op {
                    Op::Add => builder.ins().iadd(l, r),
                    Op::Sub => builder.ins().isub(l, r),
                    Op::Mul => builder.ins().imul(l, r),
                    Op::Div => builder.ins().sdiv(l, r),
                    _ => anyhow::bail!("Unsupported binary op: {:?}", op),
                };

                Ok(v)
            }
            _ => anyhow::bail!("Unsupported expression: {:?}", expr),
        }
    }
}
