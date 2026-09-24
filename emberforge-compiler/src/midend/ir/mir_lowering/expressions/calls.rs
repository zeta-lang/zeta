use ir::{
    hir::{HirExpr, StrId},
    span::SourceSpan,
    ssa_ir::{Instruction, Operand, SsaType, Value},
};
use smallvec::SmallVec;

use crate::midend::ir::mir_lowering::FunctionLowerer;

impl<'f, 'a, 'bump> FunctionLowerer<'f, 'a, 'bump> {
    pub(crate) fn lower_call_args(
        &mut self,
        args: &[HirExpr<'a, 'bump>],
        param_types: &[SsaType],
        param_offset: usize,
    ) -> SmallVec<Operand, 8> {
        let mut ops: SmallVec<Operand, 8> = SmallVec::new();
        for (i, a) in args.iter().enumerate() {
            let pty = param_types.get(i + param_offset).cloned();
            if pty.as_ref().map_or(false, |t| Self::is_move_by_value(t)) {
                self.record_arg_move(a);
            }
            let v = match &pty {
                Some(t) => self.lower_arg_expected(a, t),
                None => self.lower_expr(a),
            };
            ops.push(Operand::Value(v));
        }
        ops
    }

    pub(crate) fn lower_arg_expected(
        &mut self,
        arg: &HirExpr<'a, 'bump>,
        param_ty: &SsaType,
    ) -> Value {
        match param_ty {
            SsaType::Nullable(_) => self.lower_nullable_arg(arg, param_ty),
            _ => self.lower_expr_expected(arg, param_ty),
        }
    }

    pub(crate) fn lower_nullable_arg(
        &mut self,
        arg: &HirExpr<'a, 'bump>,
        param_ty: &SsaType,
    ) -> Value {
        let SsaType::Nullable(inner) = param_ty else {
            unreachable!()
        };

        if matches!(arg, HirExpr::Null(_)) {
            return self.lower_null_ssa(param_ty);
        }

        // Literals/arithmetic take the payload type, not the nullable wrapper.
        let v = match arg {
            HirExpr::Number(..) | HirExpr::Binary { .. } => self.lower_expr_expected(arg, inner),
            _ => self.lower_expr(arg),
        };

        match self.current_block_data.value_types.get(&v).cloned() {
            Some(SsaType::Null) => self.lower_null_ssa(param_ty),
            Some(SsaType::Nullable(_)) => v,
            _ => self.wrap_into_nullable(v, param_ty),
        }
    }

    pub(crate) fn lower_call_expr(
        &mut self,
        callee: &HirExpr<'a, 'bump>,
        args: &[HirExpr<'a, 'bump>],
    ) -> Value {
        if let Some(mangled) = self.try_flatten_module_path(callee) {
            let param_types = self.param_types_of(&mangled);
            let arg_ops = self.lower_call_args(args, &param_types, 0);

            let dest = self.current_block_data.fresh_value();
            self.emit(Instruction::Call {
                dest: Some(dest),
                func: Operand::FunctionRef(mangled),
                args: arg_ops,
            });

            let ret_ty = self
                .funcs
                .get(&mangled)
                .or_else(|| self.global_funcs.get(&mangled))
                .map(|f| f.ret_type.clone())
                .unwrap_or_else(|| {
                    panic!(
                        "lower_call: unknown non-extern function `{:?}`, not in funcs table or global_funcs",
                        mangled
                    )
                });
            self.current_block_data.value_types.insert(dest, ret_ty);
            return dest;
        }

        match callee {
            HirExpr::Ident(fname, ident_span) => {
                if let Some(&ptr_val) = self.var_map.get(fname) {
                    let ptr_ty = self.current_block_data.value_types.get(&ptr_val).cloned();

                    // Closure call: the local's type is a hoisted `__closure_env_N` struct,
                    // whose call fn is registered as its `__call` method.
                    let call_name = StrId(self.context.intern("__call"));
                    let is_closure = ptr_ty
                        .as_ref()
                        .and_then(|ty| match ty {
                            SsaType::User(name, _) => Some(*name),
                            other => self.resolve_receiver_target_key(other),
                        })
                        .and_then(|key| self.struct_mangled_map.get(&key))
                        .is_some_and(|m| m.contains_key(&call_name));
                    if is_closure {
                        return self.lower_method_call_expr(callee, call_name, args, ident_span);
                    }

                    let (param_types, ret_ty) = match ptr_ty.as_ref() {
                        Some(SsaType::FuncPointer {
                            params,
                            return_type,
                        }) => (params.clone(), (**return_type).clone()),
                        other => panic!(
                            "lower_call: `{}` is called but its value type isn't a function pointer: {:?}",
                            fname, other
                        ),
                    };

                    let arg_ops = self.lower_call_args(args, &param_types, 0);

                    let dest = self.current_block_data.fresh_value();
                    self.emit(Instruction::Call {
                        dest: Some(dest),
                        func: Operand::Value(ptr_val),
                        args: arg_ops,
                    });
                    self.current_block_data.value_types.insert(dest, ret_ty);
                    return dest;
                }

                let param_types = self.param_types_of(fname);
                let arg_ops = self.lower_call_args(args, &param_types, 0);

                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Call {
                    dest: Some(dest),
                    func: Operand::FunctionRef(fname.clone()),
                    args: arg_ops,
                });

                let ret_ty = self
                    .funcs
                    .get(fname)
                    .or_else(|| self.global_funcs.get(fname))
                    .map(|f| f.ret_type.clone())
                    .unwrap_or_else(|| {
                        panic!(
                            "lower_call: unknown function `{:?}`, not in funcs table",
                            fname
                        )
                    });
                self.current_block_data.value_types.insert(dest, ret_ty);
                dest
            }

            HirExpr::FieldAccess {
                object,
                field,
                span,
            }
            | HirExpr::Get {
                object,
                field,
                span,
            } => self.lower_method_call_expr(object, *field, args, span),

            HirExpr::ModuleAccess(acc) if acc.path.len() == 1 => {
                self.lower_static_call(acc.path[0], acc.member, args)
            }

            other => unimplemented!(
                "Non-identifier callee not yet supported in Call: {:?}",
                other
            ),
        }
    }

    pub(crate) fn lower_method_call_expr(
        &mut self,
        object: &HirExpr<'a, 'bump>,
        field: StrId,
        args: &[HirExpr<'a, 'bump>],
        span: &SourceSpan<'a>,
    ) -> Value {
        if let HirExpr::Ident(scope_name, _) = object {
            if !self.var_map.contains_key(scope_name) {
                let mangled_struct_name =
                    self.resolve_static_receiver_struct_name(*scope_name, field);

                let direct_name: Option<StrId> = mangled_struct_name
                    .and_then(|cls| self.struct_mangled_map.get(&cls))
                    .and_then(|mmap| mmap.get(&field))
                    .copied()
                    .or_else(|| self.resolve_module_access_callee(&[*scope_name], field));

                let Some(direct_name) = direct_name else {
                    panic!(
                        "lower_method_call (static): could not resolve `{}.{}` to a mangled \
                         function via struct_mangled_map. resolved struct key: {:?} at span {}",
                        scope_name, field, mangled_struct_name, span,
                    );
                };

                let param_types: Vec<SsaType> = self
                    .funcs
                    .get(&direct_name)
                    .map(|f| f.params.iter().map(|(_, ty)| ty.clone()).collect())
                    .unwrap_or_default();

                let mut operands: SmallVec<Operand, 8> = SmallVec::new();
                for (i, a) in args.iter().enumerate() {
                    if Self::is_move_by_value(param_types.get(i).unwrap()) {
                        self.record_arg_move(a);
                    }
                    operands.push(Operand::Value(self.lower_expr(a)));
                }

                let dest: Value = self.current_block_data.fresh_value();
                self.emit(Instruction::Call {
                    dest: Some(dest),
                    func: Operand::FunctionRef(direct_name),
                    args: operands,
                });

                let ret_ty = self
                    .funcs
                    .get(&direct_name)
                    .or_else(|| self.global_funcs.get(&direct_name))
                    .map(|f| f.ret_type.clone())
                    .unwrap_or_else(|| {
                        unreachable!(
                            "lower_method_call (static): resolved `{}` via struct_mangled_map but \
                             it isn't in funcs or global_funcs; registry inconsistency",
                            self.context.resolve_string(&direct_name)
                        )
                    });
                self.current_block_data.value_types.insert(dest, ret_ty);
                return dest;
            }
        }

        let obj_val: Value = self.lower_expr_as_receiver(object);
        let mut operands: SmallVec<Operand, 8> = SmallVec::new();

        let maybe_cls_name_ssa: Option<SsaType> =
            self.current_block_data.value_types.get(&obj_val).cloned();

        if let Some(prim) = self.slice_primitive_of(field) {
            if maybe_cls_name_ssa
                .as_ref()
                .and_then(Self::classify_indexed_container)
                .is_some()
            {
                return self.lower_slice_primitive(prim, object, obj_val, args);
            }
        }

        let cls_name_id: Option<StrId> = maybe_cls_name_ssa
            .as_ref()
            .and_then(|ty| self.resolve_receiver_target_key(ty));

        let param_types: Vec<SsaType> = cls_name_id
            .and_then(|cls| self.struct_mangled_map.get(&cls))
            .and_then(|mmap| mmap.get(&field))
            .and_then(|mangled| self.funcs.get(mangled))
            .map(|f| f.params.iter().map(|(_, ty)| ty.clone()).collect())
            .unwrap_or_default();

        if let Some(_) = cls_name_id {
            let receiver_is_moved = param_types
                .first()
                .map(|ty| matches!(ty, SsaType::User(_, _)))
                .unwrap_or(false);
            if receiver_is_moved {
                self.record_arg_move(object);
            }
        }

        operands.push(Operand::Value(obj_val));
        for (i, a) in args.iter().enumerate() {
            if Self::is_move_by_value(param_types.get(i + 1).unwrap_or(&SsaType::I64)) {
                self.record_arg_move(a);
            }
            let av = self.lower_expr(a);
            operands.push(Operand::Value(av));
        }

        if let Some(value) = self.emit_call_expr(field, obj_val, &mut operands, cls_name_id) {
            return value;
        }

        panic!(
            "[lower_method_call] no mangled mapping or vtable slot found for method `{}` on struct `{:?}` at span {}.",
            self.context.resolve_string(&field),
            cls_name_id.map(|id| self.context.resolve_string(&id).to_string()),
            span
        );
    }

    pub(crate) fn lower_static_call(
        &mut self,
        type_name: StrId,
        method: StrId,
        args: &[HirExpr<'a, 'bump>],
    ) -> Value {
        let struct_key = self
            .resolve_static_receiver_struct_name(type_name, method)
            .or_else(|| {
                let type_module_idx = self
                    .module_named_imports
                    .get(&self.module_idx)
                    .and_then(|named| named.get(&type_name))
                    .copied()
                    .unwrap_or(self.module_idx);

                let mangled = self.dep_graph.borrow().mangle_type_name(
                    type_module_idx,
                    type_name,
                    &self.context,
                );

                if self.struct_mangled_map.contains_key(&mangled) {
                    Some(mangled)
                } else {
                    self.struct_mangled_map
                        .keys()
                        .find(|k| **k == type_name)
                        .copied()
                }
            })
            .unwrap_or_else(|| {
                panic!(
                    "[lower_static_call] could not resolve type {} for static call {}",
                    type_name, method,
                )
            });

        let direct_name = *self
            .struct_mangled_map
            .get(&struct_key)
            .and_then(|mmap| mmap.get(&method))
            .unwrap_or_else(|| {
                panic!(
                    "[lower_static_call] struct `{}` has no static method `{}` in struct_mangled_map",
                    struct_key, method,
                )
            });

        let param_types: Vec<SsaType> = self
            .funcs
            .get(&direct_name)
            .map(|f| f.params.iter().map(|(_, ty)| ty.clone()).collect())
            .unwrap_or_default();

        let arg_ops: SmallVec<Operand, 8> = args
            .iter()
            .enumerate()
            .map(|(i, a)| {
                if Self::is_move_by_value(param_types.get(i).unwrap()) {
                    self.record_arg_move(a);
                }
                Operand::Value(self.lower_expr(a))
            })
            .collect();

        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::Call {
            dest: Some(dest),
            func: Operand::FunctionRef(direct_name),
            args: arg_ops,
        });

        let ret_ty = self
            .funcs
            .get(&direct_name)
            .or_else(|| self.global_funcs.get(&direct_name))
            .map(|f| f.ret_type.clone())
            .unwrap_or_else(|| {
                panic!(
                    "[lower_static_call] resolved `{}` but it isn't in funcs or global_funcs",
                    direct_name
                )
            });
        self.current_block_data.value_types.insert(dest, ret_ty);
        dest
    }

    pub(crate) fn emit_call_expr(
        &mut self,
        field: StrId,
        obj_val: Value,
        operands: &mut SmallVec<Operand, 8>,
        maybe_cls_name: Option<StrId>,
    ) -> Option<Value> {
        let Some(cls_name) = maybe_cls_name else {
            return None;
        };

        let mmap = self.struct_mangled_map.get(&cls_name).or_else(|| {
            self.struct_mangled_map.iter().find_map(
                |(k, v)| {
                    if *k == cls_name { Some(v) } else { None }
                },
            )
        });

        if let Some(mmap) = mmap {
            let field_str = self.context.resolve_string(&field);
            let mangled_name = mmap.get(&field).copied().or_else(|| {
                mmap.iter().find_map(|(k, v)| {
                    if self.context.resolve_string(k) == field_str {
                        Some(*v)
                    } else {
                        None
                    }
                })
            });

            if let Some(mangled_name) = mangled_name {
                let actual_func_name = if self.funcs.contains_key(&mangled_name) {
                    mangled_name
                } else {
                    panic!(
                        "uh oh. failed with {mangled_name} \n\n{:?}",
                        self.funcs.keys()
                    )
                };

                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::Call {
                    dest: Some(dest),
                    func: Operand::FunctionRef(actual_func_name),
                    args: operands.clone(),
                });

                let ret_ty = self
                    .funcs
                    .get(&actual_func_name)
                    .map(|f| f.ret_type.clone())
                    .unwrap_or(SsaType::I64);
                self.current_block_data.value_types.insert(dest, ret_ty);
                return Some(dest);
            }
        }

        let struct_slots = self.struct_method_slots.get(&cls_name).or_else(|| {
            self.struct_method_slots
                .iter()
                .find_map(|(k, v)| if *k == cls_name { Some(v) } else { None })
        });

        if let Some(struct_slots) = struct_slots {
            if let Some(slot_idx) = struct_slots.get(&field) {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::InterfaceDispatch {
                    dest: Some(dest),
                    object: obj_val,
                    method_slot: *slot_idx,
                    args: operands.clone(),
                });
                let ret_ty = self
                    .interface_methods
                    .get(&cls_name)
                    .and_then(|methods| methods.iter().find(|(name, _, _)| name == &field))
                    .map(|(_, _, ret)| ret.clone())
                    .unwrap_or(SsaType::I64);
                self.current_block_data.value_types.insert(dest, ret_ty);
                return Some(dest);
            }
        }

        let iface_slots = self.interface_method_slots.get(&cls_name).or_else(|| {
            self.interface_method_slots
                .iter()
                .find_map(|(k, v)| if *k == cls_name { Some(v) } else { None })
        });

        if let Some(iface_slots) = iface_slots {
            if let Some(slot_idx) = iface_slots.get(&field) {
                let dest = self.current_block_data.fresh_value();
                self.emit(Instruction::InterfaceDispatch {
                    dest: Some(dest),
                    object: obj_val,
                    method_slot: *slot_idx,
                    args: operands.clone(),
                });
                let ret_ty = self
                    .struct_vtable_slots
                    .get(&cls_name)
                    .filter(|slots| *slot_idx < slots.len())
                    .and_then(|slots| self.funcs.get(&slots[*slot_idx]))
                    .map(|f| f.ret_type.clone())
                    .unwrap_or(SsaType::Void);
                self.current_block_data.value_types.insert(dest, ret_ty);
                return Some(dest);
            }
        }

        None
    }

    pub(crate) fn lower_interface_call(
        &mut self,
        callee: &HirExpr<'a, 'bump>,
        args: &[HirExpr<'a, 'bump>],
        interface: StrId,
    ) -> Value {
        let HirExpr::FieldAccess {
            object,
            field,
            span: _,
        } = callee
        else {
            panic!("InterfaceCall callee not FieldAccess; unsupported shape")
        };

        let obj_val = self.lower_expr(object);

        let param_types: Vec<SsaType> = self
            .interface_methods
            .get(&interface)
            .and_then(|methods| methods.iter().find(|(name, _, _)| name == field))
            .map(|(_, params, _)| params.clone())
            .unwrap_or_default();

        if matches!(param_types.first(), Some(SsaType::Dyn)) {
            self.record_arg_move(object);
        }

        let mut operands: SmallVec<Operand, 8> = SmallVec::new();
        for (i, a) in args.iter().enumerate() {
            if Self::is_move_by_value(param_types.get(i).unwrap()) {
                self.record_arg_move(a);
            }
            operands.push(Operand::Value(self.lower_expr(a)));
        }

        let iface_id = *self.interface_id_map.get(&interface).unwrap_or_else(|| {
            panic!(
                "Unknown interface {} in InterfaceCall",
                self.context.resolve_string(&*interface)
            )
        });

        let iface_slot_map = self
            .interface_method_slots
            .get(&interface)
            .unwrap_or_else(|| {
                panic!(
                    "Interface {} has no method slots",
                    self.context.resolve_string(&*interface)
                )
            });

        let method_slot_in_iface = iface_slot_map.get(field).unwrap_or_else(|| {
            panic!(
                "Interface {} has no method {}",
                self.context.resolve_string(&*interface),
                self.context.resolve_string(&*field)
            )
        });

        let interface_val = match self.current_block_data.value_types.get(&obj_val).cloned() {
            Some(SsaType::User(ref _name, _args)) => {
                let upcast_dest = self.current_block_data.fresh_value();
                self.emit(Instruction::UpcastToInterface {
                    dest: upcast_dest,
                    object: obj_val,
                    interface_id: iface_id,
                });

                self.current_block_data
                    .value_types
                    .insert(upcast_dest, SsaType::Interface(interface));
                upcast_dest
            }
            _ => obj_val,
        };

        let dest = self.current_block_data.fresh_value();
        self.emit(Instruction::InterfaceDispatch {
            dest: Some(dest),
            object: interface_val,
            method_slot: *method_slot_in_iface,
            args: operands,
        });

        self.current_block_data
            .value_types
            .insert(dest, SsaType::I64);
        dest
    }
}
