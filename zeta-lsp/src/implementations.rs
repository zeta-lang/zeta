use ir::{ast::Stmt, hir::StrId, span::SourceSpan};
use lsp_types::{GotoDefinitionParams, GotoDefinitionResponse, Location};
use sentinel_typechecker::type_checker::SymbolId;

use crate::{
    dispatch::{
        definition_location, find_ident_range_on_line, push_unique, resolve_visible_item,
        symbol_under_cursor,
    },
    state::ServerState,
    text_utils::{ident_at_column, line_at, span_to_range},
    urls::path_to_uri,
};

/// An `impl` block, flattened to what the LSP needs so nothing else has to name `ImplDecl`.
struct ImplSite<'a, 'bump> {
    span: SourceSpan<'a>,
    target_name: Option<StrId>,
    target_span: SourceSpan<'a>,
    iface: Option<(StrId, &'bump [StrId])>,
    methods: Vec<(StrId, SourceSpan<'a>)>,
}

fn collect_impl_sites<'a, 'bump>(stmts: &[Stmt<'a, 'bump>], out: &mut Vec<ImplSite<'a, 'bump>>) {
    for stmt in stmts {
        let (_, stmt) = stmt.peel();
        match stmt {
            Stmt::ImplDecl(i) => {
                let target_name = match i.target.kind {
                    ir::ast::TypeKind::Struct { name, .. } => Some(name),
                    _ => None,
                };
                let iface = i.interface.and_then(|t| match t.kind {
                    ir::ast::TypeKind::Struct { name, path, .. } => Some((name, path)),
                    _ => None,
                });
                let target_span = if i.target.span.line != 0 {
                    i.target.span
                } else {
                    i.span
                };
                let methods = i
                    .methods
                    .unwrap_or(&[])
                    .iter()
                    .map(|m| {
                        (
                            m.name,
                            if m.name_span.line != 0 {
                                m.name_span
                            } else {
                                m.span
                            },
                        )
                    })
                    .collect();
                out.push(ImplSite {
                    span: i.span,
                    target_name,
                    target_span,
                    iface,
                    methods,
                });
            }
            Stmt::Module(md) => collect_impl_sites(md.body, out),
            _ => {}
        }
    }
}

pub(crate) fn all_modules<'a, 'bump>(
    checker: &sentinel_typechecker::TypeChecker<'a, 'bump>,
) -> Vec<usize> {
    let ctx = checker.context();
    let max = ctx
        .module_functions
        .keys()
        .copied()
        .chain(ctx.struct_owner_module.values().copied())
        .chain(ctx.enum_owner_module.values().copied())
        .chain(ctx.interface_owner_module.values().copied())
        .chain(checker.occurrences().iter().map(|o| o.module_idx))
        .max()
        .unwrap_or(0);
    (0..=max).collect()
}

/// Which interface declaration (module, item index) does an `impl .. by Name` refer to?
fn resolve_interface_ref<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    from_module: usize,
    name: StrId,
    path: &[StrId],
) -> Option<(usize, usize)> {
    let is_interface = |hit: &(usize, usize, &'static str)| hit.2 == "trait";

    if !path.is_empty() {
        let is_iface = |m: usize, idx: usize| {
            state
                .compiler
                .module_with_arena(m)
                .and_then(|a| {
                    a.parse_result
                        .statements
                        .get(idx)
                        .map(|s| matches!(s.peel().1, Stmt::InterfaceDecl(_)))
                })
                .unwrap_or(false)
        };
        let dep = state.compiler.dep_graph_ref().borrow();
        if let Some(target) = dep.resolve_module_path(path) {
            if let Some((m, idx, _)) = dep
                .resolve_item_in_module(target, name)
                .filter(|h| is_iface(h.0, h.1))
            {
                return Some((m, idx));
            }
        }
    }
    if let Some((m, idx, _)) = resolve_visible_item(state, from_module, name).filter(is_interface) {
        return Some((m, idx));
    }

    let mut owners = ctx
        .interfaces
        .iter()
        .filter(|(_, i)| i.unmangled_name == name)
        .filter_map(|(k, _)| ctx.interface_owner_module.get(k.as_str()).copied());
    let owner = owners.next()?;
    if owners.next().is_some() {
        return None;
    }
    let stmts = &state
        .compiler
        .module_with_arena(owner)?
        .parse_result
        .statements;
    let idx = stmts
        .iter()
        .position(|s| matches!(s.peel().1, Stmt::InterfaceDecl(d) if d.name == name))?;
    Some((owner, idx))
}

/// `None`: the symbol isn't an interface or an interface method.
/// `Some(vec![])`: it is, and nothing implements it.
pub fn implementations_of<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &sentinel_typechecker::TypeChecker<'a, 'bump>,
    symbol: SymbolId,
) -> Option<Vec<Location>> {
    let ctx = checker.context();

    let (iface_module, iface_idx, method_name) = match symbol {
        SymbolId::Item {
            module_idx: m,
            item_idx,
            ..
        } => {
            let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
            let (_, stmt) = stmts.get(item_idx)?.peel();
            if !matches!(stmt, Stmt::InterfaceDecl(_)) {
                return None;
            }
            (m, item_idx, None)
        }
        SymbolId::Method {
            module_idx: m,
            item_idx,
            method_idx,
        } => {
            let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
            let (_, stmt) = stmts.get(item_idx)?.peel();
            let Stmt::InterfaceDecl(i) = stmt else {
                return None;
            };
            (m, item_idx, Some(i.methods?.get(method_idx)?.name))
        }
        _ => return None,
    };

    let mut locations = Vec::new();
    for m in all_modules(checker) {
        let Some(module) = state.compiler.module_with_arena(m) else {
            continue;
        };
        let mut sites = Vec::new();
        collect_impl_sites(&module.parse_result.statements, &mut sites);
        if sites.is_empty() {
            continue;
        }
        let Some(source) = state.compiler.source_text(m) else {
            continue;
        };
        let Some(path) = state.compiler.path_for_module(m) else {
            continue;
        };
        let Ok(uri) = path_to_uri(path) else { continue };

        for site in sites {
            let Some((iname, ipath)) = site.iface else {
                continue;
            };
            if resolve_interface_ref(state, ctx, m, iname, ipath) != Some((iface_module, iface_idx))
            {
                continue;
            }
            match method_name {
                // the interface: point at the implementing type in `impl Type by Iface`
                None => {
                    let range = match site.target_name {
                        Some(n) => find_ident_range_on_line(
                            &source,
                            site.target_span.line,
                            n.as_str(),
                            site.target_span,
                        ),
                        None => span_to_range(
                            &source,
                            site.span.line,
                            site.span.column,
                            site.span.end_line,
                            site.span.end_column,
                        ),
                    };
                    push_unique(
                        &mut locations,
                        Location {
                            uri: uri.clone(),
                            range,
                        },
                    );
                }
                // an interface method: point at each implementing method of that name
                Some(name) => {
                    for (_, mspan) in site.methods.iter().filter(|(n, _)| *n == name) {
                        let range =
                            find_ident_range_on_line(&source, mspan.line, name.as_str(), *mspan);
                        push_unique(
                            &mut locations,
                            Location {
                                uri: uri.clone(),
                                range,
                            },
                        );
                    }
                }
            }
        }
    }
    Some(locations)
}

pub fn go_to_implementation<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    params: &GotoDefinitionParams,
) -> Option<GotoDefinitionResponse> {
    let uri = &params.text_document_position_params.text_document.uri;
    let position = params.text_document_position_params.position;

    let doc = state.documents.get(uri)?;
    let module_idx = state.compiler.module_idx_for_path(&doc.path)?;
    let source = state.compiler.source_text(module_idx)?;
    let word = ident_at_column(line_at(&source, position.line)?, position.character)?;

    let checker_rc = state.compiler.type_checker();
    let checker = checker_rc.borrow();
    let symbol = symbol_under_cursor(state, &checker, module_idx, &source, position, word)?;

    match implementations_of(state, &checker, symbol) {
        Some(locations) if locations.is_empty() => None,
        Some(locations) => Some(GotoDefinitionResponse::Array(locations)),
        // not an interface: the symbol is its own implementation
        None => definition_location(state, &checker, symbol, word)
            .map(|(_, l)| GotoDefinitionResponse::Scalar(l)),
    }
}
