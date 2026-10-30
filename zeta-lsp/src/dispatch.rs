#![allow(unused)]
use anyhow::{Context, Result};
use ir::ast::{ExternModifier, FuncSafety, InlineModifier, Stmt, Visibility};
use ir::hir::{
    HirEnum, HirFunc, HirGeneric, HirInterface, HirParam, HirStruct, HirType, HirTypeKind, StrId,
};
use ir::ir_hasher::HashSet;
use ir::span::SourceSpan;
use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, DidSaveTextDocument, Exit,
    Notification as _, PublishDiagnostics,
};
use lsp_types::request::{
    Completion, DocumentSymbolRequest, GotoDefinition, GotoImplementation,
    GotoTypeDefinitionParams, HoverRequest, References, Rename, Request as _,
    SemanticTokensFullRequest,
};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionParams, CompletionResponse,
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    DidSaveTextDocumentParams, DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse,
    GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents, HoverParams,
    InsertTextFormat, Location, MarkupContent, MarkupKind, Position, PublishDiagnosticsParams,
    ReferenceParams, RenameParams, SemanticTokens, SemanticTokensParams, SemanticTokensResult,
    SymbolKind, TextDocumentContentChangeEvent, TextEdit, Uri, WorkspaceEdit,
};
use sentinel_typechecker::type_checker::SymbolId;
use sentinel_typechecker::{TypeChecker, TypeContext};
use std::borrow::Cow;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use zeta_compiler_api::file_handling::compiler_lib_path;
use zeta_compiler_api::file_loader::choose_file_loader;

use crate::implementations::{all_modules, go_to_implementation, implementations_of};
use crate::lsp_diagnostics::group_by_file;
use crate::state::{Document, ServerState};
use crate::text_utils::{
    byte_offset_in_source, declaration_span, ident_at_column, ident_at_end, line_at,
    nearest_prior_occurrence, prefix_before, range_contains, span_contains, span_to_range,
    tightest_occurrence, to_span_coords,
};
use crate::urls::{path_to_uri, uri_to_path};
use lsp_types::CompletionItemKind as K;

use crate::text_utils::byte_offset_from_utf16;

use crate::state::ModuleCandidates;

struct ImportSite {
    pos: Position,
    lead: &'static str,
    trail: &'static str,
}

impl ImportSite {
    fn edit(&self, path: &str, member: &str) -> TextEdit {
        TextEdit {
            range: lsp_types::Range {
                start: self.pos,
                end: self.pos,
            },
            new_text: format!("{}import {path}.{member};{}", self.lead, self.trail),
        }
    }
}

/// Where new `import` lines go: after the last import, else after `package`, else the top.
fn import_site<'a, 'bump>(state: &ServerState<'a, 'bump>, module_idx: usize) -> Option<ImportSite> {
    let src = state.compiler.source_text(module_idx)?;
    let stmts = &state
        .compiler
        .module_with_arena(module_idx)?
        .parse_result
        .statements;
    let (mut last_import, mut package) = (None::<usize>, None::<usize>);
    for s in stmts.iter() {
        match s.peel().1 {
            Stmt::Import(i) if i.path.span.line != 0 => {
                let l = i.path.span.end_line;
                last_import = Some(last_import.map_or(l, |p| p.max(l)));
            }
            Stmt::Package(p) if p.span.line != 0 => package = Some(p.span.end_line),
            _ => {}
        }
    }
    let (after, lead, trail) = match (last_import, package) {
        (Some(l), _) => (l, "", "\n"),
        (None, Some(l)) => (l, "\n", "\n"),
        (None, None) => (0, "", "\n\n"),
    };
    // `after` is a 1-indexed line number, i.e. the 0-indexed line below it.
    if after == 0 || line_at(&src, after as u32).is_some() {
        return Some(ImportSite {
            pos: Position {
                line: after as u32,
                character: 0,
            },
            lead,
            trail,
        });
    }
    // the statement is on the last line and the file has no trailing newline
    let last = line_at(&src, after as u32 - 1)?;
    Some(ImportSite {
        pos: Position {
            line: after as u32 - 1,
            character: crate::text_utils::utf16_offset_from_byte(last, last.len()),
        },
        lead: if lead.is_empty() { "\n" } else { "\n\n" },
        trail: "",
    })
}

fn build_module_candidates<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &TypeContext<'a, 'bump>,
    m: usize,
) -> ModuleCandidates {
    let (package, package_segments) = {
        let dep = state.compiler.dep_graph_ref().borrow();
        (
            dep.get_module_package(m).map(|p| p.as_str().to_string()),
            dep.package_segments(m).map(|s| s.to_vec()),
        )
    };
    let mut items = Vec::new();
    if let (Some(arena), Some(src)) = (
        state.compiler.module_with_arena(m),
        state.compiler.source_text(m),
    ) {
        for s in arena.parse_result.statements.iter() {
            let stmt = s.peel().1;
            let vis = match stmt {
                Stmt::FuncDecl(f) => f.function_metadata.visibility,
                Stmt::StructDecl(x) => x.visibility,
                Stmt::EnumDecl(x) => x.visibility,
                Stmt::InterfaceDecl(x) => x.visibility,
                _ => continue,
            };
            if vis == Visibility::Private {
                continue;
            }
            if let Some(it) = item_from_stmt(ctx, m, stmt, usize::MAX, &src) {
                items.push((vis, it));
            }
        }
    }
    ModuleCandidates {
        package,
        package_segments,
        items,
    }
}

/// Case-insensitive subsequence match, same idea as the client's fuzzy filter.
fn fuzzy_match(label: &str, typed: &str) -> bool {
    let mut it = label.chars().flat_map(char::to_lowercase);
    typed
        .chars()
        .flat_map(char::to_lowercase)
        .all(|t| it.any(|c| c == t))
}

/// Items from modules this file has not imported; accepting one adds the import.
fn auto_import_items<'a, 'bump>(
    state: &mut ServerState<'a, 'bump>,
    checker: &TypeChecker<'a, 'bump>,
    module_idx: usize,
    visible: &HashSet<String>,
    typed: &str,
) -> Vec<CompletionItem> {
    let Some(site) = import_site(state, module_idx) else {
        return vec![];
    };
    let ctx = checker.context();
    let (auto_targets, my_pkg) = {
        let dep = state.compiler.dep_graph_ref().borrow();
        let targets: std::collections::HashSet<usize> = dep
            .auto_import_packages()
            .iter()
            .filter_map(|p| dep.resolve_module_path(p))
            .collect();
        (
            targets,
            dep.package_segments(module_idx).map(|s| s.to_vec()),
        )
    };

    let count = *state
        .completion_cache
        .module_count
        .get_or_insert_with(|| all_modules(checker).len());

    let mut offered: HashSet<(String, String)> = HashSet::default();
    let mut out = Vec::new();
    for m in 0..count {
        if m == module_idx {
            continue;
        }
        // Oh how I wanted to use `entry`.. but thank you Rust borrow checker
        // P.S: I bet I forgot a better pattern and now someone's going to be like "How could you do this?! I feel violated by this Rust code!"
        let mut optional_cands = state.completion_cache.modules.get(&m);
        let cands = if let Some(cands) = optional_cands {
            cands
        } else {
            let new_cands = build_module_candidates(state, ctx, m);
            state.completion_cache.modules.insert(m, new_cands);
            state.completion_cache.modules.get(&m).unwrap()
        };
        let Some(pkg) = cands.package.as_deref() else {
            continue;
        };
        let same_pkg = my_pkg.as_deref() == cands.package_segments.as_deref();

        for (vis, template) in &cands.items {
            let reachable = match vis {
                Visibility::Public => true,
                Visibility::Internal | Visibility::Module => same_pkg,
                Visibility::Private => false,
            };
            if !reachable
                || visible.contains(&template.label)
                || !fuzzy_match(&template.label, typed)
                || !offered.insert((pkg.to_string(), template.label.clone()))
            {
                continue;
            }
            let mut it = template.clone();
            it.detail = Some(format!("import {pkg}.{}", it.label));
            it.sort_text = Some(format!("5{}", it.label));
            it.additional_text_edits = Some(vec![site.edit(pkg, &it.label)]);
            out.push(it);
        }
    }
    out
}

pub struct ImportRef {
    pub modules: Vec<String>, // path segments up to the hovered one (all of them if the member is hovered)
    pub member: Option<String>, // set when the hovered ident is after the `.`
}

fn import_ref_at(source: &str, position: Position) -> Option<ImportRef> {
    let line = line_at(source, position.line)?;
    let trimmed = line.trim_start();
    let kw = ["import ", "package "]
        .into_iter()
        .find(|k| trimmed.starts_with(*k))?;
    let start = line.len() - trimmed.len() + kw.len();
    let cursor = crate::text_utils::byte_offset_from_utf16(line, position.character);
    if cursor < start {
        return None;
    }
    let body = &line[start..];
    let rel = cursor - start;

    let mut idents: Vec<(usize, usize, bool)> = Vec::new(); // (start, end, after_dot)
    let (mut after_dot, mut cur) = (false, None::<usize>);
    for (i, c) in body
        .char_indices()
        .chain(std::iter::once((body.len(), ' ')))
    {
        let is_id = c.is_alphanumeric() || c == '_';
        match (cur, is_id) {
            (None, true) => cur = Some(i),
            (Some(s), false) => {
                idents.push((s, i, after_dot));
                cur = None;
            }
            _ => {}
        }
        if c == '.' {
            after_dot = true;
        }
    }
    let hit = idents.iter().position(|&(s, e, _)| s <= rel && rel <= e)?;
    let hovered_is_member = idents[hit].2;
    let upto = if hovered_is_member {
        idents.len()
    } else {
        hit + 1
    };
    let modules: Vec<String> = idents[..upto]
        .iter()
        .filter(|(_, _, d)| !*d)
        .map(|&(s, e, _)| body[s..e].to_string())
        .collect();
    let member = hovered_is_member.then(|| body[idents[hit].0..idents[hit].1].to_string());
    Some(ImportRef { modules, member })
}

fn module_package_span<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    m: usize,
) -> Option<SourceSpan<'a>> {
    let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
    stmts.iter().find_map(|s| match s.peel().1 {
        Stmt::Package(p) => Some(p.span),
        _ => None,
    })
}

/// package.zeta: first `///` block in the file. Any other file: the docs above its `package` stmt.
fn module_doc<'a, 'bump>(state: &ServerState<'a, 'bump>, m: usize) -> Option<String> {
    let src = state.compiler.source_text(m)?;
    let path = state.compiler.path_for_module(m)?;
    if path.file_name().and_then(|n| n.to_str()) == Some("package.zeta") {
        return crate::doc_comments::first_doc_block(&src);
    }
    let span = module_package_span(state, m)?;
    crate::doc_comments::doc_comment_above(&src, span.line)
}

fn module_location<'a, 'bump>(state: &ServerState<'a, 'bump>, m: usize) -> Option<Location> {
    let path = state.compiler.path_for_module(m)?;
    let src = state.compiler.source_text(m)?;
    let range = module_package_span(state, m)
        .map(|sp| span_to_range(&src, sp.line, sp.column, sp.end_line, sp.end_column))
        .unwrap_or_default();
    Some(Location {
        uri: path_to_uri(path).ok()?,
        range,
    })
}

/// Hover for a declared item, resolved from the dep graph's (module, item, tag) coordinate.
fn hover_for_item<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    m: usize,
    item_idx: usize,
    tag: &str,
) -> Option<Hover> {
    let doc = state
        .compiler
        .module_with_arena(m)
        .and_then(|a| item_decl_span(&a.parse_result.statements, item_idx, tag))
        .and_then(|span| {
            let src = state.compiler.source_text(m)?;
            crate::doc_comments::doc_comment_above(&src, span.line)
        });
    let sig = match tag {
        "func_sig" | "func_body" => {
            func_for_item(state, ctx, m, item_idx, tag).map(|f| format_func_signature(&f))
        }
        "type" | "trait" => decl_hir_key(state, ctx, m, item_idx)
            .and_then(|key| {
                let k = key.as_str();
                ctx.structs
                    .get(k)
                    .map(|d| format_struct_signature(d))
                    .or_else(|| ctx.enums.get(k).map(|e| format_enum_signature(e)))
                    .or_else(|| ctx.interfaces.get(k).map(|i| format_interface_signature(i)))
            })
            // type aliases have no HIR entry: show their declaration line
            .or_else(|| item_declaration_text(state, m, item_idx, tag)),
        "const" | "static" => item_declaration_text(state, m, item_idx, tag),
        _ => None,
    }?;
    Some(Hover {
        contents: HoverContents::Markup(format_hover_markup(&sig, doc.as_deref())),
        range: None,
    })
}

fn import_hover<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    r: &ImportRef,
) -> Option<Hover> {
    let ids: Vec<StrId> = r
        .modules
        .iter()
        .map(|s| StrId(ctx.string_pool.intern(s)))
        .collect();
    let m = state
        .compiler
        .dep_graph_ref()
        .borrow()
        .resolve_module_path(&ids)?;
    if let Some(member) = &r.member {
        let mid = StrId(ctx.string_pool.intern(member));
        let (dm, idx, tag) = state
            .compiler
            .dep_graph_ref()
            .borrow()
            .resolve_item_in_module(m, mid)?;
        return hover_for_item(state, ctx, dm, idx, tag);
    }
    let sig = format!("package {}", r.modules.join("::"));
    Some(Hover {
        contents: HoverContents::Markup(format_hover_markup(&sig, module_doc(state, m).as_deref())),
        range: None,
    })
}

pub fn initialize<'a, 'bump>(
    state: &mut ServerState<'a, 'bump>,
    params: serde_json::Value,
) -> Result<()> {
    if let Ok(init) = serde_json::from_value::<lsp_types::InitializeParams>(params) {
        state.workspace_root = init
            .workspace_folders
            .as_ref()
            .and_then(|folders| folders.first())
            .and_then(|folder| uri_to_path(&folder.uri).ok())
            .or_else(|| {
                #[allow(deprecated)]
                init.root_uri.as_ref().and_then(|uri| uri_to_path(uri).ok())
            });

        let stdlib = compiler_lib_path().map_err(|e| anyhow::anyhow!("{}", e))?;
        let loader = choose_file_loader();

        state
            .compiler
            .load_directory(&loader, &stdlib, true)
            .map_err(|e| anyhow::anyhow!("{}", e))?;

        let Some(root) = state.workspace_root.clone() else {
            return Ok(());
        };
        state
            .compiler
            .load_directory(&loader, &root.join("src"), false)
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }
    Ok(())
}

pub fn request<'a, 'bump>(
    connection: &Connection,
    state: &mut ServerState<'a, 'bump>,
    request: &Request,
) -> Result<()> {
    match request.method.as_str() {
        HoverRequest::METHOD => {
            let params: HoverParams = serde_json::from_value(request.params.clone())?;

            let result = hover_at(state, &params);

            let response = Response::new_ok(request.id.clone(), result);

            connection.sender.send(Message::Response(response))?;
        }
        SemanticTokensFullRequest::METHOD => {
            let params: SemanticTokensParams = serde_json::from_value(request.params.clone())?;

            let result = semantic_tokens_at(state, &params);

            let response = Response::new_ok(request.id.clone(), result);

            connection.sender.send(Message::Response(response))?;
        }
        DocumentSymbolRequest::METHOD => {
            let params: DocumentSymbolParams = serde_json::from_value(request.params.clone())?;
            let result = document_symbols_at(state, &params);
            let response = Response::new_ok(request.id.clone(), result);
            connection.sender.send(Message::Response(response))?;
        }
        Completion::METHOD => {
            let params: CompletionParams = serde_json::from_value(request.params.clone())?;
            let items = completions_at(state, &params);
            let response =
                Response::new_ok(request.id.clone(), Some(CompletionResponse::Array(items)));
            connection.sender.send(Message::Response(response))?;
        }
        GotoDefinition::METHOD => {
            let params: GotoDefinitionParams = serde_json::from_value(request.params.clone())?;
            let result = go_to_definition(state, &params);
            let response = Response::new_ok(request.id.clone(), result);
            connection.sender.send(Message::Response(response))?;
        }
        References::METHOD => {
            let params: ReferenceParams = serde_json::from_value(request.params.clone())?;
            let result = references_at(state, &params);
            let response = Response::new_ok(request.id.clone(), result);
            connection.sender.send(Message::Response(response))?;
        }
        Rename::METHOD => {
            let params: RenameParams = serde_json::from_value(request.params.clone())?;
            match rename_at(state, &params) {
                Ok(edit) => {
                    connection.sender.send(Message::Response(Response::new_ok(
                        request.id.clone(),
                        edit,
                    )))?;
                }
                Err(msg) => {
                    connection.sender.send(Message::Response(Response::new_err(
                        request.id.clone(),
                        -32803, // LSP "RequestFailed"
                        msg,
                    )))?;
                }
            }
        }
        GotoImplementation::METHOD => {
            let params: GotoDefinitionParams = serde_json::from_value(request.params.clone())?;
            let result = go_to_implementation(state, &params);
            let response = Response::new_ok(request.id.clone(), result);
            connection.sender.send(Message::Response(response))?;
        }
        _ => {
            let response = Response::new_err(
                request.id.clone(),
                lsp_server::ErrorCode::MethodNotFound as i32,
                format!("unhandled method: {}", request.method),
            );
            connection.sender.send(Message::Response(response))?;
        }
    }
    Ok(())
}

pub fn notification<'a, 'bump>(
    connection: &Connection,
    state: &mut ServerState<'a, 'bump>,
    notification: &Notification,
    recheck: bool,
) -> Result<()> {
    match notification.method.as_str() {
        DidOpenTextDocument::METHOD => {
            let params: DidOpenTextDocumentParams =
                serde_json::from_value(notification.params.clone())?;
            let uri = params.text_document.uri;
            let path =
                uri_to_path(&uri).map_err(|_| anyhow::anyhow!("non-file URI: {}", uri.as_str()))?;

            state.documents.insert(
                uri.clone(),
                Document {
                    version: params.text_document.version,
                    text: params.text_document.text.clone(),
                    path: path.clone(),
                },
            );

            invalidate_completions(&state.compiler, &mut state.completion_cache, &path, true);

            let reporter = state.compiler.open_module(&path, params.text_document.text);
            publish_all(connection, &reporter, Some(&path))?;
        }

        DidChangeTextDocument::METHOD => {
            let params: DidChangeTextDocumentParams =
                serde_json::from_value(notification.params.clone())?;
            let uri = params.text_document.uri;

            let Some(doc) = state.documents.get_mut(&uri) else {
                return Ok(());
            };
            doc.version = params.text_document.version;
            let path = doc.path.clone();

            for change in params.content_changes {
                apply_change(&mut doc.text, change);
            }
            if !recheck {
                return Ok(());
            }

            invalidate_completions(&state.compiler, &mut state.completion_cache, &path, true);

            let reporter = state.compiler.update_module(&path, doc.text.clone());
            publish_all(connection, &reporter, Some(&path))?;
        }

        DidSaveTextDocument::METHOD => {
            let params: DidSaveTextDocumentParams =
                serde_json::from_value(notification.params.clone())?;
            let uri = params.text_document.uri;
            let Some(doc) = state.documents.get(&uri) else {
                return Ok(());
            };
            let path = doc.path.clone();
            // include_text: false, so re-read from disk explicitly here.
            let source = match std::fs::read_to_string(&path) {
                Ok(s) => s,
                Err(err) => {
                    eprintln!("[zeta-lsp] cannot read {} on save: {err}", path.display());
                    return Ok(());
                }
            };
            if doc.text == source {
                return Ok(());
            }

            invalidate_completions(&state.compiler, &mut state.completion_cache, &path, true);

            let reporter = state.compiler.update_module(&path, source);
            publish_all(connection, &reporter, Some(&path))?;
        }

        DidCloseTextDocument::METHOD => {
            let params: DidCloseTextDocumentParams =
                serde_json::from_value(notification.params.clone())?;
            let uri = params.text_document.uri;
            state.completion_cache.invalidate_all();
            if let Some(doc) = state.documents.remove(&uri) {
                if let Ok(disk) = std::fs::read_to_string(&doc.path) {
                    if disk != doc.text {
                        let r = state.compiler.update_module(&doc.path, disk);
                        publish_all(connection, &r, Some(&doc.path))?;
                    }
                }
            }
        }

        Exit::METHOD => {
            state.should_exit = true;
        }

        _ => {}
    }
    Ok(())
}

fn semantic_tokens_at<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    params: &SemanticTokensParams,
) -> Option<SemanticTokensResult> {
    let uri = &params.text_document.uri;

    let doc = state.documents.get(uri)?;
    let module_idx = state.compiler.module_idx_for_path(&doc.path)?;
    let source = state.compiler.source_text(module_idx)?;

    let checker = state.compiler.type_checker();
    let checker = checker.borrow();

    let tokens =
        crate::semantic_tokens::build_semantic_tokens(&checker, &state, module_idx, &source);

    Some(SemanticTokensResult::Tokens(SemanticTokens {
        result_id: None,
        data: tokens,
    }))
}

fn publish_all(
    connection: &Connection,
    reporter: &ir::errors::reporter::ErrorReporter,
    touched: Option<&Path>,
) -> Result<()> {
    let mut grouped = group_by_file(reporter);
    for name in reporter.source_files.keys() {
        grouped.entry(name.clone()).or_default();
    }
    if let Some(p) = touched {
        grouped.entry(p.to_string_lossy().into_owned()).or_default();
    }
    for (file_name, diagnostics) in grouped {
        let Ok(uri) = path_to_uri(Path::new(&file_name)) else {
            continue;
        };

        let params = PublishDiagnosticsParams {
            uri,
            diagnostics,
            version: None,
        };
        let notif = Notification::new(PublishDiagnostics::METHOD.to_string(), params);
        connection.sender.send(Message::Notification(notif))?;
    }
    Ok(())
}

fn rename_at<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    params: &RenameParams,
) -> Result<WorkspaceEdit, String> {
    let text_document = &params.text_document_position.text_document;
    let position = params.text_document_position.position;
    let new_name = &params.new_name;

    let doc = state
        .documents
        .get(&text_document.uri)
        .ok_or("document not open")?;
    let module_idx = state
        .compiler
        .module_idx_for_path(&doc.path)
        .ok_or("module not loaded")?;
    let source = state
        .compiler
        .source_text(module_idx)
        .ok_or("source unavailable")?;
    let (line, column) = to_span_coords(&source, position);

    let checker_rc = state.compiler.type_checker();
    let checker = checker_rc.borrow();

    let occ = tightest_occurrence(checker.occurrences(), module_idx, line, column)
        .ok_or("no renameable symbol at this position")?;
    let symbol_id = occ.symbol;

    if matches!(symbol_id, SymbolId::Item { .. }) {
        return Err(
            "renaming functions, structs, enums, and interfaces isn't supported yet, \
                    usage sites for those aren't tracked, only declarations, so renaming \
                     would break every call site silently"
                .to_string(),
        );
    }

    let mut by_module: HashMap<usize, Vec<(SourceSpan, StrId)>> = HashMap::default();
    for o in checker.occurrences() {
        if o.symbol == symbol_id {
            by_module
                .entry(o.module_idx)
                .or_default()
                .push((o.span, o.name));
        }
    }
    if by_module.is_empty() {
        return Err("symbol has no recorded occurrences".to_string());
    }

    let mut changes: HashMap<lsp_types::Uri, Vec<TextEdit>> = HashMap::default();

    for (m, entries) in by_module {
        let m_source = state
            .compiler
            .source_text(m)
            .ok_or("source unavailable for a module containing this symbol")?;
        let m_path = state
            .compiler
            .path_for_module(m)
            .ok_or("no file path recorded for a module containing this symbol")?;

        let mut edits = Vec::new();
        for (span, name) in entries {
            let expected = name.to_string();
            if crate::text_utils::span_text(&m_source, &span).as_deref() != Some(expected.as_str())
            {
                return Err(format!(
                    "rename aborted: couldn't safely locate `{}` at one of its recorded \
                     positions. This happens when a symbol has at least one usage whose span \
                     covers more than just the identifier itself (a known gap for field access \
                     expressions), renaming it isn't safe until that's fixed upstream.",
                    expected
                ));
            }
            let range = span_to_range(
                &m_source,
                span.line,
                span.column,
                span.end_line,
                span.end_column,
            );
            edits.push(TextEdit {
                range,
                new_text: new_name.clone(),
            });
        }

        let uri = path_to_uri(m_path).map_err(|_| "failed to build file URI".to_string())?;
        changes.insert(uri, edits);
    }

    Ok(WorkspaceEdit {
        changes: Some(changes),
        document_changes: None,
        change_annotations: None,
    })
}

pub fn hover_at<'a, 'bump>(state: &ServerState<'a, 'bump>, params: &HoverParams) -> Option<Hover> {
    let uri = &params.text_document_position_params.text_document.uri;
    let position = params.text_document_position_params.position;

    let doc = state.documents.get(uri)?;
    let module_idx = state.compiler.module_idx_for_path(&doc.path)?;
    let source = state.compiler.source_text(module_idx)?;
    let (line, column) = to_span_coords(&source, position);
    let word = ident_at_column(line_at(&source, position.line)?, position.character)?;

    let checker_rc = state.compiler.type_checker();
    let checker = checker_rc.borrow();
    let ctx = checker.context();

    if let Some(r) = import_ref_at(&source, position) {
        return import_hover(state, ctx, &r);
    }

    if let Some(m) = module_receiver_at(state, &checker, module_idx, &source, position, word) {
        return module_hover(state, m);
    }

    // 1. Try tightest_occurrence first
    if let Some(occ) = tightest_occurrence(checker.occurrences(), module_idx, line, column) {
        if occ.name.as_str() == word {
            let symbol_id = occ.symbol;
            match symbol_id {
                SymbolId::Variant {
                    enum_name,
                    variant_name,
                } => {
                    let doc = doc_for_symbol(state, &checker, symbol_id);
                    let def = ctx.enums.get(enum_name.as_str());
                    let owner = def.map(|e| e.unmangled_name).unwrap_or(enum_name);
                    let payload = def
                        .and_then(|e| e.variants.iter().find(|v| v.name == variant_name))
                        .filter(|v| !v.fields.is_empty())
                        .map(|v| {
                            let tys: Vec<String> =
                                v.fields.iter().map(|f| f.field_type.to_string()).collect();
                            format!("({})", tys.join(", "))
                        })
                        .unwrap_or_default();
                    let sig = format!("{}.{}{}", owner, variant_name, payload);
                    return Some(Hover {
                        contents: HoverContents::Markup(format_hover_markup(&sig, doc.as_deref())),
                        range: None,
                    });
                }
                SymbolId::Item {
                    module_idx: m,
                    item_idx,
                    tag,
                } => {
                    let doc = doc_for_symbol(state, &checker, symbol_id);
                    let key = occ.hir_name.as_str();
                    let sig = match tag {
                        "func_sig" | "func_body" => ctx
                            .module_functions
                            .get(&m)
                            .and_then(|f| f.get(key))
                            .copied()
                            .or_else(|| func_for_item(state, &ctx, m, item_idx, tag))
                            .map(|f| format_func_signature(&f)),
                        "type" => ctx
                            .structs
                            .get(key)
                            .map(|d| format_struct_signature(d))
                            .or_else(|| ctx.enums.get(key).map(|e| format_enum_signature(e)))
                            .or_else(|| item_declaration_text(state, m, item_idx, tag)),
                        "const" | "static" => item_declaration_text(state, m, item_idx, tag),
                        "trait" => ctx
                            .interfaces
                            .get(key)
                            .map(|i| format_interface_signature(i)),

                        _ => None,
                    };
                    if let Some(sig) = sig {
                        return Some(Hover {
                            contents: HoverContents::Markup(format_hover_markup(
                                &sig,
                                doc.as_deref(),
                            )),
                            range: None,
                        });
                    }
                }
                SymbolId::Method {
                    module_idx: m,
                    item_idx,
                    method_idx,
                } => {
                    let doc = doc_for_symbol(state, &checker, symbol_id);
                    if let Some(func) = method_for_symbol(state, &ctx, m, item_idx, method_idx) {
                        let sig = format_func_signature(&func);
                        return Some(Hover {
                            contents: HoverContents::Markup(format_hover_markup(
                                &sig,
                                doc.as_deref(),
                            )),
                            range: None,
                        });
                    }
                }
                SymbolId::Field {
                    struct_name,
                    field_name,
                } => {
                    let doc = doc_for_symbol(state, &checker, symbol_id);
                    let owner = ctx.structs.get(struct_name.as_str());
                    if let Some(owner) = owner {
                        // PANIC SAFETY: type checker should give this an error if the field does not exist
                        // before an occurrence is given to Field anyway.
                        let field = owner.fields.iter().find(|f| f.name == field_name).unwrap();
                        let manual_display: &str = if field.manual { "manual " } else { "" };
                        let sig = format!(
                            "{}{}{}: {}",
                            owner.visibility, manual_display, field_name, occ.ty
                        );
                        return Some(Hover {
                            contents: HoverContents::Markup(format_hover_markup(
                                &sig,
                                doc.as_deref(),
                            )),
                            range: None,
                        });
                    }
                }
                SymbolId::Local(..) => {
                    let doc = doc_for_symbol(state, &checker, symbol_id);
                    let sig = format!("{}: {}", word, occ.ty);
                    return Some(Hover {
                        contents: HoverContents::Markup(format_hover_markup(&sig, doc.as_deref())),
                        range: None,
                    });
                }
            }
        }
    }

    // 2. Fallback resolution for `word` (declarations, type annotations, imports, etc.)
    let word_id = StrId(ctx.string_pool.intern(word));
    if let Some((m, item_idx, tag)) = resolve_visible_item(state, module_idx, word_id) {
        if let Some(h) = hover_for_item(state, ctx, m, item_idx, tag) {
            return Some(h);
        }
    }

    if let Some((_, ty)) = ctx.variables.get(word) {
        let sig = format!("{}: {}", word, ty);
        return Some(Hover {
            contents: HoverContents::Markup(format_hover_markup(&sig, None)),
            range: None,
        });
    }

    None
}

pub fn module_receiver_at<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &TypeChecker<'a, 'bump>,
    module_idx: usize,
    source: &str,
    position: Position,
    word: &str,
) -> Option<usize> {
    let line_text = line_at(source, position.line)?;
    let cursor = byte_offset_from_utf16(line_text, position.character).min(line_text.len());
    let is_id = |c: char| c.is_alphanumeric() || c == '_';
    let start = line_text[..cursor]
        .char_indices()
        .rev()
        .take_while(|&(_, c)| is_id(c))
        .last()
        .map(|(i, _)| i)
        .unwrap_or(cursor);
    let end = cursor
        + line_text[cursor..]
            .find(|c: char| !is_id(c))
            .unwrap_or(line_text.len() - cursor);
    if &line_text[start..end] != word {
        return None;
    }

    // must be the head of a `recv.member` chain (not `x.args.y`, `a::args.y`, or `args..5`)
    let after = line_text[end..].trim_start();
    if !after.starts_with('.') || after.starts_with("..") {
        return None;
    }
    let before = line_text[..start].trim_end();
    if before.ends_with('.') || before.ends_with("::") {
        return None;
    }

    // a real variable / type / const wins; only a function hit (or nothing) is overridden
    let (line, column) = to_span_coords(source, position);
    if let Some(occ) = tightest_occurrence(checker.occurrences(), module_idx, line, column) {
        let is_fn = matches!(
            occ.symbol,
            SymbolId::Item {
                tag: "func_sig" | "func_body",
                ..
            } | SymbolId::Method { .. }
        );
        if occ.name.as_str() == word && !is_fn {
            return None;
        }
    }

    module_for_alias(state, checker.context(), module_idx, word)
}

fn module_hover<'a, 'bump>(state: &ServerState<'a, 'bump>, m: usize) -> Option<Hover> {
    let pkg = state
        .compiler
        .dep_graph_ref()
        .borrow()
        .get_module_package(m)?;
    let sig = format!("package {}", pkg.as_str());
    Some(Hover {
        contents: HoverContents::Markup(format_hover_markup(&sig, module_doc(state, m).as_deref())),
        range: None,
    })
}

fn item_declaration_text<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    m: usize,
    item_idx: usize,
    tag: &str,
) -> Option<String> {
    let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
    let span = item_decl_span(stmts, item_idx, tag)?;
    let src = state.compiler.source_text(m)?;
    let line = line_at(&src, span.line.checked_sub(1)? as u32)?;
    Some(line.trim().trim_end_matches(';').to_string())
}

fn strip_container<'a, 'b, 'bump>(ty: &'b HirType<'a, 'bump>) -> &'b HirType<'a, 'bump> {
    match ty.kind {
        HirTypeKind::Ref { inner, .. } => strip_container(inner),
        HirTypeKind::SafePointer { inner, .. } => strip_container(inner),
        HirTypeKind::UnsafePointer { inner, .. } => strip_container(inner),
        HirTypeKind::OwnedPointer { inner, .. } => strip_container(inner),
        _ => ty,
    }
}

fn document_symbols_at<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    params: &DocumentSymbolParams,
) -> Option<DocumentSymbolResponse> {
    let uri = &params.text_document.uri;
    let doc = state.documents.get(uri)?;
    let module_idx = state.compiler.module_idx_for_path(&doc.path)?;
    let source = state.compiler.source_text(module_idx)?;
    let module_with_arena = state.compiler.module_with_arena(module_idx)?;

    let mut symbols = Vec::new();
    for stmt in &module_with_arena.parse_result.statements {
        let (_, stmt) = stmt.peel();
        let (name, kind, span) = match stmt {
            Stmt::FuncDecl(f) => (f.name.to_string(), SymbolKind::FUNCTION, f.span),
            Stmt::StructDecl(s) => (s.name.to_string(), SymbolKind::STRUCT, s.span),
            Stmt::EnumDecl(e) => (e.name.to_string(), SymbolKind::ENUM, e.span),
            Stmt::InterfaceDecl(i) => (i.name.to_string(), SymbolKind::INTERFACE, i.span),
            _ => continue,
        };

        let range = span_to_range(
            &source,
            span.line,
            span.column,
            span.end_line,
            span.end_column,
        );

        #[allow(deprecated)] // DocumentSymbol.deprecated is a required-shape field we don't use
        symbols.push(DocumentSymbol {
            name,
            detail: None,
            kind,
            tags: None,
            deprecated: None,
            range,
            selection_range: range,
            children: None,
        });
    }

    Some(DocumentSymbolResponse::Nested(symbols))
}

pub fn item_decl_span<'a, 'bump>(
    stmts: &'bump [Stmt<'a, 'bump>],
    item_idx: usize,
    tag: &str,
) -> Option<SourceSpan<'a>> {
    let stmt = stmts.get(item_idx)?;
    let (_, stmt) = stmt.peel();
    match (stmt, tag) {
        (Stmt::FuncDecl(f), "func_sig" | "func_body") => Some(f.span),
        (Stmt::StructDecl(s), "type") => Some(s.span),
        (Stmt::EnumDecl(e), "type") => Some(e.span),
        (Stmt::InterfaceDecl(i), "trait") => Some(i.span),
        (Stmt::Const(c), "const") => Some(c.span),
        (Stmt::TypeAliasDecl(t), "type") => Some(t.span),
        (Stmt::Let(l), "static") => Some(l.span),
        _ => None,
    }
}

pub fn method_decl_span<'a, 'bump>(
    stmts: &'bump [Stmt<'a, 'bump>],
    item_idx: usize,
    method_idx: usize,
) -> Option<SourceSpan<'a>> {
    let stmt = stmts.get(item_idx)?;
    let (_, stmt) = stmt.peel();
    match stmt {
        Stmt::ImplDecl(i) => i.methods?.get(method_idx).map(|m| m.span),
        Stmt::InterfaceDecl(i) => i.methods?.get(method_idx).map(|m| m.span),
        _ => None,
    }
}

pub fn find_ident_range_on_line(
    source: &str,
    line_1_indexed: usize,
    name: &str,
    fallback_span: SourceSpan,
) -> lsp_types::Range {
    if let Some(line_str) = line_at(source, (line_1_indexed.saturating_sub(1)) as u32) {
        let mut search_start = 0;
        while let Some(idx) = line_str[search_start..].find(name) {
            let actual_idx = search_start + idx;
            let before = if actual_idx > 0 {
                line_str[..actual_idx].chars().next_back()
            } else {
                None
            };
            let after = line_str[actual_idx + name.len()..].chars().next();
            let is_ident_char = |c: char| c.is_alphanumeric() || c == '_';
            let word_boundary_before = before.map_or(true, |c| !is_ident_char(c));
            let word_boundary_after = after.map_or(true, |c| !is_ident_char(c));
            if word_boundary_before && word_boundary_after {
                let start_char = crate::text_utils::utf16_offset_from_byte(line_str, actual_idx);
                let end_char =
                    crate::text_utils::utf16_offset_from_byte(line_str, actual_idx + name.len());
                return lsp_types::Range {
                    start: lsp_types::Position {
                        line: (line_1_indexed.saturating_sub(1)) as u32,
                        character: start_char,
                    },
                    end: lsp_types::Position {
                        line: (line_1_indexed.saturating_sub(1)) as u32,
                        character: end_char,
                    },
                };
            }
            search_start = actual_idx + name.len();
        }
    }
    span_to_range(
        source,
        fallback_span.line,
        fallback_span.column,
        fallback_span.end_line,
        fallback_span.end_column,
    )
}

fn format_generic(g: &HirGeneric) -> String {
    let mut s = g.name.to_string();
    if !g.constraints.is_empty() {
        s.push_str(": ");
        let c_strs: Vec<String> = g.constraints.iter().map(|c| c.to_string()).collect();
        s.push_str(&c_strs.join(" + "));
    }
    if let Some(def) = &g.default_type {
        s.push_str(&format!(" = {}", def));
    }
    s
}

pub fn format_struct_signature(s: &HirStruct) -> String {
    let mut sig = String::new();
    sig.push_str(visibility_prefix(s.visibility));
    sig.push_str("struct ");
    sig.push_str(s.unmangled_name.as_str());
    if let Some(generics) = s.generics {
        if !generics.is_empty() {
            sig.push('<');
            let g_strs: Vec<String> = generics.iter().map(format_generic).collect();
            sig.push_str(&g_strs.join(", "));
            sig.push('>');
        }
    }
    if s.fields.is_empty() {
        sig.push_str(" {}");
    } else {
        sig.push_str(" {\n");
        for f in s.fields {
            sig.push_str("    ");
            sig.push_str(visibility_prefix(f.visibility));
            let manual_display: &str = if f.manual { "manual " } else { "" };
            sig.push_str(manual_display);
            sig.push_str(f.name.as_str());
            sig.push_str(": ");
            sig.push_str(&f.field_type.to_string());
            sig.push_str(",\n");
        }
        sig.push('}');
    }
    sig
}

pub fn format_enum_signature(e: &HirEnum) -> String {
    let mut sig = String::new();
    sig.push_str(visibility_prefix(e.visibility));
    sig.push_str("enum ");
    sig.push_str(e.unmangled_name.as_str());
    if let Some(generics) = e.generics {
        if !generics.is_empty() {
            sig.push('<');
            let g_strs: Vec<String> = generics.iter().map(format_generic).collect();
            sig.push_str(&g_strs.join(", "));
            sig.push('>');
        }
    }
    if e.variants.is_empty() {
        sig.push_str(" {}");
    } else {
        sig.push_str(" {\n");
        for v in e.variants {
            sig.push_str("    ");
            sig.push_str(v.name.as_str());
            if !v.fields.is_empty() {
                let f_strs: Vec<String> =
                    v.fields.iter().map(|f| f.field_type.to_string()).collect();
                sig.push_str("(");
                sig.push_str(f_strs.join(", ").as_str());
                sig.push_str(")");
            }
            sig.push_str(",\n");
        }
        sig.push('}');
    }
    sig
}

pub fn format_interface_signature(i: &HirInterface) -> String {
    let mut sig = String::new();
    sig.push_str(visibility_prefix(i.visibility));
    sig.push_str("interface ");
    sig.push_str(i.unmangled_name.as_str());
    if let Some(generics) = i.generics {
        if !generics.is_empty() {
            sig.push('<');
            let g_strs: Vec<String> = generics.iter().map(format_generic).collect();
            sig.push_str(&g_strs.join(", "));
            sig.push('>');
        }
    }
    sig.push_str(" { ... }");
    sig
}

pub fn format_hover_markup(code_block: &str, doc: Option<&str>) -> MarkupContent {
    let mut value = String::new();
    value.push_str("```zeta\n");
    value.push_str(code_block);
    value.push_str("\n```");
    if let Some(d) = doc {
        let trimmed = d.trim();
        if !trimmed.is_empty() {
            value.push_str("\n\n---\n\n");
            value.push_str(trimmed);
        }
    }
    MarkupContent {
        kind: MarkupKind::Markdown,
        value,
    }
}

fn find_func<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    module_idx: usize,
    name: &str,
) -> Option<(HirFunc<'a, 'bump>, usize)> {
    if let Some(func) = ctx
        .module_functions
        .get(&module_idx)
        .and_then(|f| f.get(name))
    {
        return Some((*func, func.declaring_module_idx));
    }

    let dep_graph = state.compiler.dep_graph_ref().borrow();
    for imp_idx in dep_graph.get_module_imports(module_idx) {
        if let Some(func) = ctx.module_functions.get(&imp_idx).and_then(|f| f.get(name)) {
            return Some((*func, func.declaring_module_idx));
        }
    }

    let name_id = StrId(ctx.string_pool.intern(name));
    if let Some((decl_m, _, _)) = dep_graph.resolve_item_in_module(module_idx, name_id) {
        if let Some(func) = ctx.module_functions.get(&decl_m).and_then(|f| f.get(name)) {
            return Some((*func, func.declaring_module_idx));
        }
    }

    let mut found: Vec<HirFunc<'a, 'bump>> = Vec::new();
    for cand_m in dep_graph.find_function_by_name_anywhere(name_id) {
        if let Some(f) = ctx.module_functions.get(&cand_m).and_then(|f| f.get(name)) {
            found.push(*f);
        }
    }
    if found.len() == 1 {
        let f = found[0];
        return Some((f, f.declaring_module_idx));
    }
    if found.len() > 1 {
        return None;
    }

    let mut methods: Vec<HirFunc<'a, 'bump>> = ctx
        .type_methods
        .values()
        .filter_map(|t| t.methods.get(&StrId(ctx.string_pool.intern(name))).copied())
        .collect();
    if methods.len() == 1 {
        let f = methods.remove(0);
        return Some((f, f.declaring_module_idx));
    }
    None
}

pub fn go_to_definition<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    params: &GotoDefinitionParams,
) -> Option<GotoDefinitionResponse> {
    let uri: &Uri = &params.text_document_position_params.text_document.uri;
    let position: Position = params.text_document_position_params.position;

    let doc = state.documents.get(uri)?;
    let module_idx = state.compiler.module_idx_for_path(&doc.path)?;
    let source = state.compiler.source_text(module_idx)?;
    let word = ident_at_column(line_at(&source, position.line)?, position.character)?;

    let checker_rc = state.compiler.type_checker();
    let checker = checker_rc.borrow();

    if let Some(r) = import_ref_at(&source, position) {
        let ctx = checker.context();
        let ids: Vec<StrId> = r
            .modules
            .iter()
            .map(|s| StrId(ctx.string_pool.intern(s)))
            .collect();
        let m = state
            .compiler
            .dep_graph_ref()
            .borrow()
            .resolve_module_path(&ids)?;
        if let Some(member) = &r.member {
            let mid = StrId(ctx.string_pool.intern(member));
            let (dm, idx, tag) = state
                .compiler
                .dep_graph_ref()
                .borrow()
                .resolve_item_in_module(m, mid)?;
            let (_, loc) = definition_location(
                state,
                &checker,
                SymbolId::Item {
                    module_idx: dm,
                    item_idx: idx,
                    tag,
                },
                member,
            )?;
            return Some(GotoDefinitionResponse::Scalar(loc));
        }
        return module_location(state, m).map(GotoDefinitionResponse::Scalar);
    }

    if let Some(m) = module_receiver_at(state, &checker, module_idx, &source, position, word) {
        return module_location(state, m).map(GotoDefinitionResponse::Scalar);
    }

    if let Some(symbol) = symbol_under_cursor(state, &checker, module_idx, &source, position, word)
    {
        if let Some((def_module, def)) = definition_location(state, &checker, symbol, word) {
            // Ctrl+click on the declaration itself: show who uses it.
            if def_module == module_idx && range_contains(&def.range, position) {
                return Some(declaration_alternatives(state, &checker, symbol, word, def));
            }
            return Some(GotoDefinitionResponse::Scalar(def));
        }
    }

    // Module path fallback
    let word_id = StrId(checker.context().string_pool.intern(word));
    let dep_graph = state.compiler.dep_graph_ref().borrow();
    let m = dep_graph.resolve_module_path(&[word_id])?;
    let target_path = state.compiler.path_for_module(m)?;
    Some(GotoDefinitionResponse::Scalar(Location {
        uri: path_to_uri(target_path).ok()?,
        range: lsp_types::Range::default(),
    }))
}

/// Interfaces and interface methods -> their implementations (usages when there are none).
/// Everything else -> the declaration followed by every usage.
fn declaration_alternatives<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &sentinel_typechecker::TypeChecker<'a, 'bump>,
    symbol: SymbolId,
    word: &str,
    decl: Location,
) -> GotoDefinitionResponse {
    if let Some(impls) = implementations_of(state, checker, symbol) {
        if !impls.is_empty() {
            return GotoDefinitionResponse::Array(impls);
        }
    }
    // The declaration stays in the list so a single usage opens a list instead of jumping.
    let mut all = vec![decl];
    for usage in collect_references(state, checker, symbol, false, word) {
        push_unique(&mut all, usage);
    }
    if all.len() == 1 {
        GotoDefinitionResponse::Scalar(all.remove(0))
    } else {
        GotoDefinitionResponse::Array(all)
    }
}

fn references_at<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    params: &ReferenceParams,
) -> Vec<Location> {
    let text_document = &params.text_document_position.text_document;
    let position = params.text_document_position.position;

    let Some(doc) = state.documents.get(&text_document.uri) else {
        return vec![];
    };
    let Some(module_idx) = state.compiler.module_idx_for_path(&doc.path) else {
        return vec![];
    };
    let Some(source) = state.compiler.source_text(module_idx) else {
        return vec![];
    };
    let Some(word) =
        line_at(&source, position.line).and_then(|l| ident_at_column(l, position.character))
    else {
        return vec![];
    };

    let checker_rc = state.compiler.type_checker();
    let checker = checker_rc.borrow();
    let Some(symbol) = symbol_under_cursor(state, &checker, module_idx, &source, position, word)
    else {
        return vec![];
    };
    collect_references(
        state,
        &checker,
        symbol,
        params.context.include_declaration,
        word,
    )
}

pub fn apply_change(source: &mut String, change: TextDocumentContentChangeEvent) {
    if let Some(range) = change.range {
        let start = byte_offset_in_source(source, range.start);
        let end = byte_offset_in_source(source, range.end);

        source.replace_range(start..end, &change.text);
    } else {
        // Client sent the whole document.
        *source = change.text;
    }
}

fn doc_for_symbol<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &sentinel_typechecker::TypeChecker<'a, 'bump>,
    sid: SymbolId,
) -> Option<String> {
    let (m, span) = match sid {
        SymbolId::Item {
            module_idx,
            item_idx,
            tag,
        } => {
            let stmts = &state
                .compiler
                .module_with_arena(module_idx)?
                .parse_result
                .statements;
            (module_idx, item_decl_span(stmts, item_idx, tag)?)
        }
        SymbolId::Method {
            module_idx,
            item_idx,
            method_idx,
        } => {
            let stmts = &state
                .compiler
                .module_with_arena(module_idx)?
                .parse_result
                .statements;
            (module_idx, method_decl_span(stmts, item_idx, method_idx)?)
        }
        SymbolId::Field {
            struct_name,
            field_name,
        } => {
            let struct_str = struct_name.to_string();
            let &owner_m = checker.context().struct_owner_module.get(&struct_str)?;
            let stmts = &state
                .compiler
                .module_with_arena(owner_m)?
                .parse_result
                .statements;
            let span = stmts.iter().find_map(|s| {
                let (_, s) = s.peel();
                if let Stmt::StructDecl(sd) = s {
                    if sd.name == struct_name {
                        if let Some(params) = sd.params {
                            for p in params {
                                if let ir::ast::Param::Normal(np) = p {
                                    if np.name == field_name {
                                        return Some(np.span);
                                    }
                                }
                            }
                        }
                    }
                }
                None
            })?;
            (owner_m, span)
        }
        SymbolId::Field {
            struct_name,
            field_name,
        } => field_decl_span(state, checker.context(), struct_name, field_name)?,
        SymbolId::Variant {
            enum_name,
            variant_name,
        } => variant_decl_span(state, checker.context(), enum_name, variant_name)?,
        _ => checker
            .occurrences()
            .iter()
            .find(|o| o.symbol == sid && o.is_decl)
            .map(|o| (o.module_idx, o.span))?,
    };
    let src = state.compiler.source_text(m)?;
    crate::doc_comments::doc_comment_above(&src, span.line)
}

pub fn field_decl_span<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    struct_name: StrId,
    field_name: StrId,
) -> Option<(usize, SourceSpan<'a>)> {
    let m = *ctx.struct_owner_module.get(struct_name.as_str())?;
    let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
    let dep = state.compiler.dep_graph_ref().borrow();
    stmts.iter().find_map(|s| {
        let (_, s) = s.peel();
        let Stmt::StructDecl(sd) = s else { return None };
        if dep.mangle_type_name(m, sd.name, &ctx.string_pool) != struct_name {
            return None;
        }
        sd.params?.iter().find_map(|p| match p {
            ir::ast::Param::Normal(np) if np.name == field_name => Some((m, np.span)),
            _ => None,
        })
    })
}

pub fn variant_decl_span<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    enum_name: StrId,
    variant_name: StrId,
) -> Option<(usize, SourceSpan<'a>)> {
    let m = *ctx.enum_owner_module.get(enum_name.as_str())?;
    let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
    let dep = state.compiler.dep_graph_ref().borrow();
    stmts.iter().find_map(|s| {
        let (_, s) = s.peel();
        let Stmt::EnumDecl(ed) = s else { return None };
        if dep.mangle_type_name(m, ed.name, &ctx.string_pool) != enum_name {
            return None;
        }
        ed.variants
            .iter()
            .find(|v| v.name == variant_name)
            .map(|v| (m, v.span))
    })
}

fn func_for_item<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    m: usize,
    item_idx: usize,
    tag: &str,
) -> Option<HirFunc<'a, 'bump>> {
    let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
    let span = item_decl_span(stmts, item_idx, tag)?;
    ctx.module_functions
        .get(&m)?
        .values()
        .find(|f| f.span == span)
        .copied()
}

fn method_for_symbol<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    m: usize,
    item_idx: usize,
    method_idx: usize,
) -> Option<HirFunc<'a, 'bump>> {
    let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
    let span = method_decl_span(stmts, item_idx, method_idx)?;
    ctx.type_methods
        .values()
        .flat_map(|t| t.methods.values())
        .find(|f| f.declaring_module_idx == m && f.span == span)
        .copied()
        .or_else(|| {
            ctx.interfaces
                .values()
                .filter_map(|i| i.methods)
                .flatten()
                .find(|f| f.declaring_module_idx == m && f.span == span)
                .copied()
        })
}

fn struct_in_module<'a, 'bump>(
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    m: usize,
    unmangled: &str,
) -> Option<HirStruct<'a, 'bump>> {
    ctx.structs
        .iter()
        .find(|(k, s)| {
            s.unmangled_name == unmangled && ctx.struct_owner_module.get(k.as_str()) == Some(&m)
        })
        .map(|(_, s)| *s)
}

fn enum_in_module<'a, 'bump>(
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    m: usize,
    unmangled: &str,
) -> Option<HirEnum<'a, 'bump>> {
    ctx.enums
        .iter()
        .find(|(k, e)| {
            e.unmangled_name == unmangled && ctx.enum_owner_module.get(k.as_str()) == Some(&m)
        })
        .map(|(_, e)| *e)
}

fn interface_in_module<'a, 'bump>(
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    m: usize,
    unmangled: &str,
) -> Option<HirInterface<'a, 'bump>> {
    ctx.interfaces
        .iter()
        .find(|(k, i)| {
            i.unmangled_name == unmangled && ctx.interface_owner_module.get(k.as_str()) == Some(&m)
        })
        .map(|(_, i)| *i)
}

/// Items compare by declaration site. The tag ("func_sig" vs "func_body", ...) only says which
/// facet of the item an occurrence was recorded for.
fn same_symbol(a: SymbolId, b: SymbolId) -> bool {
    match (a, b) {
        (
            SymbolId::Item {
                module_idx: m1,
                item_idx: i1,
                ..
            },
            SymbolId::Item {
                module_idx: m2,
                item_idx: i2,
                ..
            },
        ) => m1 == m2 && i1 == i2,
        _ => a == b,
    }
}

/// The symbol under the cursor: an occurrence, else a declaration name found in the AST
/// (declarations don't always have occurrences), else a name visible from this module.
pub fn symbol_under_cursor<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &sentinel_typechecker::TypeChecker<'a, 'bump>,
    module_idx: usize,
    source: &str,
    position: Position,
    word: &str,
) -> Option<SymbolId> {
    let (line, column) = to_span_coords(source, position);
    if let Some(occ) = tightest_occurrence(checker.occurrences(), module_idx, line, column) {
        if occ.name.as_str() == word {
            return Some(occ.symbol);
        }
    }
    let ctx = checker.context();
    if let Some(symbol) = declaration_symbol_at(state, ctx, module_idx, source, position, word) {
        return Some(symbol);
    }
    let word_id = StrId(ctx.string_pool.intern(word));
    let (m, item_idx, tag) = resolve_visible_item(state, module_idx, word_id)?;
    Some(SymbolId::Item {
        module_idx: m,
        item_idx,
        tag,
    })
}

fn declaration_symbol_at<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    module_idx: usize,
    source: &str,
    position: Position,
    word: &str,
) -> Option<SymbolId> {
    let (line, column) = to_span_coords(source, position);
    let on_name = |span: SourceSpan| span.line != 0 && span_contains(&span, line, column);
    let on_ident_in = |span: SourceSpan, name: StrId| {
        span.line != 0
            && name.as_str() == word
            && range_contains(
                &find_ident_range_on_line(source, span.line, word, span),
                position,
            )
    };

    let stmts = &state
        .compiler
        .module_with_arena(module_idx)?
        .parse_result
        .statements;
    let dep = state.compiler.dep_graph_ref().borrow();

    for (item_idx, stmt) in stmts.iter().enumerate() {
        let (_, stmt) = stmt.peel();
        let item = |tag: &'static str| SymbolId::Item {
            module_idx,
            item_idx,
            tag,
        };
        match stmt {
            Stmt::TypeAliasDecl(t) if on_name(t.name_span) => return Some(item("type")),
            Stmt::FuncDecl(f) if on_name(f.name_span) => return Some(item("func_sig")),
            Stmt::Const(c) if on_ident_in(c.span, c.ident) => return Some(item("const")),
            Stmt::StructDecl(s) => {
                if on_name(s.name_span) {
                    return Some(item("type"));
                }
                for p in s.params.into_iter().flatten() {
                    if let ir::ast::Param::Normal(np) = p {
                        if on_ident_in(np.span, np.name) {
                            return Some(SymbolId::Field {
                                struct_name: dep.mangle_type_name(
                                    module_idx,
                                    s.name,
                                    &ctx.string_pool,
                                ),
                                field_name: np.name,
                            });
                        }
                    }
                }
            }
            Stmt::EnumDecl(e) => {
                if on_name(e.name_span) {
                    return Some(item("type"));
                }
                for v in e.variants.iter() {
                    if on_ident_in(v.span, v.name) {
                        return Some(SymbolId::Variant {
                            enum_name: dep.mangle_type_name(module_idx, e.name, &ctx.string_pool),
                            variant_name: v.name,
                        });
                    }
                }
            }
            Stmt::InterfaceDecl(i) => {
                if on_name(i.name_span) {
                    return Some(item("trait"));
                }
                let methods = i.methods.unwrap_or(&[]);
                if let Some(method_idx) = methods.iter().position(|m| on_name(m.name_span)) {
                    return Some(SymbolId::Method {
                        module_idx,
                        item_idx,
                        method_idx,
                    });
                }
            }
            Stmt::ImplDecl(i) => {
                let methods = i.methods.unwrap_or(&[]);
                if let Some(method_idx) = methods.iter().position(|m| on_name(m.name_span)) {
                    return Some(SymbolId::Method {
                        module_idx,
                        item_idx,
                        method_idx,
                    });
                }
            }
            _ => {}
        }
    }
    None
}

/// Where `symbol` is declared (the module it lives in, and the name's range).
/// This is the old per-symbol blocks of `go_to_definition`, folded into one place.
pub fn definition_location<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &sentinel_typechecker::TypeChecker<'a, 'bump>,
    symbol: SymbolId,
    word: &str,
) -> Option<(usize, Location)> {
    let ctx = checker.context();
    let (m, span) = match symbol {
        SymbolId::Variant {
            enum_name,
            variant_name,
        } => variant_decl_span(state, ctx, enum_name, variant_name)?,
        SymbolId::Item {
            module_idx: m,
            item_idx,
            tag,
        } => {
            let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
            (m, item_decl_span(stmts, item_idx, tag)?)
        }
        SymbolId::Method {
            module_idx: m,
            item_idx,
            method_idx,
        } => {
            let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
            (m, method_decl_span(stmts, item_idx, method_idx)?)
        }
        SymbolId::Field {
            struct_name,
            field_name,
        } => field_decl_span(state, ctx, struct_name, field_name)?,
        SymbolId::Local(..) => {
            let d = checker
                .occurrences()
                .iter()
                .find(|o| o.symbol == symbol && o.is_decl)?;
            (d.module_idx, d.span)
        }
    };
    let target_path = state.compiler.path_for_module(m)?;
    let target_source = state.compiler.source_text(m)?;
    let range = find_ident_range_on_line(&target_source, span.line, word, span);
    Some((
        m,
        Location {
            uri: path_to_uri(target_path).ok()?,
            range,
        },
    ))
}

pub fn push_unique(locations: &mut Vec<Location>, loc: Location) {
    if !locations.contains(&loc) {
        locations.push(loc);
    }
}

pub fn collect_references<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &sentinel_typechecker::TypeChecker<'a, 'bump>,
    symbol: SymbolId,
    include_declaration: bool,
    word: &str,
) -> Vec<Location> {
    let mut locations = Vec::new();

    // Declarations are not always recorded as occurrences; add them from the AST.
    let has_decl = checker
        .occurrences()
        .iter()
        .any(|o| o.is_decl && same_symbol(o.symbol, symbol));
    if include_declaration && !has_decl {
        if let Some((_, def)) = definition_location(state, checker, symbol, word) {
            push_unique(&mut locations, def);
        }
    }

    for o in checker.occurrences() {
        if !same_symbol(o.symbol, symbol) || (o.is_decl && !include_declaration) {
            continue;
        }
        let Some(m_source) = state.compiler.source_text(o.module_idx) else {
            continue;
        };
        let Some(m_path) = state.compiler.path_for_module(o.module_idx) else {
            continue;
        };
        let Ok(uri) = path_to_uri(m_path) else {
            continue;
        };
        let range = span_to_range(
            &m_source,
            o.span.line,
            o.span.column,
            o.span.end_line,
            o.span.end_column,
        );
        push_unique(&mut locations, Location { uri, range });
    }
    locations
}

fn visibility_prefix(v: Visibility) -> &'static str {
    match v {
        Visibility::Public => "public ",
        Visibility::Internal => "internal ",
        Visibility::Module => "module ",
        Visibility::Private => "", // the default, so no keyword
    }
}

const KEYWORDS: &[&str] = &[
    "func",
    "let",
    "mut",
    "const",
    "static",
    "struct",
    "enum",
    "interface",
    "impl",
    "by",
    "import",
    "package",
    "public",
    "private",
    "internal",
    "unsafe",
    "extern",
    "inline",
    "noinline",
    "if",
    "else",
    "while",
    "for",
    "match",
    "return",
    "break",
    "continue",
    "defer",
    "this",
    "null",
    "true",
    "false",
    "undefined",
    "uninit",
    "dyn",
    "as",
];
const PRIMITIVES: &[&str] = &[
    "i8", "i16", "i32", "i64", "i128", "u8", "u16", "u32", "u64", "u128", "usize", "isize", "f32",
    "f64", "bool", "char", "str", "void",
];

fn invalidate_completions<'a, 'bump>(
    compiler: &zeta_compiler_api::Compiler<'a, 'bump>,
    cache: &mut crate::state::CompletionCache,
    path: &PathBuf,
    structural: bool,
) {
    if structural {
        cache.invalidate_structure();
    }
    if let Some(m) = compiler.module_idx_for_path(path) {
        cache.invalidate_module(m);
    }
}

pub fn completions_at<'a, 'bump>(
    state: &mut ServerState<'a, 'bump>,
    params: &CompletionParams,
) -> Vec<CompletionItem> {
    let uri = &params.text_document_position.text_document.uri;
    let position = params.text_document_position.position;

    let Some(doc) = state.documents.get(uri) else {
        return vec![];
    };
    let Some(module_idx) = state.compiler.module_idx_for_path(&doc.path) else {
        return vec![];
    };
    let Some(source) = state.compiler.source_text(module_idx) else {
        return vec![];
    };
    let Some(line_text) = line_at(&source, position.line) else {
        return vec![];
    };
    let cursor = byte_offset_from_utf16(line_text, position.character);
    let before = &line_text[..cursor];

    if let Some(i) = before.find("//") {
        if before[..i].matches('"').count() % 2 == 0 {
            return vec![];
        }
    }
    if before.matches('"').count() % 2 == 1 {
        return vec![];
    }

    let checker_rc = state.compiler.type_checker();
    let checker = checker_rc.borrow();

    if let Some(items) = import_completions(state, &checker, line_text, cursor) {
        return items;
    }

    let (head, typed) = split_typed(before);

    if let Some(base) = head.strip_suffix('.') {
        return match receiver_chain(base) {
            Some(segs) => member_items(state, &checker, module_idx, &source, position, &segs),
            None => vec![],
        };
    }
    if head.ends_with(':') {
        return vec![]; // trigger char of a type annotation, nothing to offer yet
    }
    if let Some(items) =
        struct_init_items(state, &checker, module_idx, &source, position, typed.len())
    {
        return items;
    }

    let (line, column) = to_span_coords(&source, position);
    global_items(
        state,
        &checker,
        module_idx,
        line,
        column,
        (!typed.is_empty()).then_some(typed),
    )
}

fn ident_start(s: &str) -> usize {
    s.char_indices()
        .rev()
        .take_while(|(_, c)| c.is_alphanumeric() || *c == '_')
        .last()
        .map(|(i, _)| i)
        .unwrap_or(s.len())
}

fn split_typed(before: &str) -> (&str, &str) {
    let i = ident_start(before);
    (&before[..i], &before[i..])
}

fn receiver_chain(base: &str) -> Option<Vec<&str>> {
    let mut segs = Vec::new();
    let mut rest = base;
    loop {
        let id = &rest[ident_start(rest)..];
        if id.is_empty() {
            return None;
        }
        segs.push(id);
        rest = &rest[..rest.len() - id.len()];
        match rest.strip_suffix('.') {
            Some(r) => rest = r,
            None => break,
        }
    }
    segs.reverse();
    Some(segs)
}

fn mk(label: &str, kind: K, detail: Option<String>, sort: &str) -> CompletionItem {
    CompletionItem {
        label: label.to_string(),
        kind: Some(kind),
        detail,
        sort_text: Some(format!("{sort}{label}")),
        ..Default::default()
    }
}

pub fn format_func_signature(func: &HirFunc) -> String {
    let mut sig = String::new();

    sig.push_str(visibility_prefix(func.function_metadata.visibility));

    if func.function_metadata.func_safety == FuncSafety::Unsafe {
        sig.push_str("unsafe ");
    }

    match func.function_metadata.extern_modifier {
        ExternModifier::None => {}
        ExternModifier::Abi(abi) => {
            sig.push_str(&format!("extern \"{}\" ", abi));
        }
    }

    match func.function_metadata.inline_modifier {
        InlineModifier::Inline => sig.push_str("inline "),
        InlineModifier::Noinline => sig.push_str("noinline "),
        InlineModifier::None => {}
    }

    sig.push_str("func ");
    sig.push_str(func.unmangled_name.as_str());

    if let Some(generics) = func.generics {
        if !generics.is_empty() {
            sig.push('<');
            let g_strs: Vec<String> = generics.iter().map(format_generic).collect();
            sig.push_str(&g_strs.join(", "));
            sig.push('>');
        }
    }

    sig.push('(');
    if let Some(params) = func.params {
        let p_strs: Vec<String> = params
            .iter()
            .map(|p| match p {
                HirParam::Normal {
                    name, param_type, ..
                } => format!("{}: {}", name, param_type),
                HirParam::This { kind, .. } => format!("{}", kind),
            })
            .collect();
        sig.push_str(&p_strs.join(", "));
    }
    sig.push(')');

    if let Some(ret) = &func.return_type {
        sig.push_str(&format!(": {}", ret));
    }

    sig.push(';');
    sig
}

fn func_item(label: &str, func: &HirFunc, kind: K, sort: &str) -> CompletionItem {
    let args: Vec<String> = func
        .params
        .unwrap_or(&[])
        .iter()
        .filter_map(|p| match p {
            HirParam::Normal { name, .. } => Some(name.to_string()),
            HirParam::This { .. } => None,
        })
        .collect();
    let snippet = args
        .iter()
        .enumerate()
        .map(|(i, n)| format!("${{{}:{}}}", i + 1, n))
        .collect::<Vec<_>>()
        .join(", ");
    let mut it = mk(label, kind, Some(format_func_signature(func)), sort);
    it.insert_text = Some(format!("{label}({snippet})"));
    it.insert_text_format = Some(InsertTextFormat::SNIPPET);
    it
}

fn push(items: &mut Vec<CompletionItem>, seen: &mut HashSet<String>, it: CompletionItem) {
    if seen.insert(it.label.clone()) {
        items.push(it);
    }
}

fn decl_line(src: &str, line: usize) -> Option<String> {
    let l = line_at(src, line.checked_sub(1)? as u32)?;
    Some(
        l.trim()
            .trim_end_matches('{')
            .trim()
            .trim_end_matches(';')
            .trim()
            .to_string(),
    )
}

fn item_from_stmt<'a, 'bump>(
    ctx: &TypeContext<'a, 'bump>,
    m: usize,
    stmt: Stmt<'a, 'bump>,
    from: usize,
    src: &str,
) -> Option<CompletionItem> {
    let (name, kind, private, span) = match stmt {
        Stmt::FuncDecl(f) => (
            f.name,
            K::FUNCTION,
            f.function_metadata.visibility == Visibility::Private,
            f.span,
        ),
        Stmt::StructDecl(s) => (
            s.name,
            K::STRUCT,
            s.visibility == Visibility::Private,
            s.span,
        ),
        Stmt::EnumDecl(e) => (e.name, K::ENUM, e.visibility == Visibility::Private, e.span),
        Stmt::InterfaceDecl(i) => (
            i.name,
            K::INTERFACE,
            i.visibility == Visibility::Private,
            i.span,
        ),
        Stmt::TypeAliasDecl(t) => (t.name, K::TYPE_PARAMETER, false, t.span),
        Stmt::Const(c) => (c.ident, K::CONSTANT, false, c.span),
        _ => return None,
    };
    if private && m != from {
        return None;
    }
    if matches!(stmt, Stmt::FuncDecl(_)) {
        let func = ctx
            .module_functions
            .get(&m)
            .and_then(|t| t.get(name.as_str()))
            .copied();
        if let Some(func) = func {
            return Some(func_item(name.as_str(), &func, K::FUNCTION, "1"));
        }
    }
    Some(mk(name.as_str(), kind, decl_line(src, span.line), "1"))
}

fn module_items<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &TypeContext<'a, 'bump>,
    m: usize,
    from: usize,
) -> Vec<CompletionItem> {
    let Some(arena) = state.compiler.module_with_arena(m) else {
        return vec![];
    };
    let Some(src) = state.compiler.source_text(m) else {
        return vec![];
    };
    arena
        .parse_result
        .statements
        .iter()
        .filter_map(|s| item_from_stmt(ctx, m, s.peel().1, from, &src))
        .collect()
}

fn item_at<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &TypeContext<'a, 'bump>,
    m: usize,
    idx: usize,
    from: usize,
) -> Option<CompletionItem> {
    let arena = state.compiler.module_with_arena(m)?;
    let src = state.compiler.source_text(m)?;
    let stmt = arena.parse_result.statements.get(idx)?.peel().1;
    item_from_stmt(ctx, m, stmt, from, &src)
}

/// Names this module imported: `import a.b` -> module `b`, `import a.b.Name` -> `Name`.
fn import_items<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &TypeContext<'a, 'bump>,
    module_idx: usize,
) -> Vec<CompletionItem> {
    let Some(arena) = state.compiler.module_with_arena(module_idx) else {
        return vec![];
    };
    let dep = state.compiler.dep_graph_ref().borrow();
    let mut out = Vec::new();
    for s in arena.parse_result.statements.iter() {
        let Stmt::Import(i) = s.peel().1 else {
            continue;
        };
        match i.path.member {
            Some(member) => {
                let Some(target) = dep.resolve_module_path(i.path.path) else {
                    continue;
                };
                let real = dep.canonical_member_module(target, member);
                let hit = dep
                    .resolve_item_in_module(real, member)
                    .or_else(|| dep.resolve_item_in_module(target, member));
                if let Some((dm, idx, _)) = hit {
                    out.extend(item_at(state, ctx, dm, idx, module_idx));
                }
            }
            None => {
                if let Some(&last) = i.path.path.last() {
                    let full = i
                        .path
                        .path
                        .iter()
                        .map(|s| s.as_str())
                        .collect::<Vec<_>>()
                        .join("::");
                    out.push(mk(
                        last.as_str(),
                        K::MODULE,
                        Some(format!("import {full}")),
                        "1",
                    ));
                }
            }
        }
    }
    out
}

fn module_for_alias<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &TypeContext<'a, 'bump>,
    module_idx: usize,
    alias: &str,
) -> Option<usize> {
    let qid = StrId(ctx.string_pool.intern(alias));
    let dep = state.compiler.dep_graph_ref().borrow();
    let stmts = &state
        .compiler
        .module_with_arena(module_idx)?
        .parse_result
        .statements;
    stmts
        .iter()
        .find_map(|s| match s.peel().1 {
            Stmt::Import(i) if i.path.member.is_none() && i.path.path.last() == Some(&qid) => {
                dep.resolve_module_path(i.path.path)
            }
            _ => None,
        })
        .or_else(|| {
            dep.auto_import_packages()
                .into_iter()
                .find(|p| p.last() == Some(&qid))
                .and_then(|p| dep.resolve_module_path(&p))
        })
        .or_else(|| dep.resolve_module_path(&[qid]))
}

fn import_completions<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &TypeChecker<'a, 'bump>,
    line: &str,
    cursor: usize,
) -> Option<Vec<CompletionItem>> {
    let trimmed = line.trim_start();
    let kw = ["import ", "package "]
        .into_iter()
        .find(|k| trimmed.starts_with(*k))?;
    let body_start = line.len() - trimmed.len() + kw.len();
    if cursor < body_start {
        return None;
    }
    let body = &line[body_start..cursor];
    let ctx = checker.context();
    let dep = state.compiler.dep_graph_ref().borrow();

    // `a::b.Mem`: members of module a::b
    if let Some((path, _)) = body.split_once('.') {
        let ids: Vec<StrId> = path
            .split("::")
            .filter(|s| !s.is_empty())
            .map(|s| StrId(ctx.string_pool.intern(s)))
            .collect();
        let m = dep.resolve_module_path(&ids)?;
        return Some(module_items(state, ctx, m, usize::MAX));
    }

    // `a::b::par`: next path segment of every known package
    let done = body.rfind("::").map(|i| &body[..i]).unwrap_or("");
    let prefix = if done.is_empty() {
        String::new()
    } else {
        format!("{done}::")
    };
    let mut seen = HashSet::default();
    let mut items = Vec::new();
    for m in all_modules(checker) {
        let Some(pkg) = dep.get_module_package(m) else {
            continue;
        };
        let pkg = pkg.as_str();
        let Some(rest) = pkg.strip_prefix(prefix.as_str()) else {
            continue;
        };
        let seg = rest.split("::").next().unwrap_or("");
        if !seg.is_empty() {
            push(
                &mut items,
                &mut seen,
                mk(seg, K::MODULE, Some(pkg.to_string()), "0"),
            );
        }
    }
    Some(items)
}

enum Recv<'a, 'bump> {
    Value(HirType<'a, 'bump>),
    Type(usize, usize),
    Module(usize),
}

fn peel_type<'a, 'bump>(mut ty: HirType<'a, 'bump>) -> (HirType<'a, 'bump>, bool) {
    let mut owned_slice = false;
    loop {
        match ty.kind {
            HirTypeKind::Ref { inner, .. }
            | HirTypeKind::SafePointer { inner, .. }
            | HirTypeKind::UnsafePointer { inner, .. }
            | HirTypeKind::Nullable(inner) => ty = *inner,
            HirTypeKind::OwnedPointer { inner, .. } => {
                owned_slice = matches!(inner.kind, HirTypeKind::Slice(_));
                ty = *inner;
            }
            _ => return (ty, owned_slice),
        }
    }
}

fn field_type<'a, 'bump>(
    ctx: &TypeContext<'a, 'bump>,
    ty: HirType<'a, 'bump>,
    field: &str,
) -> Option<HirType<'a, 'bump>> {
    let (p, _) = peel_type(ty);
    let HirTypeKind::Struct { name, .. } = p.kind else {
        return None;
    };
    ctx.structs
        .get(name.as_str())?
        .fields
        .iter()
        .find(|f| f.name.as_str() == field)
        .map(|f| f.field_type)
}

fn member_items<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &TypeChecker<'a, 'bump>,
    module_idx: usize,
    source: &str,
    position: lsp_types::Position,
    segs: &[&str],
) -> Vec<CompletionItem> {
    let ctx = checker.context();
    let (line, column) = to_span_coords(source, position);
    let first = segs[0];

    // newest occurrence of `first` before the cursor (local, param, `this`, field, type name)
    let occ = checker
        .occurrences()
        .iter()
        .filter(|o| {
            o.module_idx == module_idx
                && o.name.as_str() == first
                && (o.span.line < line || (o.span.line == line && o.span.column <= column))
        })
        .max_by_key(|o| (o.span.line, o.span.column));

    let mut recv: Option<Recv> = match occ.map(|o| (o.symbol, o.ty)) {
        Some((
            SymbolId::Item {
                module_idx: m,
                item_idx,
                tag: "type" | "trait",
            },
            _,
        )) => Some(Recv::Type(m, item_idx)),
        Some((
            SymbolId::Item {
                tag: "func_sig" | "func_body",
                ..
            },
            _,
        ))
        | Some((SymbolId::Method { .. }, _)) => None,
        Some((_, ty)) => Some(Recv::Value(ty)),
        None => None,
    };
    if recv.is_none() {
        if let Some(m) = module_for_alias(state, ctx, module_idx, first) {
            recv = Some(Recv::Module(m));
        } else if let Some((m, idx, tag)) =
            resolve_visible_item(state, module_idx, StrId(ctx.string_pool.intern(first)))
        {
            if matches!(tag, "type" | "trait") {
                recv = Some(Recv::Type(m, idx));
            }
        }
    }
    let Some(mut cur) = recv else { return vec![] };

    for seg in &segs[1..] {
        cur = match cur {
            Recv::Value(ty) => match field_type(ctx, ty, seg) {
                Some(t) => Recv::Value(t),
                None => return vec![],
            },
            Recv::Module(m) => {
                let id = StrId(ctx.string_pool.intern(seg));
                let dep = state.compiler.dep_graph_ref().borrow();
                match dep.resolve_item_in_module(m, id) {
                    Some((dm, idx, tag)) if matches!(tag, "type" | "trait") => Recv::Type(dm, idx),
                    _ => return vec![],
                }
            }
            Recv::Type(..) => return vec![],
        };
    }

    match cur {
        Recv::Module(m) => module_items(state, ctx, m, module_idx),
        Recv::Type(m, idx) => static_items(state, ctx, m, idx, module_idx),
        Recv::Value(ty) => instance_items(ctx, module_idx, ty),
    }
}

fn method_items<'a, 'bump>(
    ctx: &TypeContext<'a, 'bump>,
    target: StrId,
    want_this: bool,
    from: usize,
    items: &mut Vec<CompletionItem>,
    seen: &mut HashSet<String>,
) {
    let Some(table) = ctx.type_methods.get(&target) else {
        return;
    };
    for (mname, func) in &table.methods {
        let has_this = matches!(
            func.params.and_then(|p| p.first()),
            Some(HirParam::This { .. })
        );
        if has_this != want_this {
            continue;
        }
        if func.function_metadata.visibility == Visibility::Private
            && func.declaring_module_idx != from
        {
            continue;
        }
        push(items, seen, func_item(mname.as_str(), func, K::METHOD, "1"));
    }
}

fn iface_items<'a, 'bump>(
    ctx: &TypeContext<'a, 'bump>,
    iface: &str,
    only_defaults: bool,
    items: &mut Vec<CompletionItem>,
    seen: &mut HashSet<String>,
) {
    let Some(i) = ctx.interfaces.get(iface) else {
        return;
    };
    for m in i.methods.unwrap_or(&[]) {
        let has_this = matches!(
            m.params.and_then(|p| p.first()),
            Some(HirParam::This { .. })
        );
        if !has_this || (only_defaults && m.body.is_none()) {
            continue;
        }
        push(
            items,
            seen,
            func_item(m.unmangled_name.as_str(), m, K::METHOD, "1"),
        );
    }
}

fn prim_name(kind: &HirTypeKind) -> Option<&'static str> {
    use HirTypeKind as T;
    Some(match kind {
        T::I8 => "i8",
        T::I16 => "i16",
        T::I32 => "i32",
        T::I64 => "i64",
        T::I128 => "i128",
        T::U8 => "u8",
        T::U16 => "u16",
        T::U32 => "u32",
        T::U64 => "u64",
        T::U128 => "u128",
        T::Usize => "usize",
        T::Isize => "isize",
        T::F32 => "f32",
        T::F64 => "f64",
        T::Boolean => "bool",
        T::Char => "char",
        T::String => "str",
        _ => return None,
    })
}

/// `type_methods` keys for user `impl`s on primitives / slices.
fn primitive_keys<'a, 'bump>(ctx: &TypeContext<'a, 'bump>, ty: HirType<'a, 'bump>) -> Vec<StrId> {
    let mut keys = Vec::new();
    if let Some(n) = prim_name(&ty.kind) {
        keys.push(StrId(ctx.string_pool.intern(n)));
    }
    if let HirTypeKind::Slice(e) | HirTypeKind::Array(e, _) = ty.kind {
        keys.push(StrId(ctx.string_pool.intern("slice")));
        let elem = prim_name(&e.kind)
            .map(str::to_string)
            .or_else(|| match e.kind {
                HirTypeKind::Struct { name, .. } => Some(name.to_string()),
                _ => None,
            });
        if let Some(elem) = elem {
            keys.push(StrId(ctx.string_pool.intern(&format!("slice_{elem}"))));
        }
    }
    keys
}

fn instance_items<'a, 'bump>(
    ctx: &TypeContext<'a, 'bump>,
    from: usize,
    ty: HirType<'a, 'bump>,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let mut seen = HashSet::default();
    let (p, owned_slice) = peel_type(ty);
    match p.kind {
        HirTypeKind::Struct { name, .. } => {
            if let Some(def) = ctx.structs.get(name.as_str()) {
                let owner = ctx.struct_owner_module.get(name.as_str()).copied();
                for f in def.fields {
                    if f.visibility == Visibility::Private && owner != Some(from) {
                        continue;
                    }
                    push(
                        &mut items,
                        &mut seen,
                        mk(
                            f.name.as_str(),
                            K::FIELD,
                            Some(f.field_type.to_string()),
                            "0",
                        ),
                    );
                }
            }
            method_items(ctx, name, true, from, &mut items, &mut seen);
            for (_, i) in ctx.interfaces.iter() {
                if ctx.struct_implements(&name, &i.name) {
                    iface_items(ctx, i.name.as_str(), true, &mut items, &mut seen);
                }
            }
        }
        HirTypeKind::Enum { name, .. } => {
            method_items(ctx, name, true, from, &mut items, &mut seen)
        }
        HirTypeKind::DynInterface { name, .. } => {
            iface_items(ctx, name.as_str(), false, &mut items, &mut seen)
        }
        HirTypeKind::Dyn { bounds } => {
            for b in bounds {
                if let HirTypeKind::DynInterface { name, .. } | HirTypeKind::Struct { name, .. } =
                    b.kind
                {
                    iface_items(ctx, name.as_str(), false, &mut items, &mut seen);
                }
            }
        }
        HirTypeKind::Slice(_) | HirTypeKind::Array(..) => {
            push(
                &mut items,
                &mut seen,
                mk("len", K::FIELD, Some("usize".into()), "0"),
            );
            push(
                &mut items,
                &mut seen,
                mk("ptr", K::FIELD, Some("[*]T".into()), "0"),
            );
            if owned_slice {
                push(
                    &mut items,
                    &mut seen,
                    mk("cap", K::FIELD, Some("usize".into()), "0"),
                );
            }
        }
        _ => {}
    }
    for key in primitive_keys(ctx, p) {
        method_items(ctx, key, true, from, &mut items, &mut seen);
    }
    items
}

fn static_items<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &TypeContext<'a, 'bump>,
    m: usize,
    idx: usize,
    from: usize,
) -> Vec<CompletionItem> {
    let mut items = Vec::new();
    let mut seen = HashSet::default();
    let Some(key) = decl_hir_key(state, ctx, m, idx) else {
        return items;
    };
    if let Some(e) = ctx.enums.get(key.as_str()) {
        for v in e.variants {
            let payload = if v.fields.is_empty() {
                String::new()
            } else {
                let tys: Vec<String> = v.fields.iter().map(|f| f.field_type.to_string()).collect();
                format!("({})", tys.join(", "))
            };
            push(
                &mut items,
                &mut seen,
                mk(
                    v.name.as_str(),
                    K::ENUM_MEMBER,
                    Some(format!("{}{}", e.unmangled_name, payload)),
                    "0",
                ),
            );
        }
    }
    method_items(ctx, key, false, from, &mut items, &mut seen);
    items
}

fn struct_init_items<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &TypeChecker<'a, 'bump>,
    module_idx: usize,
    source: &str,
    position: lsp_types::Position,
    typed_len: usize,
) -> Option<Vec<CompletionItem>> {
    let ctx = checker.context();
    let off = byte_offset_in_source(source, position).checked_sub(typed_len)?;
    let before = &source[..off];
    let window = before
        .char_indices()
        .rev()
        .take(4000)
        .last()
        .map(|(i, _)| i)
        .unwrap_or(0);
    let trimmed = before[window..].trim_end();
    if !(trimmed.ends_with('{') || trimmed.ends_with(',')) {
        return None;
    }

    let (mut depth, mut open) = (0usize, None);
    for (i, c) in trimmed.char_indices().rev() {
        match c {
            '}' | ')' | ']' => depth += 1,
            '{' | '(' | '[' => {
                if depth == 0 {
                    if c == '{' {
                        open = Some(i);
                    }
                    break;
                }
                depth -= 1;
            }
            _ => {}
        }
    }
    let open = open?;

    let mut head = trimmed[..open].trim_end();
    if head.ends_with('>') {
        if let Some(i) = head.rfind('<') {
            head = head[..i].trim_end();
        }
    }
    let id = &head[ident_start(head)..];
    if id.is_empty() {
        return None;
    }
    let prev = head[..head.len() - id.len()].trim_end();
    let prev_word = &prev[ident_start(prev)..];
    if matches!(
        prev_word,
        "impl"
            | "struct"
            | "enum"
            | "interface"
            | "by"
            | "for"
            | "while"
            | "if"
            | "match"
            | "else"
            | "func"
            | "unsafe"
    ) {
        return None;
    }

    let (m, idx, tag) = resolve_visible_item(state, module_idx, StrId(ctx.string_pool.intern(id)))?;
    if tag != "type" {
        return None;
    }
    let key = decl_hir_key(state, ctx, m, idx)?;
    let def = ctx.structs.get(key.as_str())?;
    let done: HashSet<&str> = trimmed[open + 1..]
        .split(',')
        .filter_map(|p| p.split_once(':').map(|(n, _)| n.trim()))
        .collect();
    Some(
        def.fields
            .iter()
            .filter(|f| !done.contains(f.name.as_str()))
            .map(|f| {
                let mut it = mk(
                    f.name.as_str(),
                    K::FIELD,
                    Some(f.field_type.to_string()),
                    "0",
                );
                it.insert_text = Some(format!("{}: ", f.name));
                it
            })
            .collect(),
    )
}

pub fn resolve_visible_item<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    module_idx: usize,
    name: StrId,
) -> Option<(usize, usize, &'static str)> {
    let dep = state.compiler.dep_graph_ref().borrow();
    let usable = |h: &(usize, usize, &'static str)| !matches!(h.2, "module" | "impl");

    if let Some(hit) = dep.resolve_item_in_module(module_idx, name).filter(usable) {
        return Some(hit);
    }
    let mut hits = dep
        .get_module_imports(module_idx)
        .into_iter()
        .filter_map(|m| dep.resolve_item_in_module(m, name))
        .filter(usable);
    let first = hits.next()?;
    if hits.any(|h| h != first) {
        return None;
    }
    Some(first)
}

/// Mangled (HIR) key of a struct/enum/interface declaration.
fn decl_hir_key<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    ctx: &sentinel_typechecker::TypeContext<'a, 'bump>,
    m: usize,
    item_idx: usize,
) -> Option<StrId> {
    let stmts = &state.compiler.module_with_arena(m)?.parse_result.statements;
    let (_, stmt) = stmts.get(item_idx)?.peel();
    let name = match stmt {
        Stmt::StructDecl(s) => s.name,
        Stmt::EnumDecl(e) => e.name,
        Stmt::InterfaceDecl(i) => i.name,
        _ => return None,
    };
    let dep = state.compiler.dep_graph_ref().borrow();
    Some(dep.mangle_type_name(m, name, &ctx.string_pool))
}

fn local_items<'a, 'bump>(
    state: &ServerState<'a, 'bump>,
    checker: &TypeChecker<'a, 'bump>,
    module_idx: usize,
    line: usize,
    column: usize,
) -> Vec<CompletionItem> {
    let Some(arena) = state.compiler.module_with_arena(module_idx) else {
        return vec![];
    };
    // enclosing function = nearest function/method starting at or before the cursor line
    let mut starts: Vec<usize> = Vec::new();
    for s in arena.parse_result.statements.iter() {
        match s.peel().1 {
            Stmt::FuncDecl(f) => starts.push(f.span.line),
            Stmt::ImplDecl(i) => {
                starts.extend(i.methods.unwrap_or(&[]).iter().map(|m| m.span.line))
            }
            _ => {}
        }
    }
    let start = starts
        .iter()
        .copied()
        .filter(|&l| l <= line)
        .max()
        .unwrap_or(0);
    let end = starts
        .iter()
        .copied()
        .filter(|&l| l > line)
        .min()
        .unwrap_or(usize::MAX);

    let mut best: HashMap<String, (usize, usize, String)> = HashMap::new();
    for o in checker.occurrences() {
        if o.module_idx != module_idx || !o.is_decl || !matches!(o.symbol, SymbolId::Local(_)) {
            continue;
        }
        let (l, c) = (o.span.line, o.span.column);
        if l < start || l >= end || (l, c) > (line, column) || o.name.as_str() == "_" {
            continue;
        }
        let e = best
            .entry(o.name.as_str().to_string())
            .or_insert((l, c, o.ty.to_string()));
        if (l, c) >= (e.0, e.1) {
            *e = (l, c, o.ty.to_string());
        }
    }
    best.into_iter()
        .map(|(n, (_, _, ty))| mk(&n, K::VARIABLE, Some(ty), "0"))
        .collect()
}

fn global_items<'a, 'bump>(
    state: &mut ServerState<'a, 'bump>,
    checker: &TypeChecker<'a, 'bump>,
    module_idx: usize,
    line: usize,
    column: usize,
    auto_import: Option<&str>,
) -> Vec<CompletionItem> {
    let ctx = checker.context();
    let mut items = Vec::new();
    let mut seen = HashSet::default();

    for it in local_items(state, checker, module_idx, line, column) {
        push(&mut items, &mut seen, it);
    }
    for it in module_items(state, ctx, module_idx, module_idx) {
        push(&mut items, &mut seen, it);
    }
    for it in import_items(state, ctx, module_idx) {
        push(&mut items, &mut seen, it);
    }

    let autos = state
        .compiler
        .dep_graph_ref()
        .borrow()
        .auto_import_packages();
    for segs in autos {
        let Some(&last) = segs.last() else { continue };
        let target = state
            .compiler
            .dep_graph_ref()
            .borrow()
            .resolve_module_path(&segs);
        let Some(target) = target else { continue };
        if target == module_idx {
            continue;
        }
        push(
            &mut items,
            &mut seen,
            mk(last.as_str(), K::MODULE, None, "1"),
        );
        for it in module_items(state, ctx, target, module_idx) {
            push(&mut items, &mut seen, it);
        }
    }

    if let Some(typed) = auto_import {
        items.extend(auto_import_items(state, checker, module_idx, &seen, typed));
    }

    for p in PRIMITIVES {
        push(&mut items, &mut seen, mk(p, K::KEYWORD, None, "3"));
    }
    for k in KEYWORDS {
        push(&mut items, &mut seen, mk(k, K::KEYWORD, None, "4"));
    }
    items
}
