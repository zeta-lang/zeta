#![allow(unused)]
use anyhow::{Context, Result};
use ir::ast::{ExternModifier, FuncSafety, InlineModifier, Stmt, Visibility};
use ir::hir::{HirEnum, HirFunc, HirGeneric, HirInterface, HirParam, HirStruct, HirType, StrId};
use ir::span::SourceSpan;
use lsp_server::{Connection, Message, Notification, Request, RequestId, Response};
use lsp_types::notification::{
    DidChangeTextDocument, DidCloseTextDocument, DidOpenTextDocument, DidSaveTextDocument, Exit,
    Notification as _, PublishDiagnostics,
};
use lsp_types::request::{
    Completion, DocumentSymbolRequest, GotoDefinition, GotoTypeDefinitionParams, HoverRequest,
    References, Rename, Request as _, SemanticTokensFullRequest,
};
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionParams, CompletionResponse,
    DidChangeTextDocumentParams, DidCloseTextDocumentParams, DidOpenTextDocumentParams,
    DidSaveTextDocumentParams, DocumentSymbol, DocumentSymbolParams, DocumentSymbolResponse,
    GotoDefinitionParams, GotoDefinitionResponse, Hover, HoverContents, HoverParams, Location,
    MarkupContent, MarkupKind, Position, PublishDiagnosticsParams, ReferenceParams, RenameParams,
    SemanticTokens, SemanticTokensParams, SemanticTokensResult, SymbolKind,
    TextDocumentContentChangeEvent, TextEdit, Uri, WorkspaceEdit,
};
use sentinel_typechecker::type_checker::SymbolId;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::str::FromStr;
use zeta_compiler_api::file_handling::compiler_lib_path;
use zeta_compiler_api::file_loader::choose_file_loader;

use crate::lsp_diagnostics::group_by_file;
use crate::state::{Document, ServerState};
use crate::text_utils::{
    byte_offset_in_source, declaration_span, ident_at_column, ident_at_end, line_at,
    nearest_prior_occurrence, prefix_before, span_to_range, tightest_occurrence, to_span_coords,
};
use crate::urls::{path_to_uri, uri_to_path};

pub fn initialize(state: &mut ServerState, params: serde_json::Value) -> Result<()> {
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

        state
            .compiler
            .load_directory(
                &loader,
                &state.workspace_root.as_ref().unwrap().join("src"),
                false,
            )
            .map_err(|e| anyhow::anyhow!("{}", e))?;
    }
    Ok(())
}

pub fn request(connection: &Connection, state: &mut ServerState, request: &Request) -> Result<()> {
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

pub fn notification(
    connection: &Connection,
    state: &mut ServerState,
    notification: &Notification,
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

            let reporter = state.compiler.open_module(&path, params.text_document.text);
            publish_all(connection, &reporter)?;
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

            let reporter = state.compiler.update_module(&path, doc.text.clone());
            publish_all(connection, &reporter)?;
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
            let source = std::fs::read_to_string(&path).unwrap_or_else(|err| {
                panic!(
                    "failed to read file '{}' during didSave: {}",
                    path.display(),
                    err
                )
            });

            let reporter = state.compiler.update_module(&path, source);
            publish_all(connection, &reporter)?;
        }

        DidCloseTextDocument::METHOD => {
            let params: DidCloseTextDocumentParams =
                serde_json::from_value(notification.params.clone())?;
            let uri = params.text_document.uri;
            if let Some(doc) = state.documents.remove(&uri) {
                if let Some(stem) = doc.path.file_stem().and_then(|s| s.to_str()) {
                    state.compiler.close_module(stem);
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

pub fn response(state: &mut ServerState, response: &Response) -> Result<()> {
    Ok(())
}

fn semantic_tokens_at(
    state: &ServerState,
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

/// Publishes one publishDiagnostics notification per file touched by this
/// reporter, converting each file_name string back to a file:// URI.
fn publish_all(
    connection: &Connection,
    reporter: &ir::errors::reporter::ErrorReporter,
) -> Result<()> {
    for (file_name, diagnostics) in group_by_file(reporter) {
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

fn rename_at(state: &ServerState, params: &RenameParams) -> Result<WorkspaceEdit, String> {
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

    let (_, _, symbol_id) = tightest_occurrence(checker.occurrences(), module_idx, line, column)
        .ok_or("no renameable symbol at this position")?;

    if matches!(symbol_id, SymbolId::Item { .. }) {
        return Err(
            "renaming functions, structs, enums, and interfaces isn't supported yet, \
                    usage sites for those aren't tracked, only declarations, so renaming \
                     would break every call site silently"
                .to_string(),
        );
    }

    let mut by_module: HashMap<usize, Vec<(SourceSpan, StrId)>> = HashMap::default();
    for (span, name, _, m, sid, _) in checker.occurrences() {
        if *sid == symbol_id {
            by_module.entry(*m).or_default().push((*span, *name));
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

fn completions_at(state: &ServerState, params: &CompletionParams) -> Vec<CompletionItem> {
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

    let prefix = prefix_before(&source, position);

    let checker_rc = state.compiler.type_checker();
    let checker = checker_rc.borrow();
    let ctx = checker.context();

    if let Some(dot_idx) = prefix.rfind('.') {
        let base = ident_at_end(&prefix[..dot_idx]);
        if !base.is_empty() {
            // `base` is either a type name (static access) or a local
            // variable (instance access via last-known type before cursor).
            let struct_name: Option<String> = if ctx.structs.contains_key(base) {
                Some(base.to_string())
            } else {
                let (line, column) = to_span_coords(&source, position);
                nearest_prior_occurrence(checker.occurrences(), module_idx, base, line, column)
                    .and_then(|ty| match strip_container(&ty) {
                        HirType::Struct { name, .. } => Some(name.to_string()),
                        _ => None,
                    })
            };

            let mut items = Vec::new();
            if let Some(struct_name) = struct_name {
                if let Some(def) = ctx.structs.get(&struct_name) {
                    for field in def.fields {
                        items.push(CompletionItem {
                            label: field.name.to_string(),
                            kind: Some(CompletionItemKind::FIELD),
                            detail: Some(field.field_type.to_string()),
                            ..Default::default()
                        });
                    }
                }
                if let Some(table) = ctx.type_methods.get(&struct_name) {
                    for (name, func) in &table.methods {
                        let ret = func
                            .return_type
                            .map(|t| t.to_string())
                            .unwrap_or_else(|| "void".to_string());
                        items.push(CompletionItem {
                            label: name.clone(),
                            kind: Some(CompletionItemKind::METHOD),
                            detail: Some(format!("() -> {}", ret)),
                            ..Default::default()
                        });
                    }
                }
            }
            return items;
        }
    }

    // Otherwise: global symbol completion (this module + its direct imports).
    let mut items = Vec::new();

    for name in ctx.structs.keys() {
        items.push(CompletionItem {
            label: name.clone(),
            kind: Some(CompletionItemKind::STRUCT),
            ..Default::default()
        });
    }
    for name in ctx.enums.keys() {
        items.push(CompletionItem {
            label: name.clone(),
            kind: Some(CompletionItemKind::ENUM),
            ..Default::default()
        });
    }
    for name in ctx.interfaces.keys() {
        items.push(CompletionItem {
            label: name.clone(),
            kind: Some(CompletionItemKind::INTERFACE),
            ..Default::default()
        });
    }
    if let Some(funcs) = ctx.module_functions.get(&module_idx) {
        for (name, func) in funcs {
            let ret = func
                .return_type
                .map(|t| t.to_string())
                .unwrap_or_else(|| "void".to_string());
            items.push(CompletionItem {
                label: name.clone(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some(ret),
                ..Default::default()
            });
        }
    }
    for imp_idx in state
        .compiler
        .dep_graph_ref()
        .borrow()
        .get_module_imports(module_idx)
    {
        if let Some(funcs) = ctx.module_functions.get(&imp_idx) {
            for name in funcs.keys() {
                items.push(CompletionItem {
                    label: name.clone(),
                    kind: Some(CompletionItemKind::FUNCTION),
                    ..Default::default()
                });
            }
        }
    }

    items
}

pub fn hover_at(state: &ServerState, params: &HoverParams) -> Option<Hover> {
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

    // 1. Try tightest_occurrence first
    if let Some((occ_name, ty, symbol_id)) =
        tightest_occurrence(checker.occurrences(), module_idx, line, column)
    {
        if occ_name.as_str() == word {
            match symbol_id {
                SymbolId::Item {
                    module_idx: m,
                    item_idx,
                    tag,
                } => {
                    let doc = doc_for_symbol(state, &checker, symbol_id);
                    let sig = match tag {
                        "func_sig" | "func_body" => func_for_item(state, &ctx, m, item_idx, tag)
                            .map(|f| format_func_signature(&f)),
                        "type" => struct_in_module(&ctx, m, word)
                            .map(|d| format_struct_signature(&d))
                            .or_else(|| {
                                enum_in_module(&ctx, m, word).map(|e| format_enum_signature(&e))
                            }),
                        "trait" => interface_in_module(&ctx, m, word)
                            .map(|i| format_interface_signature(&i)),
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
                    let sig = format!("{}.{}: {}", struct_name, field_name, ty);
                    return Some(Hover {
                        contents: HoverContents::Markup(format_hover_markup(&sig, doc.as_deref())),
                        range: None,
                    });
                }
                SymbolId::Local(..) => {
                    let doc = doc_for_symbol(state, &checker, symbol_id);
                    let sig = format!("{}: {}", word, ty);
                    return Some(Hover {
                        contents: HoverContents::Markup(format_hover_markup(&sig, doc.as_deref())),
                        range: None,
                    });
                }
            }
        }
    }

    // 2. Fallback resolution for `word` (declarations, type annotations, imports, etc.)
    if let Some((func, decl_m)) = find_func(state, &ctx, module_idx, word) {
        let doc = state
            .compiler
            .source_text(decl_m)
            .and_then(|src| crate::doc_comments::doc_comment_above(&src, func.span.line));
        let sig = format_func_signature(&func);
        return Some(Hover {
            contents: HoverContents::Markup(format_hover_markup(&sig, doc.as_deref())),
            range: None,
        });
    }

    if let Some(def) = ctx.structs.get(word) {
        let doc = ctx
            .struct_owner_module
            .get(word)
            .and_then(|&m| state.compiler.source_text(m))
            .and_then(|src| {
                let s_id = StrId(ctx.string_pool.intern(word));
                let &m = ctx.struct_owner_module.get(word)?;
                let mod_arena = state.compiler.module_with_arena(m)?;
                let span = mod_arena.parse_result.statements.iter().find_map(|s| {
                    let (_, s) = s.peel();
                    if let Stmt::StructDecl(sd) = s {
                        if sd.name == s_id {
                            return Some(sd.span);
                        }
                    }
                    None
                })?;
                crate::doc_comments::doc_comment_above(&src, span.line)
            });
        let sig = format_struct_signature(&def);
        return Some(Hover {
            contents: HoverContents::Markup(format_hover_markup(&sig, doc.as_deref())),
            range: None,
        });
    }

    if let Some(def) = ctx.enums.get(word) {
        let doc = ctx
            .enum_owner_module
            .get(word)
            .and_then(|&m| state.compiler.source_text(m))
            .and_then(|src| {
                let e_id = StrId(ctx.string_pool.intern(word));
                let &m = ctx.enum_owner_module.get(word)?;
                let mod_arena = state.compiler.module_with_arena(m)?;
                let span = mod_arena.parse_result.statements.iter().find_map(|s| {
                    let (_, s) = s.peel();
                    if let Stmt::EnumDecl(ed) = s {
                        if ed.name == e_id {
                            return Some(ed.span);
                        }
                    }
                    None
                })?;
                crate::doc_comments::doc_comment_above(&src, span.line)
            });
        let sig = format_enum_signature(&def);
        return Some(Hover {
            contents: HoverContents::Markup(format_hover_markup(&sig, doc.as_deref())),
            range: None,
        });
    }

    if let Some(def) = ctx.interfaces.get(word) {
        let doc = ctx
            .interface_owner_module
            .get(word)
            .and_then(|&m| state.compiler.source_text(m))
            .and_then(|src| {
                let i_id = StrId(ctx.string_pool.intern(word));
                let &m = ctx.interface_owner_module.get(word)?;
                let mod_arena = state.compiler.module_with_arena(m)?;
                let span = mod_arena.parse_result.statements.iter().find_map(|s| {
                    let (_, s) = s.peel();
                    if let Stmt::InterfaceDecl(id) = s {
                        if id.name == i_id {
                            return Some(id.span);
                        }
                    }
                    None
                })?;
                crate::doc_comments::doc_comment_above(&src, span.line)
            });
        let sig = format_interface_signature(&def);
        return Some(Hover {
            contents: HoverContents::Markup(format_hover_markup(&sig, doc.as_deref())),
            range: None,
        });
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

fn strip_container<'a, 'b, 'bump>(ty: &'b HirType<'a, 'bump>) -> &'b HirType<'a, 'bump> {
    match ty {
        HirType::Ref { inner, .. } => strip_container(inner),
        HirType::SafePointer { inner, .. } => strip_container(inner),
        HirType::UnsafePointer { inner, .. } => strip_container(inner),
        HirType::OwnedPointer { inner, .. } => strip_container(inner),
        _ => ty,
    }
}

fn document_symbols_at(
    state: &ServerState,
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

fn item_decl_span<'a, 'bump>(
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
        (Stmt::ImplDecl(impl_decl), _) => Some(impl_decl.span),
        _ => None,
    }
}

fn method_decl_span<'a, 'bump>(
    stmts: &'bump [Stmt<'a, 'bump>],
    item_idx: usize,
    method_idx: usize,
) -> Option<SourceSpan<'a>> {
    let stmt = stmts.get(item_idx)?;
    let (_, stmt) = stmt.peel();
    if let Stmt::ImplDecl(impl_decl) = stmt {
        let methods = impl_decl.methods.as_ref()?;
        let method = methods.get(method_idx)?;
        Some(method.span)
    } else {
        None
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

pub fn format_func_signature(func: &HirFunc) -> String {
    let mut sig = String::new();

    if func.function_metadata.visibility == Visibility::Private {
        sig.push_str("private ");
    }

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

pub fn format_struct_signature(s: &HirStruct) -> String {
    let mut sig = String::new();
    if s.visibility == Visibility::Private {
        sig.push_str("private ");
    }
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
            if f.visibility == Visibility::Private {
                sig.push_str("    private ");
            } else {
                sig.push_str("    ");
            }
            sig.push_str(&format!("{}: {},\n", f.name, f.field_type));
        }
        sig.push('}');
    }
    sig
}

pub fn format_enum_signature(e: &HirEnum) -> String {
    let mut sig = String::new();
    if e.visibility == Visibility::Private {
        sig.push_str("private ");
    }
    sig.push_str("enum ");
    sig.push_str(e.name.as_str());
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
                sig.push_str(&format!("({})", f_strs.join(", ")));
            }
            sig.push_str(",\n");
        }
        sig.push('}');
    }
    sig
}

pub fn format_interface_signature(i: &HirInterface) -> String {
    let mut sig = String::new();
    if i.visibility == Visibility::Private {
        sig.push_str("private ");
    }
    sig.push_str("interface ");
    sig.push_str(i.name.as_str());
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
    // 1. Current module
    if let Some(func) = ctx
        .module_functions
        .get(&module_idx)
        .and_then(|f| f.get(name))
    {
        return Some((*func, func.declaring_module_idx));
    }

    // 2. Direct imports
    let dep_graph = state.compiler.dep_graph_ref().borrow();
    for imp_idx in dep_graph.get_module_imports(module_idx) {
        if let Some(func) = ctx.module_functions.get(&imp_idx).and_then(|f| f.get(name)) {
            return Some((*func, func.declaring_module_idx));
        }
    }

    // 3. Anywhere via dep_graph symbol lookup or find_function_by_name_anywhere
    let name_id = StrId(ctx.string_pool.intern(name));
    if let Some((decl_m, _, _)) = dep_graph.resolve_item_in_module(module_idx, name_id) {
        if let Some(func) = ctx.module_functions.get(&decl_m).and_then(|f| f.get(name)) {
            return Some((*func, func.declaring_module_idx));
        }
    }

    // 3b. Anywhere: only if exactly one candidate
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

    // 4. Any method: only if exactly one candidate
    let mut methods: Vec<HirFunc<'a, 'bump>> = ctx
        .type_methods
        .values()
        .filter_map(|t| t.methods.get(name).copied())
        .collect();
    if methods.len() == 1 {
        let f = methods.remove(0);
        return Some((f, f.declaring_module_idx));
    }
    None
}

pub fn go_to_definition(
    state: &mut ServerState,
    params: &GotoDefinitionParams,
) -> Option<GotoDefinitionResponse> {
    let uri: &Uri = &params.text_document_position_params.text_document.uri;
    let position: Position = params.text_document_position_params.position;

    let doc = state.documents.get(uri)?;
    let module_idx = state.compiler.module_idx_for_path(&doc.path)?;
    let source = state.compiler.source_text(module_idx)?;
    let (line, column) = to_span_coords(&source, position);
    let word = ident_at_column(line_at(&source, position.line)?, position.character)?;

    let checker_rc = state.compiler.type_checker();
    let checker = checker_rc.borrow();
    let ctx = checker.context();

    // 1. Check tightest_occurrence
    if let Some((occ_name, _ty, symbol_id)) =
        tightest_occurrence(checker.occurrences(), module_idx, line, column)
    {
        if occ_name.as_str() == word {
            match symbol_id {
                SymbolId::Item {
                    module_idx: m,
                    item_idx,
                    tag,
                } => {
                    let module_with_arena = state.compiler.module_with_arena(m)?;
                    let span =
                        item_decl_span(&module_with_arena.parse_result.statements, item_idx, tag)?;
                    let target_path = state.compiler.path_for_module(m)?;
                    let target_source = state.compiler.source_text(m)?;
                    let range = find_ident_range_on_line(&target_source, span.line, word, span);
                    let target_uri = path_to_uri(target_path).ok()?;
                    return Some(GotoDefinitionResponse::Scalar(Location {
                        uri: target_uri,
                        range,
                    }));
                }
                SymbolId::Method {
                    module_idx: m,
                    item_idx,
                    method_idx,
                } => {
                    let module_with_arena = state.compiler.module_with_arena(m)?;
                    let span = method_decl_span(
                        &module_with_arena.parse_result.statements,
                        item_idx,
                        method_idx,
                    )?;
                    let target_path = state.compiler.path_for_module(m)?;
                    let target_source = state.compiler.source_text(m)?;
                    let range = find_ident_range_on_line(&target_source, span.line, word, span);
                    let target_uri = path_to_uri(target_path).ok()?;
                    return Some(GotoDefinitionResponse::Scalar(Location {
                        uri: target_uri,
                        range,
                    }));
                }
                SymbolId::Field {
                    struct_name,
                    field_name,
                } => {
                    let struct_str = struct_name.to_string();
                    let &m = ctx.struct_owner_module.get(&struct_str)?;
                    let module_with_arena = state.compiler.module_with_arena(m)?;
                    let span = module_with_arena
                        .parse_result
                        .statements
                        .iter()
                        .find_map(|s| {
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
                    let target_path = state.compiler.path_for_module(m)?;
                    let target_source = state.compiler.source_text(m)?;
                    let range = find_ident_range_on_line(&target_source, span.line, word, span);
                    let target_uri = path_to_uri(target_path).ok()?;
                    return Some(GotoDefinitionResponse::Scalar(Location {
                        uri: target_uri,
                        range,
                    }));
                }
                SymbolId::Local(..) => {
                    if let Some((span, _, _, occ_m, _, _)) = checker
                        .occurrences()
                        .iter()
                        .find(|(_, _, _, _, sid, is_decl)| *sid == symbol_id && *is_decl)
                    {
                        let target_path = state.compiler.path_for_module(*occ_m)?;
                        let target_source = state.compiler.source_text(*occ_m)?;
                        let range =
                            find_ident_range_on_line(&target_source, span.line, word, *span);
                        let target_uri = path_to_uri(target_path).ok()?;
                        return Some(GotoDefinitionResponse::Scalar(Location {
                            uri: target_uri,
                            range,
                        }));
                    }
                }
            }
        }
    }

    // 2. Fallback resolution by `word`
    // Function fallback
    if let Some((func, decl_m)) = find_func(state, &ctx, module_idx, word) {
        let span = func.span;
        let target_path = state.compiler.path_for_module(decl_m)?;
        let target_source = state.compiler.source_text(decl_m)?;
        let range = find_ident_range_on_line(&target_source, span.line, word, span);
        let target_uri = path_to_uri(target_path).ok()?;
        return Some(GotoDefinitionResponse::Scalar(Location {
            uri: target_uri,
            range,
        }));
    }

    // Struct fallback
    if let Some(&m) = ctx.struct_owner_module.get(word) {
        let s_id = StrId(ctx.string_pool.intern(word));
        let module_with_arena = state.compiler.module_with_arena(m)?;
        let span = module_with_arena
            .parse_result
            .statements
            .iter()
            .find_map(|s| {
                let (_, s) = s.peel();
                if let Stmt::StructDecl(sd) = s {
                    if sd.name == s_id {
                        return Some(sd.span);
                    }
                }
                None
            })?;
        let target_path = state.compiler.path_for_module(m)?;
        let target_source = state.compiler.source_text(m)?;
        let range = find_ident_range_on_line(&target_source, span.line, word, span);
        let target_uri = path_to_uri(target_path).ok()?;
        return Some(GotoDefinitionResponse::Scalar(Location {
            uri: target_uri,
            range,
        }));
    }

    // Enum fallback
    if let Some(&m) = ctx.enum_owner_module.get(word) {
        let e_id = StrId(ctx.string_pool.intern(word));
        let module_with_arena = state.compiler.module_with_arena(m)?;
        let span = module_with_arena
            .parse_result
            .statements
            .iter()
            .find_map(|s| {
                let (_, s) = s.peel();
                if let Stmt::EnumDecl(ed) = s {
                    if ed.name == e_id {
                        return Some(ed.span);
                    }
                }
                None
            })?;
        let target_path = state.compiler.path_for_module(m)?;
        let target_source = state.compiler.source_text(m)?;
        let range = find_ident_range_on_line(&target_source, span.line, word, span);
        let target_uri = path_to_uri(target_path).ok()?;
        return Some(GotoDefinitionResponse::Scalar(Location {
            uri: target_uri,
            range,
        }));
    }

    // Interface fallback
    if let Some(&m) = ctx.interface_owner_module.get(word) {
        let i_id = StrId(ctx.string_pool.intern(word));
        let module_with_arena = state.compiler.module_with_arena(m)?;
        let span = module_with_arena
            .parse_result
            .statements
            .iter()
            .find_map(|s| {
                let (_, s) = s.peel();
                if let Stmt::InterfaceDecl(id) = s {
                    if id.name == i_id {
                        return Some(id.span);
                    }
                }
                None
            })?;
        let target_path = state.compiler.path_for_module(m)?;
        let target_source = state.compiler.source_text(m)?;
        let range = find_ident_range_on_line(&target_source, span.line, word, span);
        let target_uri = path_to_uri(target_path).ok()?;
        return Some(GotoDefinitionResponse::Scalar(Location {
            uri: target_uri,
            range,
        }));
    }

    // Item in dep_graph
    let word_id = StrId(ctx.string_pool.intern(word));
    let dep_graph = state.compiler.dep_graph_ref().borrow();
    if let Some((decl_m, item_idx, tag)) = dep_graph.resolve_item_in_module(module_idx, word_id) {
        let module_with_arena = state.compiler.module_with_arena(decl_m)?;
        let span = item_decl_span(&module_with_arena.parse_result.statements, item_idx, tag)?;
        let target_path = state.compiler.path_for_module(decl_m)?;
        let target_source = state.compiler.source_text(decl_m)?;
        let range = find_ident_range_on_line(&target_source, span.line, word, span);
        let target_uri = path_to_uri(target_path).ok()?;
        return Some(GotoDefinitionResponse::Scalar(Location {
            uri: target_uri,
            range,
        }));
    }

    // Module package path fallback
    if let Some(m) = dep_graph.resolve_module_path(&[word_id]) {
        let target_path = state.compiler.path_for_module(m)?;
        let target_uri = path_to_uri(target_path).ok()?;
        return Some(GotoDefinitionResponse::Scalar(Location {
            uri: target_uri,
            range: lsp_types::Range::default(),
        }));
    }

    None
}

fn references_at(state: &ServerState, params: &ReferenceParams) -> Vec<Location> {
    let text_document = &params.text_document_position.text_document;
    let position = params.text_document_position.position;
    let include_declaration = params.context.include_declaration;

    let Some(doc) = state.documents.get(&text_document.uri) else {
        return vec![];
    };
    let Some(module_idx) = state.compiler.module_idx_for_path(&doc.path) else {
        return vec![];
    };
    let Some(source) = state.compiler.source_text(module_idx) else {
        return vec![];
    };
    let (line, column) = to_span_coords(&source, position);

    let checker_rc = state.compiler.type_checker();
    let checker = checker_rc.borrow();

    let Some((_, _, symbol_id)) =
        tightest_occurrence(checker.occurrences(), module_idx, line, column)
    else {
        return vec![];
    };

    let mut locations = Vec::new();

    if let SymbolId::Item {
        module_idx: m,
        item_idx,
        tag,
    } = symbol_id
    {
        if include_declaration {
            if let Some(module_with_arena) = state.compiler.module_with_arena(m) {
                if let Some(span) =
                    item_decl_span(&module_with_arena.parse_result.statements, item_idx, tag)
                {
                    if let Some(target_path) = state.compiler.path_for_module(m) {
                        if let Some(target_source) = state.compiler.source_text(m) {
                            let range = span_to_range(
                                &target_source,
                                span.line,
                                span.column,
                                span.end_line,
                                span.end_column,
                            );
                            if let Ok(uri) = path_to_uri(target_path) {
                                locations.push(Location { uri, range });
                            }
                        }
                    }
                }
            }
        }
    }

    for (span, _, _, m, sid, is_decl) in checker.occurrences() {
        if *sid != symbol_id {
            continue;
        }
        if *is_decl && !include_declaration {
            continue;
        }
        let Some(m_source) = state.compiler.source_text(*m) else {
            continue;
        };
        let Some(m_path) = state.compiler.path_for_module(*m) else {
            continue;
        };
        let range = span_to_range(
            &m_source,
            span.line,
            span.column,
            span.end_line,
            span.end_column,
        );
        let Ok(uri) = path_to_uri(m_path) else {
            continue;
        };
        locations.push(Location { uri, range });
    }

    locations
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

fn doc_for_symbol(
    state: &ServerState,
    checker: &sentinel_typechecker::TypeChecker,
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
        _ => checker
            .occurrences()
            .iter()
            .find(|(_, _, _, _, s, is_decl)| *s == sid && *is_decl)
            .map(|(span, _, _, m, _, _)| (*m, *span))?,
    };
    let src = state.compiler.source_text(m)?;
    crate::doc_comments::doc_comment_above(&src, span.line)
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
