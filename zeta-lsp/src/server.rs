use std::collections::VecDeque;

use anyhow::Result;
use lsp_server::{Connection, Message};
use lsp_types::{
    CompletionOptions, HoverProviderCapability, SaveOptions, SemanticTokensFullOptions,
    SemanticTokensOptions, SemanticTokensServerCapabilities, ServerCapabilities,
    TextDocumentSyncCapability, TextDocumentSyncKind, TextDocumentSyncOptions,
    TextDocumentSyncSaveOptions,
};

use crate::{
    dispatch,
    state::{CompletionCache, ServerState},
};

pub struct Server<'a, 'bump> {
    connection: Connection,
    state: ServerState<'a, 'bump>,
}

impl<'a, 'bump> Server<'a, 'bump>
where
    'bump: 'a,
{
    pub fn new(compiler: zeta_compiler_api::Compiler<'a, 'bump>) -> Result<Self> {
        let (connection, _) = Connection::stdio();

        Ok(Self {
            connection,
            state: ServerState {
                compiler,
                workspace_root: None,
                documents: Default::default(),
                should_exit: false,
                completion_cache: CompletionCache::default(),
            },
        })
    }

    pub fn run(mut self) -> Result<()> {
        let capabilities = serde_json::to_value(ServerCapabilities {
            text_document_sync: Some(TextDocumentSyncCapability::Options(
                TextDocumentSyncOptions {
                    open_close: Some(true),
                    change: Some(TextDocumentSyncKind::INCREMENTAL),
                    save: Some(TextDocumentSyncSaveOptions::SaveOptions(SaveOptions {
                        include_text: Some(false),
                    })),
                    ..Default::default()
                },
            )),
            implementation_provider: Some(lsp_types::ImplementationProviderCapability::Simple(
                true,
            )),
            document_symbol_provider: Some(lsp_types::OneOf::Left(true)),
            definition_provider: Some(lsp_types::OneOf::Left(true)),
            hover_provider: Some(HoverProviderCapability::Simple(true)),
            completion_provider: Some(CompletionOptions {
                trigger_characters: Some(vec![".".to_string(), "::".to_string()]),
                resolve_provider: Some(false), // set true later if we add completionItem/resolve
                ..Default::default()
            }),
            semantic_tokens_provider: Some(
                SemanticTokensServerCapabilities::SemanticTokensOptions(SemanticTokensOptions {
                    legend: crate::semantic_tokens::legend(),
                    full: Some(SemanticTokensFullOptions::Bool(true)),
                    range: None,
                    work_done_progress_options: Default::default(),
                }),
            ),
            rename_provider: Some(lsp_types::OneOf::Right(lsp_types::RenameOptions {
                prepare_provider: Some(false),
                work_done_progress_options: Default::default(),
            })),
            ..Default::default()
        })
        .unwrap();

        let initialize_params = self.connection.initialize(capabilities)?;
        dispatch::initialize(&mut self.state, initialize_params)?;

        let mut queue: VecDeque<Message> = VecDeque::new();
        loop {
            if queue.is_empty() {
                match self.connection.receiver.recv() {
                    Ok(m) => queue.push_back(m),
                    Err(_) => break,
                }
            }
            while let Ok(m) = self.connection.receiver.try_recv() {
                queue.push_back(m);
            }
            let Some(message) = queue.pop_front() else {
                continue;
            };
            match message {
                Message::Request(req) => {
                    if self.connection.handle_shutdown(&req)? {
                        break;
                    }
                    if let Err(e) = dispatch::request(&self.connection, &mut self.state, &req) {
                        eprintln!("[zeta-lsp] {} failed: {e}", req.method);
                        let _ = self.connection.sender.send(Message::Response(
                            lsp_server::Response::new_err(
                                req.id.clone(),
                                lsp_server::ErrorCode::InternalError as i32,
                                e.to_string(),
                            ),
                        ));
                    }
                }
                Message::Notification(n) => {
                    let superseded = n.method == "textDocument/didChange"
                        && matches!(queue.front(), Some(Message::Notification(next))
                            if next.method == "textDocument/didChange" && uri_of(next) == uri_of(&n));
                    if let Err(e) =
                        dispatch::notification(&self.connection, &mut self.state, &n, !superseded)
                    {
                        eprintln!("[zeta-lsp] {} failed: {e}", n.method);
                    }
                    if self.state.should_exit {
                        break;
                    }
                }
                Message::Response(_) => {}
            }
        }
        Ok(())
    }
}

fn uri_of(n: &lsp_server::Notification) -> Option<String> {
    n.params
        .pointer("/textDocument/uri")?
        .as_str()
        .map(str::to_owned)
}
