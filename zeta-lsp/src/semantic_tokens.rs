use ir::{ast::Stmt, ir_hasher::HashMap};
use lsp_types::{SemanticToken, SemanticTokenModifier, SemanticTokenType, SemanticTokensLegend};
use sentinel_typechecker::type_checker::{Occurrence, SymbolId, TypeChecker};

use crate::state::ServerState;

pub const TOKEN_TYPES: &[SemanticTokenType] = &[
    SemanticTokenType::VARIABLE,    // 0
    SemanticTokenType::PROPERTY,    // 1
    SemanticTokenType::FUNCTION,    // 2
    SemanticTokenType::METHOD,      // 3
    SemanticTokenType::STRUCT,      // 4
    SemanticTokenType::ENUM,        // 5
    SemanticTokenType::INTERFACE,   // 6
    SemanticTokenType::ENUM_MEMBER, // 7
    SemanticTokenType::TYPE,        // 8 (type aliases)
];

pub const TOKEN_MODIFIERS: &[SemanticTokenModifier] = &[
    SemanticTokenModifier::STATIC,   // bit 0
    SemanticTokenModifier::READONLY, // bit 1
];
const MOD_STATIC: u32 = 1 << 0;
const MOD_READONLY: u32 = 1 << 1;

pub fn legend() -> SemanticTokensLegend {
    SemanticTokensLegend {
        token_types: TOKEN_TYPES.to_vec(),
        token_modifiers: TOKEN_MODIFIERS.to_vec(),
    }
}

pub fn build_semantic_tokens(
    checker: &TypeChecker,
    state: &ServerState,
    module_idx: usize,
    source: &str,
) -> Vec<SemanticToken> {
    struct Raw {
        line: u32,
        start_char: u32,
        length: u32,
        token_type: u32,
        modifiers: u32,
    }
    let mut raw = Vec::new();

    let mut kinds: HashMap<(usize, usize, &'static str), Option<(u32, u32)>> = HashMap::default();

    for occ in checker.occurrences() {
        let Occurrence {
            span,
            name,
            module_idx: m,
            symbol,
            ..
        } = occ;
        if *m != module_idx {
            continue;
        }
        let (token_type, modifiers) = match symbol {
            SymbolId::Local(_) => (0, 0),
            SymbolId::Field { .. } => (1, 0),
            SymbolId::Method { .. } => (3, 0),
            SymbolId::Variant { .. } => (7, 0),
            SymbolId::Item {
                module_idx: decl_m,
                item_idx,
                tag,
            } => {
                let kind = *kinds.entry((*decl_m, *item_idx, *tag)).or_insert_with(|| {
                    let m = state.compiler.module_with_arena(*decl_m)?;
                    let (_, stmt) = m.parse_result.statements.get(*item_idx)?.peel();
                    match (stmt, *tag) {
                        (Stmt::FuncDecl(_), _) => Some((2, 0)),
                        (Stmt::StructDecl(_), _) => Some((4, 0)),
                        (Stmt::EnumDecl(_), _) => Some((5, 0)),
                        (Stmt::InterfaceDecl(_), _) => Some((6, 0)),
                        (Stmt::TypeAliasDecl(_), _) => Some((8, 0)),
                        (Stmt::Const(_), _) => Some((0, MOD_READONLY)),
                        (Stmt::Let(_), "static") => Some((0, MOD_STATIC)),
                        _ => None,
                    }
                });
                let Some(k) = kind else { continue };

                k
            }
        };

        // `name` is the source spelling, so this compares like with like.
        let expected = name.to_string();
        if crate::text_utils::span_text(source, span).as_deref() != Some(expected.as_str()) {
            continue;
        }

        let start = crate::text_utils::span_to_position(source, span.line, span.column);
        let line_text = crate::text_utils::line_at(source, start.line).unwrap_or("");
        let end_char =
            crate::text_utils::utf16_offset_from_byte(line_text, span.end_column.saturating_sub(1));
        raw.push(Raw {
            line: start.line,
            start_char: start.character,
            length: end_char.saturating_sub(start.character),
            token_type,
            modifiers,
        });
    }

    raw.sort_by_key(|t| (t.line, t.start_char));
    raw.dedup_by_key(|t| (t.line, t.start_char));

    let mut tokens = Vec::with_capacity(raw.len());
    let (mut prev_line, mut prev_char) = (0u32, 0u32);
    for t in raw {
        let delta_line = t.line - prev_line;
        let delta_start = if delta_line == 0 {
            t.start_char - prev_char
        } else {
            t.start_char
        };
        tokens.push(SemanticToken {
            delta_line,
            delta_start,
            length: t.length,
            token_type: t.token_type,
            token_modifiers_bitset: t.modifiers,
        });
        prev_line = t.line;
        prev_char = t.start_char;
    }
    tokens
}
