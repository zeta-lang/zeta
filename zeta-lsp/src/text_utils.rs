use ir::{hir::HirType, span::SourceSpan};
use sentinel_typechecker::type_checker::{Occurrence, SymbolId};

/// Byte offset within `line` corresponding to a UTF-16 code-unit offset
/// (LSP's `Position.character`). Clamped to `line`'s length if `utf16_offset`
/// runs past the end.
pub fn byte_offset_from_utf16(line: &str, utf16_offset: u32) -> usize {
    let mut utf16_count = 0u32;
    for (byte_idx, ch) in line.char_indices() {
        if utf16_count >= utf16_offset {
            return byte_idx;
        }
        utf16_count += ch.len_utf16() as u32;
    }
    line.len()
}

/// Inverse: UTF-16 code-unit offset corresponding to a byte offset into
/// `line`. Assumes `byte_offset` lands on a char boundary (true for any
/// offset derived from a SourceSpan column, since those always point at
/// token starts).
pub fn utf16_offset_from_byte(line: &str, byte_offset: usize) -> u32 {
    let clamped = byte_offset.min(line.len());
    line[..clamped].chars().map(|c| c.len_utf16() as u32).sum()
}

pub fn line_at(source: &str, line: u32) -> Option<&str> {
    source.lines().nth(line as usize)
}

/// Text on `position`'s line up to (not including) the cursor column,
/// UTF-16-correct.
pub fn prefix_before(source: &str, position: lsp_types::Position) -> String {
    let Some(line) = line_at(source, position.line) else {
        return String::new();
    };
    let byte_idx = byte_offset_from_utf16(line, position.character);
    line[..byte_idx].to_string()
}

pub fn ident_at_end(prefix: &str) -> &str {
    let start = prefix
        .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
        .map(|i| i + 1)
        .unwrap_or(0);
    &prefix[start..]
}

/// Converts an LSP position into the 1-indexed (line, byte-column) pair
/// matching SourceSpan's convention, resolving UTF-16 -> byte via `source`.
pub fn to_span_coords(source: &str, position: lsp_types::Position) -> (usize, usize) {
    let line_idx = position.line as usize + 1;
    let byte_col = match line_at(source, position.line) {
        Some(line) => byte_offset_from_utf16(line, position.character) + 1,
        None => position.character as usize + 1,
    };
    (line_idx, byte_col)
}

pub fn ident_at_column(line: &str, utf16_char: u32) -> Option<&str> {
    let byte_idx = byte_offset_from_utf16(line, utf16_char);
    let start = line[..byte_idx]
        .rfind(|c: char| !(c.is_alphanumeric() || c == '_'))
        .map(|i| i + 1)
        .unwrap_or(0);
    let end = line[byte_idx..]
        .find(|c: char| !(c.is_alphanumeric() || c == '_'))
        .map(|i| byte_idx + i)
        .unwrap_or(line.len());
    if start >= end {
        None
    } else {
        Some(&line[start..end])
    }
}

/// Span coordinate (1-indexed line/byte-column) -> LSP Position (0-indexed
/// line/UTF-16 character).
pub fn span_to_position(source: &str, line: usize, column: usize) -> lsp_types::Position {
    let byte_col = column.saturating_sub(1);
    let character = match line_at(source, (line.saturating_sub(1)) as u32) {
        Some(text) => utf16_offset_from_byte(text, byte_col),
        None => byte_col as u32,
    };
    lsp_types::Position {
        line: line.saturating_sub(1) as u32,
        character,
    }
}

pub fn span_to_range(
    source: &str,
    line: usize,
    column: usize,
    end_line: usize,
    end_column: usize,
) -> lsp_types::Range {
    lsp_types::Range {
        start: span_to_position(source, line, column),
        end: span_to_position(source, end_line, end_column),
    }
}

pub fn span_contains(span: &SourceSpan, line: usize, column: usize) -> bool {
    if line < span.line || line > span.end_line {
        return false;
    }
    if line == span.line && column < span.column {
        return false;
    }
    if line == span.end_line && column > span.end_column {
        return false;
    }
    true
}

fn span_width(span: &SourceSpan) -> usize {
    if span.end_line == span.line {
        span.end_column.saturating_sub(span.column)
    } else {
        usize::MAX
    }
}

pub fn tightest_occurrence<'o, 'a, 'bump>(
    occurrences: &'o [Occurrence<'a, 'bump>],
    module_idx: usize,
    line: usize,
    column: usize,
) -> Option<&'o Occurrence<'a, 'bump>> {
    occurrences
        .iter()
        .filter(|o| o.module_idx == module_idx && span_contains(&o.span, line, column))
        .min_by_key(|o| span_width(&o.span))
}

pub fn declaration_span<'a, 'bump>(
    occurrences: &[Occurrence<'a, 'bump>],
    symbol_id: SymbolId,
) -> Option<SourceSpan<'a>> {
    occurrences
        .iter()
        .find(|o| o.symbol == symbol_id && o.is_decl)
        .map(|o| o.span)
}

pub fn nearest_prior_occurrence<'a, 'bump>(
    occurrences: &[Occurrence<'a, 'bump>],
    module_idx: usize,
    name: &str,
    line: usize,
    column: usize,
) -> Option<HirType<'a, 'bump>> {
    occurrences
        .iter()
        .filter(|o| {
            o.module_idx == module_idx
                && o.name == name
                && (o.span.line < line || (o.span.line == line && o.span.column <= column))
        })
        .max_by_key(|o| (o.span.line, o.span.column))
        .map(|o| o.ty)
}

pub fn span_text(source: &str, span: &SourceSpan) -> Option<String> {
    if span.line != span.end_line {
        return None; // identifiers shouldn't span multiple lines
    }
    let line = line_at(source, (span.line - 1) as u32)?;
    let start = span.column.saturating_sub(1);
    let end = span.end_column.saturating_sub(1);
    if end > line.len() || start > end {
        return None;
    }
    Some(line[start..end].to_string())
}

pub fn byte_offset_in_source(source: &str, position: lsp_types::Position) -> usize {
    let mut offset = 0;

    for (i, line) in source.split_inclusive('\n').enumerate() {
        if i == position.line as usize {
            return offset + byte_offset_from_utf16(line, position.character);
        }
        offset += line.len();
    }

    if let Some(last) = source.lines().last() {
        let line_count = source.lines().count();
        if position.line as usize == line_count.saturating_sub(1) {
            return source.len() - last.len() + byte_offset_from_utf16(last, position.character);
        }
    }

    source.len()
}

/// Is the 0-indexed LSP `position` inside `range`? End-inclusive, so a cursor sitting right
/// after the last character of an identifier still counts.
pub fn range_contains(range: &lsp_types::Range, position: lsp_types::Position) -> bool {
    let p = (position.line, position.character);
    (range.start.line, range.start.character) <= p && p <= (range.end.line, range.end.character)
}
