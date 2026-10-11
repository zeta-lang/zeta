pub fn doc_comment_above(source: &str, decl_line: usize) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    let mut i = decl_line.checked_sub(1)?; // 0-indexed decl line
    let mut docs: Vec<&str> = Vec::new();

    while i > 0 {
        i -= 1;
        let t = lines.get(i)?.trim_start();
        if let Some(rest) = t.strip_prefix("///") {
            docs.push(rest.strip_prefix(' ').unwrap_or(rest));
        } else if docs.is_empty()
            && (t.starts_with("#[")
                || t.starts_with("extern")
                || t.starts_with("public")
                || t.starts_with("private")
                || t.starts_with("unsafe")
                || t.starts_with("inline")
                || t.starts_with("noinline")
                || t.is_empty())
        {
            continue; // modifier, attribute, or blank line between the doc and the item
        } else {
            break;
        }
    }

    if docs.is_empty() {
        return None;
    }
    docs.reverse();
    Some(docs.join("\n"))
}

pub fn format_as_zeta_comments(docs: &str) -> String {
    let mut out = String::new();
    for line in docs.lines() {
        if line.is_empty() {
            out.push_str("///\n");
        } else {
            out.push_str("/// ");
            out.push_str(line);
            out.push('\n');
        }
    }
    out
}

