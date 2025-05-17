use pest::error::LineColLocation::{Pos, Span};

mod tokens;
mod parser;
mod ast;
mod opcodes;

fn main() {
    match parser::parse_program("\
    public async unsafe fun add(a: int, b: int): int {
    }

    public fun main() {
        for (let mut hi: int = 0; hi < 10; hi += 1) {
            
        }
    }
    ") {
        Ok(nodes) => println!("Parsed successfully: {:?}", nodes),
        Err(err) => {
            eprintln!("Parsing failed: {:?}", err);
            match err.line_col {
                Pos(pos) => eprintln!("At line: {}, column: {}", pos.0, pos.1),
                Span(span, ..) => eprintln!("At span: {:?}", span),
            }
        }
    }
}
