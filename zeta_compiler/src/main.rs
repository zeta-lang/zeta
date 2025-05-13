use pest::error::LineColLocation::{Pos, Span};

mod tokens;
mod parser;
mod ast;
mod opcodes;

fn main() {
    match parser::parse_program("\
    import std::io

    let mut x: int = 10
    x += 5
    
    if (x > 10) {
        return x
    } else {
        return x - 1
    }
    
    match x {
        0 => {
            return 0
        },
        1 => {
            return 1
        },
        _ => {
            return x
        }
    }
    
    func add(a: int, b: int): int {
        return a + b
    }
    
    class Person(name: string, age: int) {
        func greet(): void {
            print(\"Hello\")
        }
    }
    
    let ref: &mut int = &mut x

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
