use pest_derive::Parser;

use pest::Parser;
use pest::iterators::Pair;

use crate::ast::*;

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct ZetaParser;

pub fn parse_program(input: &str) -> Result<Vec<Stmt>, pest::error::Error<Rule>> {
    let pairs = ZetaParser::parse(Rule::program, input)?;
    for pair in pairs.clone() {
        println!("{:?}\n", pair);
    }

    let mut stmts = Vec::new();
    for pair in pairs {
        // Each pair here represents a statement in the program
        // You need to recursively process statements within the program
        if let Rule::stmt = pair.as_rule() {
            stmts.push(parse_stmt(pair));
        } else {
            panic!("Unexpected rule: {:?}", pair.as_rule());
        }
    }

    Ok(stmts)
}

fn parse_stmt(pair: Pair<Rule>) -> Stmt {
    match pair.as_rule() {
        Rule::import_stmt => {
            let mut inner = pair.into_inner();
            let path = inner.next().unwrap().as_str().to_string();
            Stmt::Import(ImportStmt { path })
        }
        Rule::let_stmt => {
            let mut inner = pair.into_inner();
            let mutability = inner.next().map(|_| MutKeyword::Mut);
            let ident = inner.next().unwrap().as_str().to_string();
            let type_annotation = inner.next().map(|p| p.as_str().to_string());
            let value = Box::new(parse_expr(inner.next().unwrap()));
            Stmt::Let(LetStmt {
                mutability,
                ident,
                type_annotation,
                value,
            })
        }
        Rule::return_stmt => {
            let mut inner = pair.into_inner();
            let value = inner.next().map(|p| Box::new(parse_expr(p)));
            Stmt::Return(ReturnStmt { value })
        }
        Rule::match_stmt => {
            let mut inner = pair.into_inner();
            let expr = parse_expr(inner.next().unwrap());  // Parse the matched expression
            let mut arms = Vec::new();

            // Parse match arms
            for arm in inner {
                arms.push(parse_match_arm(arm)); // Parse individual match arms
            }

            Stmt::Match(MatchStmt { expr, arms }) // Construct the match statement
        }
        _ => panic!("Unexpected rule: {:?}", pair.as_rule()),
    }
}

fn parse_match_arm(pair: Pair<Rule>) -> MatchArm {
    let mut inner = pair.into_inner();
    let pattern = parse_pattern(inner.next().unwrap()); // Parse the pattern
    let block = parse_block(inner.next().unwrap()); // Parse the block for the match arm
    MatchArm { pattern, block }
}

fn parse_pattern(pair: Pair<Rule>) -> Pattern {
    match pair.as_rule() {
        Rule::ident => Pattern::Ident(pair.as_str().to_string()),
        Rule::number => Pattern::Number(pair.as_str().parse().unwrap()),
        Rule::string => Pattern::String(pair.as_str().to_string()),
        Rule::tuple_pattern => {
            let inner = pair.into_inner();
            let mut patterns = Vec::new();
            for p in inner {
                patterns.push(parse_pattern(p));
            }
            Pattern::Tuple(patterns)
        }
        _ => panic!("Unexpected pattern: {:?}", pair.as_rule()),
    }
}

fn parse_block(pair: Pair<Rule>) -> Block {
    let mut stmts = Vec::new();
    for stmt in pair.into_inner() {
        stmts.push(parse_stmt(stmt));
    }
    Block { stmts }
}

fn parse_expr(pair: Pair<Rule>) -> Expr {
    // Implement logic for parsing expressions
    match pair.as_rule() {
        Rule::number => Expr::Number(pair.as_str().parse().unwrap()),
        Rule::string => Expr::String(pair.as_str().to_string()),
        Rule::ident => Expr::Ident(pair.as_str().to_string()),
        _ => panic!("Unexpected expression rule: {:?}", pair.as_rule()),
    }
}
