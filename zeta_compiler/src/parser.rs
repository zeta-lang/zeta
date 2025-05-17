use pest_derive::Parser;

use pest::Parser;
use pest::iterators::{Pair, Pairs};
use pest::pratt_parser::PrattParser;

use crate::ast::*;
use crate::ast::ElseBranch::If;
use crate::ast::Stmt::ExprStmt;
use crate::tokens::TokenType::Else;

#[derive(Parser)]
#[grammar = "grammar.pest"]
pub struct ZetaParser;

pub fn parse_program(input: &str) -> Result<Vec<Stmt>, pest::error::Error<Rule>> {
    let pairs = ZetaParser::parse(Rule::program, input)?;
    /*for pair in pairs.clone() {
        println!("{:?}\n", pair);
    }*/

    let mut stmts: Vec<Stmt> = Vec::new();
    let program_pair = pairs.into_iter().next().expect("Expected program rule");
    assert_eq!(program_pair.as_rule(), Rule::program);

    let mut pairs = ZetaParser::parse(Rule::program, input)?;
    let program_pair = pairs.next().unwrap(); // Rule::program

    let mut stmts = Vec::new();
    for pair in program_pair.into_inner() {
        if parse_stmt(&mut stmts, pair) { continue; }
    }

    Ok(stmts)
}

fn parse_stmt(stmts: &mut Vec<Stmt>, pair: Pair<Rule>) -> bool {
    match pair.as_rule() {
        Rule::import_stmt => stmts.push(parse_import_stmt(&pair)),
        Rule::let_stmt => stmts.push(parse_let_stmt(&pair)),
        Rule::return_stmt => stmts.push(parse_return_stmt(&pair)),
        Rule::if_stmt => stmts.push(parse_if_stmt(&pair)),
        Rule::while_stmt => stmts.push(parse_while_stmt(&pair)),
        Rule::for_stmt => stmts.push(parse_for_stmt(&pair)),
        Rule::match_stmt => stmts.push(parse_match_stmt(&pair)),
        Rule::unsafe_block => stmts.push(parse_unsafe_block(&pair)),
        Rule::fun_decl => stmts.push(parse_fun_decl(&pair)),
        Rule::class_decl => stmts.push(parse_class_decl(&pair)),
        Rule::expr_stmt => stmts.push(parse_expr_stmt(&pair)),

        Rule::EOI => return true,
        other => panic!("Unexpected rule in program: {:?}", other),
    }
    false
}

fn parse_while_stmt(pair: &Pair<Rule>) -> Stmt {
    let mut inner = pair.clone().into_inner();
    let value: Option<Box<Expr>> = inner.next().map(|p| Box::new(parse_expr(p)));

    match value {
        Some(expr) => Stmt::While(WhileStmt {
            condition: expr,
            block: parse_block(inner.next().unwrap())
        }),
        None => panic!("Expected expression in while loop")
    }
}

fn parse_for_stmt(pair: &Pair<Rule>) -> Stmt {
    let mut inner = pair.clone().into_inner();

    let initializer_pair = inner.next().unwrap();
    let initializer = match initializer_pair.as_rule() {
        Rule::let_stmt => match parse_let_stmt(&initializer_pair) {
            Stmt::Let(let_stmt) => let_stmt,
            _ => panic!("Expected let_stmt in for loop initializer"),
        },
        _ => panic!("Expected let_stmt in for loop initializer"),
    };

    let condition = parse_expr(inner.next().unwrap());
    let increment = parse_expr(inner.next().unwrap());
    let body = parse_block(inner.next().unwrap());

    Stmt::For(ForStmt {
        let_stmt: initializer,
        condition: Box::new(condition),
        increment: Box::new(increment),
        block: body,
    })
}


fn parse_unsafe_block(pair: &Pair<Rule>) -> Stmt {
    let mut inner = pair.clone().into_inner();
    let block = parse_block(inner.next().unwrap());  // Parse the matched expression
    Stmt::UnsafeBlock(UnsafeBlock { block })
}

fn parse_expr_stmt(pair: &Pair<Rule>) -> Stmt {
    let expr = parse_expr(pair.clone().into_inner().next().unwrap());
    ExprStmt(InternalExprStmt { expr: Box::new(expr) })
}

fn parse_class_decl(pair: &Pair<Rule>) -> Stmt {
    let mut inner = pair.clone().into_inner();

    let visibility = get_visibility(&mut inner);

    let name = inner.next().unwrap().as_str().to_string();

    let params = if let Some(p) = inner.peek() {
        if p.as_rule() == Rule::param_list {
            let param_list = inner.next().unwrap();
            let params = param_list
                .into_inner()
                .map(parse_param)
                .collect::<Vec<_>>();
            Some(params)
        } else {
            inner.next(); // consume optional parens if empty
            None
        }
    } else {
        None
    };

    let body = parse_block(inner.next().unwrap());

    Stmt::ClassDecl(ClassDecl {
        visibility,
        name,
        params,
        body,
    })
}

fn get_visibility(inner: &mut Pairs<Rule>) -> Option<String> {
    match inner.peek().map(|p| p.as_rule()) {
        Some(Rule::public_keyword | Rule::private_keyword | Rule::protected_keyword) => {
            Some(inner.next().unwrap().as_str().to_string())
        }
        _ => None,
    }
}

fn parse_fun_decl(pair: &Pair<Rule>) -> Stmt {
    let mut inner = pair.clone().into_inner();

    let mut visibility = None;
    let mut is_async = false;
    let mut is_unsafe = false;

    while let Some(peek) = inner.peek() {
        match peek.as_rule() {
            Rule::async_modifier => {
                is_async = true;
                println!("async true");
                inner.next();
            },
            Rule::unsafe_modifier => {
                is_unsafe = true;
                println!("unsafe true");
                inner.next();
            },
            Rule::visibility_modifier => {
                if visibility.is_none() {
                    visibility = Some(inner.next().unwrap().as_str().to_string());
                    println!("{:?}", visibility);
                } else {
                    inner.next(); // skip extra
                }
            }
            _ => break,
        }
    }

    let name = inner.next().unwrap().as_str().to_string();

    let mut params = Vec::new();
    if let Some(param_list) = inner.peek().filter(|p| p.as_rule() == Rule::param_list) {
        for param in inner.next().unwrap().into_inner() {
            params.push(parse_param(param));
        }
    }

    let return_type = match inner.peek().filter(|p| p.as_rule() == Rule::type_annotation) {
        Some(_) => Some(inner.next().unwrap().as_str().to_string()),
        None => None,
    };

    let body = parse_block(inner.next().unwrap());

    Stmt::FuncDecl(FuncDecl {
        visibility,
        is_async,
        is_unsafe,
        name,
        params,
        return_type,
        body,
    })
}

fn parse_param(pair: Pair<Rule>) -> Param {
    let mut inner = pair.into_inner();
    let name = inner.next().unwrap().as_str().to_string();
    let type_annotation = inner.next().map(|pair| pair.as_str().to_string());
    Param { name, type_annotation }
}

fn parse_if_stmt(pair: &Pair<Rule>) -> Stmt {
    let mut inner = pair.clone().into_inner();

    let condition = parse_expr(inner.next().unwrap());
    let then_branch = parse_block(inner.next().unwrap());

    let mut else_branch = None;

    if let Some(next) = inner.next() {
        let branch = match next.as_rule() {
            Rule::if_stmt => ElseBranch::If(Box::new(match parse_if_stmt(&next) {
                Stmt::If(if_stmt) => if_stmt,
                _ => panic!("Expected if_stmt"),
            })),
            Rule::block => {
                let block = parse_block(next);
                ElseBranch::Else(block)
            },
            _ => panic!("Unexpected rule in else branch"),
        };
        else_branch = Some(Box::new(branch));
    }

    Stmt::If(IfStmt {
        condition,
        then_branch,
        else_branch,
    })
}


fn parse_match_stmt(pair: &Pair<Rule>) -> Stmt {
    let mut inner = pair.clone().into_inner();
    let expr = parse_expr(inner.next().unwrap());  // Parse the matched expression
    let mut arms = Vec::new();

    // Parse match arms
    for arm in inner {
        arms.push(parse_match_arm(arm)); // Parse individual match arms
    }

    Stmt::Match(MatchStmt { expr, arms }) // Construct the match statement
}

fn parse_return_stmt(pair: &Pair<Rule>) -> Stmt {
    let mut inner = pair.clone().into_inner();
    let value = inner.next().map(|p| Box::new(parse_expr(p)));
    Stmt::Return(ReturnStmt { value })
}

fn parse_let_stmt(pair: &Pair<Rule>) -> Stmt {
    let mut inner = pair.clone().into_inner();

    let mutability = match inner.peek().map(|p| p.as_rule()) {
        Some(Rule::mut_keyword) => {
            inner.next();
            Some(MutKeyword::Mut)
        },
        _ => None,
    };

    let ident = inner.next().unwrap().as_str().to_string();

    let type_annotation = match inner.peek().map(|p| p.as_rule()) {
        Some(Rule::type_annotation) => Some(inner.next().unwrap().as_str().to_string()),
        _ => None,
    };

    // Skip '=' token (if your grammar includes it as its own pair, which is not typical, else just continue)
    // inner.next(); // Only if needed

    let value = Box::new(parse_expr(inner.next().expect("Expected expression after '='")));

    Stmt::Let(LetStmt {
        mutability,
        ident,
        type_annotation,
        value,
    })
}

fn parse_import_stmt(pair: &Pair<Rule>) -> Stmt {
    let mut inner = pair.clone().into_inner();
    let path = inner.next().unwrap().as_str().to_string();
    Stmt::Import(ImportStmt { path })
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
        },
        Rule::wildcard_pattern => Pattern::Wildcard,
        _ => panic!("Unexpected pattern: {:?}", pair.as_rule()),
    }
}


pub fn parse_block(pair: Pair<Rule>) -> Block {
    let mut stmts = Vec::new();
    let stmts_ref: &mut Vec<Stmt> = &mut stmts;
    for stmt in pair.into_inner() {
        parse_stmt(stmts_ref, stmt);
    }
    Block { block: stmts }
}

fn parse_expr(pair: Pair<Rule>) -> Expr {
    match pair.as_rule() {
        Rule::number => Expr::Number(pair.as_str().parse().unwrap()),
        Rule::string => Expr::String(pair.as_str().to_string()),
        Rule::ident => Expr::Ident(pair.as_str().to_string()),

        Rule::assignment => {
            let mut inner = pair.clone().into_inner();
            let first = inner.next().unwrap();

            if inner.clone().count() == 2 {
                // This is an assignment: lhs, op, rhs
                let op = inner.next().unwrap().as_str().to_string();
                let rhs = parse_expr(inner.next().unwrap());

                Expr::Assignment {
                    0: Box::new(parse_expr(first)),
                    1: op,
                    2: Box::new(rhs),
                }
            } else {
                // It's just a logic_or fallback
                parse_expr(first)
            }
        }


        Rule::term => {
            let mut inner = pair.into_inner();
            let mut expr = parse_expr(inner.next().unwrap());

            while let Some(op_pair) = inner.next() {
                let op = op_pair.as_str().to_string();
                let rhs = parse_expr(inner.next().unwrap());
                expr = Expr::Binary {
                    left: Box::new(expr),
                    op,
                    right: Box::new(rhs),
                };
            }

            expr
        }

        Rule::lhs => parse_expr(pair.into_inner().next().unwrap()),

        Rule::expr
        | Rule::logic_or
        | Rule::logic_and
        | Rule::bit_or
        | Rule::bit_xor
        | Rule::bit_and
        | Rule::equality
        | Rule::comparison
        | Rule::shift
        | Rule::factor
        | Rule::unary
        | Rule::primary => {
            // descend to the actual operand
            parse_expr(pair.into_inner().next().unwrap())
        }

        _ => panic!("Unexpected expression rule: {:?}", pair.as_rule()),
    }
}