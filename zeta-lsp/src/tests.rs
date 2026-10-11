#[cfg(test)]
mod tests {
    use crate::dispatch::{
        format_enum_signature, format_func_signature, format_hover_markup, format_struct_signature,
        go_to_definition, hover_at,
    };
    use crate::doc_comments::{doc_comment_above, format_as_zeta_comments};
    use crate::state::{Document, ServerState};
    use crate::urls::path_to_uri;
    use ir::ast::{
        ExternModifier, FuncModifiers, FuncSafety, InlineModifier, MutabilityState, Visibility,
    };
    use ir::hir::{HirEnum, HirField, HirFunc, HirParam, HirStruct, HirType, StrId};
    use ir::span::SourceSpan;
    use lsp_types::{
        GotoDefinitionParams, HoverContents, HoverParams, Position, TextDocumentIdentifier,
        TextDocumentPositionParams,
    };
    use std::path::PathBuf;
    use zeta_compiler_api::file_handling::compiler_lib_path;
    use zeta_compiler_api::file_loader::choose_file_loader;

    static HIR_I32: HirType<'static, 'static> = HirType::I32;
    static HIR_U64: HirType<'static, 'static> = HirType::U64;

    #[test]
    fn test_format_func_signature_extern_c_with_unsafe_ptr_param() {
        // extern "C" func pthread_mutex_unlock(mutex: [*]mut i32): i32;
        let param_type = HirType::UnsafePointer {
            inner: &HIR_I32,
            mutability_state: MutabilityState::Mut,
        };
        let params: &[HirParam<'static, 'static>] = &[HirParam::Normal {
            name: StrId::from_static("mutex"),
            param_type,
            multi_place: None,
            span: SourceSpan::default(),
        }];
        let func = HirFunc {
            name: StrId::from_static("pthread_mutex_unlock"),
            function_metadata: FuncModifiers {
                visibility: Visibility::Public,
                extern_modifier: ExternModifier::Abi(StrId::from_static("C")),
                inline_modifier: InlineModifier::None,
                func_safety: FuncSafety::Safe,
            },
            generics: None,
            params: Some(params),
            return_type: Some(HirType::I32),
            body: None,
            unmangled_name: StrId::from_static("pthread_mutex_unlock"),
            declaring_module_idx: 0,
            impl_target: None,
            span: SourceSpan::default(),
        };

        let sig = format_func_signature(&func);
        assert_eq!(
            sig,
            r#"extern "C" func pthread_mutex_unlock(mutex: [*]mut i32): i32;"#
        );
    }

    #[test]
    fn test_format_func_signature_simple() {
        // func square(x: i32): i32;
        let params: &[HirParam<'static, 'static>] = &[HirParam::Normal {
            name: StrId::from_static("x"),
            param_type: HirType::I32,
            multi_place: None,
            span: SourceSpan::default(),
        }];
        let func = HirFunc {
            name: StrId::from_static("square"),
            function_metadata: FuncModifiers {
                visibility: Visibility::Public,
                extern_modifier: ExternModifier::None,
                inline_modifier: InlineModifier::None,
                func_safety: FuncSafety::Safe,
            },
            generics: None,
            params: Some(params),
            return_type: Some(HirType::I32),
            body: None,
            unmangled_name: StrId::from_static("square"),
            declaring_module_idx: 0,
            impl_target: None,
            span: SourceSpan::default(),
        };

        let sig = format_func_signature(&func);
        assert_eq!(sig, "func square(x: i32): i32;");
    }

    #[test]
    fn test_format_func_signature_void_return_no_params() {
        // func init(): void;
        let func = HirFunc {
            name: StrId::from_static("init"),
            function_metadata: FuncModifiers {
                visibility: Visibility::Public,
                extern_modifier: ExternModifier::None,
                inline_modifier: InlineModifier::None,
                func_safety: FuncSafety::Safe,
            },
            generics: None,
            params: Some(&[]),
            return_type: Some(HirType::Void),
            body: None,
            unmangled_name: StrId::from_static("init"),
            declaring_module_idx: 0,
            impl_target: None,
            span: SourceSpan::default(),
        };

        let sig = format_func_signature(&func);
        assert_eq!(sig, "func init(): void;");
    }

    #[test]
    fn test_format_func_signature_unsafe_and_private() {
        // private unsafe func dangerous(): void;
        let func = HirFunc {
            name: StrId::from_static("dangerous"),
            function_metadata: FuncModifiers {
                visibility: Visibility::Private,
                extern_modifier: ExternModifier::None,
                inline_modifier: InlineModifier::None,
                func_safety: FuncSafety::Unsafe,
            },
            generics: None,
            params: Some(&[]),
            return_type: Some(HirType::Void),
            body: None,
            unmangled_name: StrId::from_static("dangerous"),
            declaring_module_idx: 0,
            impl_target: None,
            span: SourceSpan::default(),
        };

        let sig = format_func_signature(&func);
        assert_eq!(sig, "private unsafe func dangerous(): void;");
    }

    #[test]
    fn test_format_struct_signature_empty() {
        let s = HirStruct {
            name: StrId::from_static("Empty"),
            visibility: Visibility::Public,
            generics: None,
            fields: &[],
            unmangled_name: StrId::from_static("Empty"),
        };
        let sig = format_struct_signature(&s);
        assert_eq!(sig, "struct Empty {}");
    }

    #[test]
    fn test_format_struct_signature_with_primitive_field() {
        // struct Wrapper { value: i32, }
        let fields: &[HirField<'static, 'static>] = &[HirField {
            name: StrId::from_static("value"),
            field_type: HirType::I32,
            visibility: Visibility::Public,
            manual: false,
        }];
        let s = HirStruct {
            name: StrId::from_static("Wrapper"),
            visibility: Visibility::Public,
            generics: None,
            fields,
            unmangled_name: StrId::from_static("Wrapper"),
        };
        let sig = format_struct_signature(&s);
        assert!(sig.contains("struct Wrapper {"), "got: {sig:?}");
        assert!(sig.contains("value: i32,"), "got: {sig:?}");
    }

    #[test]
    fn test_format_struct_signature_with_array_field() {
        // struct Buffer { data: &[8]u64, }
        let field_type = HirType::Array(&HIR_U64, 8);
        let fields: &[HirField<'static, 'static>] = &[HirField {
            name: StrId::from_static("data"),
            field_type,
            visibility: Visibility::Public,
            manual: false,
        }];
        let s = HirStruct {
            name: StrId::from_static("Buffer"),
            visibility: Visibility::Public,
            generics: None,
            fields,
            unmangled_name: StrId::from_static("Buffer"),
        };
        let sig = format_struct_signature(&s);
        assert!(sig.contains("struct Buffer {"), "got: {sig:?}");
        assert!(sig.contains("data: &[8]u64,"), "got: {sig:?}");
    }

    #[test]
    fn test_format_enum_signature_empty() {
        let e = HirEnum {
            name: StrId::from_static("Color"),
            visibility: Visibility::Public,
            generics: None,
            variants: &[],
            unmangled_name: StrId::from_static("Color"),
        };
        let sig = format_enum_signature(&e);
        assert_eq!(sig, "enum Color {}");
    }

    #[test]
    fn test_format_enum_signature_private() {
        let e = HirEnum {
            name: StrId::from_static("Internal"),
            visibility: Visibility::Private,
            generics: None,
            variants: &[],
            unmangled_name: StrId::from_static("Internal"),
        };
        let sig = format_enum_signature(&e);
        assert_eq!(sig, "private enum Internal {}");
    }

    #[test]
    fn test_doc_comment_above_basic() {
        let source = "/// potential docs\n/// line 2\nextern \"C\" func pthread_mutex_unlock(mutex: [*]mut PthreadMutex): i32;\n";
        let doc = doc_comment_above(source, 3);
        assert_eq!(doc.as_deref(), Some("potential docs\nline 2"));
    }

    #[test]
    fn test_doc_comment_above_through_attribute() {
        let source = "/// docs here\n#[no_mangle]\nextern \"C\" func foo(): void;\n";
        let doc = doc_comment_above(source, 3);
        assert_eq!(doc.as_deref(), Some("docs here"));
    }

    #[test]
    fn test_doc_comment_above_through_modifier() {
        let source = "/// A documented function.\npublic\nfunc bar(): i32;\n";
        let doc = doc_comment_above(source, 3);
        assert_eq!(doc.as_deref(), Some("A documented function."));
    }

    #[test]
    fn test_doc_comment_above_no_comment() {
        let source = "func baz(): void;\n";
        let doc = doc_comment_above(source, 1);
        assert_eq!(doc, None);
    }

    #[test]
    fn test_doc_comment_above_blank_lines_between_doc_and_item() {
        let source = "/// First line.\n\nextern \"C\" func qux(): void;\n";
        let doc = doc_comment_above(source, 3);
        assert_eq!(doc.as_deref(), Some("First line."));
    }

    #[test]
    fn test_format_as_zeta_comments_single_line() {
        let out = format_as_zeta_comments("Hello, world!");
        assert_eq!(out, "/// Hello, world!\n");
    }

    #[test]
    fn test_format_as_zeta_comments_multiline() {
        let out = format_as_zeta_comments("potential docs\nline 2");
        assert_eq!(out, "/// potential docs\n/// line 2\n");
    }

    #[test]
    fn test_format_as_zeta_comments_empty_inner_line() {
        let out = format_as_zeta_comments("first\n\nsecond");
        assert_eq!(out, "/// first\n///\n/// second\n");
    }

    #[test]
    fn test_format_hover_markup_with_doc() {
        let sig = r#"extern "C" func pthread_mutex_unlock(mutex: [*]mut i32): i32;"#;
        let markup = format_hover_markup(sig, Some("potential docs"));
        let expected = concat!(
            "```zeta\n",
            r#"extern "C" func pthread_mutex_unlock(mutex: [*]mut i32): i32;"#,
            "\n```\n\n---\n\npotential docs"
        );
        assert_eq!(markup.value, expected);
    }

    #[test]
    fn test_format_hover_markup_without_doc() {
        let markup = format_hover_markup("func square(x: i32): i32;", None);
        assert_eq!(markup.value, "```zeta\nfunc square(x: i32): i32;\n```");
    }

    #[test]
    fn test_format_hover_markup_whitespace_only_doc_ignored() {
        let markup = format_hover_markup("func noop(): void;", Some("   "));
        assert!(
            !markup.value.contains("---"),
            "unexpected ---; got: {:?}",
            markup.value
        );
        assert!(
            !markup.value.contains("///"),
            "unexpected ///; got: {:?}",
            markup.value
        );
    }

    #[test]
    fn test_lsp_hover_and_goto() {
        let compiler = match zeta_compiler_api::Compiler::new() {
            Ok(c) => c,
            Err(_) => return,
        };
        let stdlib = match compiler_lib_path() {
            Ok(p) => p,
            Err(_) => return,
        };
        let loader = choose_file_loader();

        let mut state = ServerState {
            compiler,
            workspace_root: Some(PathBuf::from("/home/flameyosflow/zeta")),
            documents: Default::default(),
            should_exit: false,
        };
        let _ = state.compiler.load_directory(&loader, &stdlib, true);

        let test_path = PathBuf::from("/tmp/zeta_lsp_test_example.zeta");
        let test_uri = match path_to_uri(&test_path) {
            Ok(u) => u,
            Err(_) => return,
        };

        let test_src = concat!(
            "package test_example;\n",
            "\n",
            "/// Computes the square of a number.\n",
            "func square(x: i32): i32 {\n",
            "    return x * x;\n",
            "}\n",
            "\n",
            "func main(): i32 {\n",
            "    let result: i32 = square(5);\n",
            "    return result;\n",
            "}\n",
        );

        state.documents.insert(
            test_uri.clone(),
            Document {
                version: 1,
                path: test_path.clone(),
                text: test_src.to_string(),
            },
        );
        let _ = state.compiler.open_module(&test_path, test_src.to_string());

        // hover on `square` declaration (LSP 0-indexed: line 3, char 6)
        let hover_params = HoverParams {
            text_document_position_params: TextDocumentPositionParams {
                text_document: TextDocumentIdentifier {
                    uri: test_uri.clone(),
                },
                position: Position {
                    line: 3,
                    character: 6,
                },
            },
            work_done_progress_params: Default::default(),
        };
        if let Some(hover) = hover_at(&state, &hover_params) {
            if let HoverContents::Markup(content) = hover.contents {
                assert!(
                    content.value.contains("func square"),
                    "hover missing 'func square'; got: {:?}",
                    content.value
                );
                assert!(
                    content.value.contains("Computes the square of a number."),
                    "hover missing doc; got: {:?}",
                    content.value
                );
            }
        }

        // goto definition on `square` call in main (LSP 0-indexed: line 8, char 22)
        let goto_params = GotoDefinitionParams {
            text_document_position_params: TextDocumentPositionParams {
                text_document: TextDocumentIdentifier {
                    uri: test_uri.clone(),
                },
                position: Position {
                    line: 8,
                    character: 22,
                },
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };
        if let Some(goto) = go_to_definition(&mut state, &goto_params) {
            match goto {
                lsp_types::GotoDefinitionResponse::Scalar(loc) => {
                    assert_eq!(loc.uri, test_uri, "goto pointed to wrong file");
                    assert_eq!(
                        loc.range.start.line, 3,
                        "goto pointed to wrong line; got {}",
                        loc.range.start.line
                    );
                }
                _ => panic!("Expected GotoDefinitionResponse::Scalar"),
            }
        }
    }
}
