use proc_macro::TokenStream;
use proc_macro2::TokenStream as TokenStream2;
use quote::{format_ident, quote};
use syn::{
    Attribute, Data, DeriveInput, Error, Fields, GenericParam, LitStr, Path, Result,
    parse_macro_input, parse_quote, spanned::Spanned,
};

#[proc_macro_derive(StableHash, attributes(stable_hash))]
pub fn derive_stable_hash(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    expand(input)
        .unwrap_or_else(Error::into_compile_error)
        .into()
}

fn container_crate(attrs: &[Attribute]) -> Result<Path> {
    let mut out: Path = parse_quote!(::ir::ir_hasher);
    for attr in attrs.iter().filter(|a| a.path().is_ident("stable_hash")) {
        attr.parse_nested_meta(|m| {
            if m.path.is_ident("crate") {
                let s: LitStr = m.value()?.parse()?;
                out = s.parse()?;
                Ok(())
            } else {
                Err(m.error("unknown container attribute; expected `crate = \"path\"`"))
            }
        })?;
    }
    Ok(out)
}

fn field_skipped(attrs: &[Attribute]) -> Result<bool> {
    let mut skip = false;
    for attr in attrs.iter().filter(|a| a.path().is_ident("stable_hash")) {
        attr.parse_nested_meta(|m| {
            if m.path.is_ident("skip") {
                skip = true;
                Ok(())
            } else {
                Err(m.error("unknown field attribute; expected `skip`"))
            }
        })?;
    }
    Ok(skip)
}

/// Generates the statements hashing `fields`, given the expression for each field.
fn hash_fields(
    krate: &Path,
    fields: &Fields,
    access: impl Fn(usize, &syn::Field) -> TokenStream2,
) -> Result<TokenStream2> {
    let mut stmts = Vec::new();
    for (i, f) in fields.iter().enumerate() {
        if field_skipped(&f.attrs)? {
            continue;
        }
        let a = access(i, f);
        stmts.push(quote! { #krate::StableHash::stable_hash(#a, cx, h); });
    }
    Ok(quote! { #(#stmts)* })
}

fn expand(input: DeriveInput) -> Result<TokenStream2> {
    let krate = container_crate(&input.attrs)?;
    let name = &input.ident;

    let mut generics = input.generics.clone();
    for p in generics.params.iter_mut() {
        if let GenericParam::Type(t) = p {
            t.bounds.push(parse_quote!(#krate::StableHash));
        }
    }
    let (impl_g, ty_g, where_g) = generics.split_for_impl();

    let body = match &input.data {
        Data::Struct(s) => hash_fields(&krate, &s.fields, |i, f| match &f.ident {
            Some(id) => quote!(&self.#id),
            None => {
                let idx = syn::Index::from(i);
                quote!(&self.#idx)
            }
        })?,
        Data::Enum(e) => {
            let mut arms = Vec::new();
            for (idx, v) in e.variants.iter().enumerate() {
                let tag = idx as u32;
                let vname = &v.ident;
                let arm = match &v.fields {
                    Fields::Unit => quote! { Self::#vname => { h.u32(#tag); } },
                    Fields::Named(n) => {
                        let binds: Vec<_> = n
                            .named
                            .iter()
                            .map(|f| {
                                let id = f.ident.as_ref().unwrap();
                                if field_skipped(&f.attrs).unwrap_or(false) {
                                    quote!(#id: _)
                                } else {
                                    quote!(#id)
                                }
                            })
                            .collect();
                        let stmts = hash_fields(&krate, &v.fields, |_, f| {
                            let id = f.ident.as_ref().unwrap();
                            quote!(#id)
                        })?;
                        quote! { Self::#vname { #(#binds),* } => { h.u32(#tag); #stmts } }
                    }
                    Fields::Unnamed(u) => {
                        let binds: Vec<_> = u
                            .unnamed
                            .iter()
                            .enumerate()
                            .map(|(i, f)| {
                                if field_skipped(&f.attrs).unwrap_or(false) {
                                    quote!(_)
                                } else {
                                    let id = format_ident!("f{}", i);
                                    quote!(#id)
                                }
                            })
                            .collect();
                        let stmts = hash_fields(&krate, &v.fields, |i, _| {
                            let id = format_ident!("f{}", i);
                            quote!(#id)
                        })?;
                        quote! { Self::#vname ( #(#binds),* ) => { h.u32(#tag); #stmts } }
                    }
                };
                arms.push(arm);
            }
            quote! { match self { #(#arms)* } }
        }
        Data::Union(u) => {
            return Err(Error::new(
                u.union_token.span(),
                "StableHash cannot be derived for unions",
            ));
        }
    };

    Ok(quote! {
        impl #impl_g #krate::StableHash for #name #ty_g #where_g {
            #[allow(unused_variables)]
            fn stable_hash(&self, cx: &#krate::HashCtx<'_>, h: &mut #krate::FpHasher) {
                #body
            }
        }
    })
}
