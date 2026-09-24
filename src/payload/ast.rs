use oxc_ast::ast::{
    Argument, BindingPattern, Expression, FormalParameters, Function, MemberExpression, Statement,
    UnaryOperator,
};

#[derive(Clone, Copy)]
pub enum Mem<'b, 'a> {
    Static(&'b Expression<'a>, &'a str),
    Computed(&'b Expression<'a>, &'b Expression<'a>),
}

impl<'b, 'a> Mem<'b, 'a> {
    #[inline]
    pub fn object(self) -> &'b Expression<'a> {
        match self {
            Mem::Static(o, _) | Mem::Computed(o, _) => o,
        }
    }
}

#[inline]
pub fn member_view<'b, 'a>(m: &'b MemberExpression<'a>) -> Option<Mem<'b, 'a>> {
    match m {
        MemberExpression::StaticMemberExpression(s) => {
            Some(Mem::Static(&s.object, s.property.name.as_str()))
        }
        MemberExpression::ComputedMemberExpression(c) => Some(Mem::Computed(&c.object, &c.expression)),
        MemberExpression::PrivateFieldExpression(_) => None,
    }
}

#[inline]
pub fn member<'b, 'a>(e: &'b Expression<'a>) -> Option<Mem<'b, 'a>> {
    member_view(e.as_member_expression()?)
}

#[inline]
pub fn ident<'a>(e: &Expression<'a>) -> Option<&'a str> {
    match e {
        Expression::Identifier(id) => Some(id.name.as_str()),
        _ => None,
    }
}

#[inline]
pub fn is_ident(e: &Expression<'_>, name: &str) -> bool {
    matches!(e, Expression::Identifier(id) if id.name.as_str() == name)
}

#[inline]
pub fn num(e: &Expression<'_>) -> Option<f64> {
    match e {
        Expression::NumericLiteral(n) => Some(n.value),
        Expression::UnaryExpression(u) if u.operator == UnaryOperator::UnaryNegation => {
            match &u.argument {
                Expression::NumericLiteral(n) => Some(-n.value),
                _ => None,
            }
        }
        _ => None,
    }
}

#[inline]
pub fn str_lit<'a>(e: &Expression<'a>) -> Option<&'a str> {
    match e {
        Expression::StringLiteral(s) => Some(s.value.as_str()),
        _ => None,
    }
}

#[inline]
pub fn static_prop<'b, 'a>(e: &'b Expression<'a>) -> Option<(&'b Expression<'a>, &'a str)> {
    match member(e)? {
        Mem::Static(o, p) => Some((o, p)),
        Mem::Computed(o, k) => str_lit(k).map(|p| (o, p)),
    }
}

#[inline]
pub fn is_static_call<'b, 'a>(e: &'b Expression<'a>, object: &str, property: &str) -> bool {
    matches!(static_prop(e), Some((o, p)) if p == property && is_ident(o, object))
}

#[inline]
pub fn param_name<'a>(params: &FormalParameters<'a>, index: usize) -> Option<&'a str> {
    match &params.items.get(index)?.pattern {
        BindingPattern::BindingIdentifier(id) => Some(id.name.as_str()),
        _ => None,
    }
}

#[inline]
pub fn param_names<'a>(f: &Function<'a>, out: &mut Vec<&'a str>) -> bool {
    out.clear();
    if f.params.rest.is_some() {
        return false;
    }
    for p in &f.params.items {
        match &p.pattern {
            BindingPattern::BindingIdentifier(id) => out.push(id.name.as_str()),
            _ => return false,
        }
    }
    true
}

#[inline]
pub fn body<'b, 'a>(f: &'b Function<'a>) -> &'b [Statement<'a>] {
    match &f.body {
        Some(b) => b.statements.as_slice(),
        None => &[],
    }
}

#[inline]
pub fn arg<'b, 'a>(args: &'b [Argument<'a>], index: usize) -> Option<&'b Expression<'a>> {
    args.get(index)?.as_expression()
}

#[inline]
pub fn binding_name<'a>(p: &BindingPattern<'a>) -> Option<&'a str> {
    match p {
        BindingPattern::BindingIdentifier(id) => Some(id.name.as_str()),
        _ => None,
    }
}

pub fn returned<'b, 'a>(stmts: &'b [Statement<'a>]) -> Option<&'b Expression<'a>> {
    stmts.iter().find_map(|s| match s {
        Statement::ReturnStatement(r) => r.argument.as_ref(),
        _ => None,
    })
}
