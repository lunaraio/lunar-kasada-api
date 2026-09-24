use oxc_ast::ast::{
    ArrayExpressionElement, Expression, ForStatementInit, ForStatementLeft, Function,
    ObjectPropertyKind, PropertyKey, SimpleAssignmentTarget, Statement, VariableDeclaration,
    VariableDeclarator,
};

use super::ast::member;

pub trait Sink<'a> {
    fn enter_fn(&mut self, _f: &'a Function<'a>) -> bool {
        true
    }
    fn exit_fn(&mut self, _f: &'a Function<'a>) {}
    fn stmt(&mut self, _s: &'a Statement<'a>) {}
    fn expr(&mut self, _e: &'a Expression<'a>) {}
    fn decl(&mut self, _d: &'a VariableDeclarator<'a>) {}
}

pub fn function<'a, S: Sink<'a>>(f: &'a Function<'a>, s: &mut S) {
    if s.enter_fn(f) {
        if let Some(b) = &f.body {
            stmts(&b.statements, s);
        }
        s.exit_fn(f);
    }
}

pub fn stmts<'a, S: Sink<'a>>(list: &'a [Statement<'a>], s: &mut S) {
    for st in list {
        stmt(st, s);
    }
}

fn var_decl<'a, S: Sink<'a>>(d: &'a VariableDeclaration<'a>, s: &mut S) {
    for decl in &d.declarations {
        s.decl(decl);
        if let Some(init) = &decl.init {
            expr(init, s);
        }
    }
}

pub fn stmt<'a, S: Sink<'a>>(st: &'a Statement<'a>, s: &mut S) {
    s.stmt(st);
    match st {
        Statement::BlockStatement(b) => stmts(&b.body, s),
        Statement::ExpressionStatement(e) => expr(&e.expression, s),
        Statement::IfStatement(i) => {
            expr(&i.test, s);
            stmt(&i.consequent, s);
            if let Some(a) = &i.alternate {
                stmt(a, s);
            }
        }
        Statement::ForStatement(f) => {
            match &f.init {
                Some(ForStatementInit::VariableDeclaration(d)) => var_decl(d, s),
                Some(other) => {
                    if let Some(e) = other.as_expression() {
                        expr(e, s);
                    }
                }
                None => {}
            }
            if let Some(t) = &f.test {
                expr(t, s);
            }
            if let Some(u) = &f.update {
                expr(u, s);
            }
            stmt(&f.body, s);
        }
        Statement::ForInStatement(f) => {
            if let ForStatementLeft::VariableDeclaration(d) = &f.left {
                var_decl(d, s);
            }
            expr(&f.right, s);
            stmt(&f.body, s);
        }
        Statement::ForOfStatement(f) => {
            if let ForStatementLeft::VariableDeclaration(d) = &f.left {
                var_decl(d, s);
            }
            expr(&f.right, s);
            stmt(&f.body, s);
        }
        Statement::WhileStatement(w) => {
            expr(&w.test, s);
            stmt(&w.body, s);
        }
        Statement::DoWhileStatement(w) => {
            stmt(&w.body, s);
            expr(&w.test, s);
        }
        Statement::ReturnStatement(r) => {
            if let Some(a) = &r.argument {
                expr(a, s);
            }
        }
        Statement::ThrowStatement(t) => expr(&t.argument, s),
        Statement::TryStatement(t) => {
            stmts(&t.block.body, s);
            if let Some(h) = &t.handler {
                stmts(&h.body.body, s);
            }
            if let Some(f) = &t.finalizer {
                stmts(&f.body, s);
            }
        }
        Statement::LabeledStatement(l) => stmt(&l.body, s),
        Statement::SwitchStatement(sw) => {
            expr(&sw.discriminant, s);
            for case in &sw.cases {
                if let Some(t) = &case.test {
                    expr(t, s);
                }
                stmts(&case.consequent, s);
            }
        }
        Statement::VariableDeclaration(d) => var_decl(d, s),
        Statement::FunctionDeclaration(f) => function(f, s),
        _ => {}
    }
}

pub fn expr<'a, S: Sink<'a>>(e: &'a Expression<'a>, s: &mut S) {
    s.expr(e);
    match e {
        Expression::ArrayExpression(a) => {
            for el in &a.elements {
                match el {
                    ArrayExpressionElement::SpreadElement(sp) => expr(&sp.argument, s),
                    ArrayExpressionElement::Elision(_) => {}
                    other => {
                        if let Some(x) = other.as_expression() {
                            expr(x, s);
                        }
                    }
                }
            }
        }
        Expression::ObjectExpression(o) => {
            for p in &o.properties {
                match p {
                    ObjectPropertyKind::ObjectProperty(p) => {
                        if p.computed
                            && let Some(k) = p.key.as_expression()
                        {
                            expr(k, s);
                        }
                        if !matches!(p.key, PropertyKey::PrivateIdentifier(_)) {
                            expr(&p.value, s);
                        }
                    }
                    ObjectPropertyKind::SpreadProperty(sp) => expr(&sp.argument, s),
                }
            }
        }
        Expression::AssignmentExpression(a) => {
            if let Some(m) = a.left.as_member_expression() {
                match m {
                    oxc_ast::ast::MemberExpression::StaticMemberExpression(x) => expr(&x.object, s),
                    oxc_ast::ast::MemberExpression::ComputedMemberExpression(x) => {
                        expr(&x.object, s);
                        expr(&x.expression, s);
                    }
                    oxc_ast::ast::MemberExpression::PrivateFieldExpression(x) => expr(&x.object, s),
                }
            }
            expr(&a.right, s);
        }
        Expression::BinaryExpression(b) => {
            expr(&b.left, s);
            expr(&b.right, s);
        }
        Expression::LogicalExpression(l) => {
            expr(&l.left, s);
            expr(&l.right, s);
        }
        Expression::ConditionalExpression(c) => {
            expr(&c.test, s);
            expr(&c.consequent, s);
            expr(&c.alternate, s);
        }
        Expression::CallExpression(c) => {
            expr(&c.callee, s);
            for a in &c.arguments {
                if let Some(x) = a.as_expression() {
                    expr(x, s);
                }
            }
        }
        Expression::NewExpression(c) => {
            expr(&c.callee, s);
            for a in &c.arguments {
                if let Some(x) = a.as_expression() {
                    expr(x, s);
                }
            }
        }
        Expression::SequenceExpression(q) => {
            for x in &q.expressions {
                expr(x, s);
            }
        }
        Expression::UnaryExpression(u) => expr(&u.argument, s),
        Expression::UpdateExpression(u) => {
            if let SimpleAssignmentTarget::AssignmentTargetIdentifier(_) = &u.argument {
            } else if let Some(m) = u.argument.as_member_expression() {
                match m {
                    oxc_ast::ast::MemberExpression::StaticMemberExpression(x) => expr(&x.object, s),
                    oxc_ast::ast::MemberExpression::ComputedMemberExpression(x) => {
                        expr(&x.object, s);
                        expr(&x.expression, s);
                    }
                    oxc_ast::ast::MemberExpression::PrivateFieldExpression(x) => expr(&x.object, s),
                }
            }
        }
        Expression::FunctionExpression(f) => function(f, s),
        Expression::ParenthesizedExpression(p) => expr(&p.expression, s),
        Expression::ChainExpression(_) => {}
        _ => {
            if let Some(m) = member(e) {
                match m {
                    super::ast::Mem::Static(o, _) => expr(o, s),
                    super::ast::Mem::Computed(o, k) => {
                        expr(o, s);
                        expr(k, s);
                    }
                }
            }
        }
    }
}
