use super::disasm::{BAD_INSTR, Disasm, NO_INSTR, int_target};
use super::ir::{Block, ExprId, Function, Program, Span32, Template, Term};
use super::lift::LiftError;

const NOT_SEEN: u32 = u32::MAX;

#[derive(Clone, Copy, Default)]
struct Flow {
    ok: bool,
    falls: bool,
    ender: bool,
    dynamic: bool,
    jump: bool,
    targets: bool,
    branch: Option<(ExprId, bool)>,
}

fn flow_of(t: &Result<Template, LiftError>) -> Flow {
    let Ok(t) = t else {
        return Flow::default();
    };
    Flow {
        ok: true,
        falls: t.falls,
        ender: t.branch.is_some() || !t.falls || t.dynamic || !t.seeks.is_empty(),
        dynamic: t.dynamic,
        jump: !t.seeks.is_empty(),
        targets: !t.targets.is_empty(),
        branch: t.branch.map(|b| (b.cond, b.when)),
    }
}

pub fn decompile(
    dis: Disasm,
    templates: Vec<Result<Template, LiftError>>,
    strings: Vec<Box<str>>,
    pool: Vec<u16>,
) -> Program {
    let n = dis.instrs.len();
    let mut blocks: Vec<Block> = Vec::with_capacity(n / 4);
    let mut functions: Vec<Function> = Vec::with_capacity(dis.entries.len());
    let mut block_instrs: Vec<u32> = Vec::with_capacity(n + n / 4);
    let mut owner = vec![NOT_SEEN; n];
    let mut preds = vec![0u32; n];
    let mut leader = vec![false; n];
    let mut members: Vec<u32> = Vec::with_capacity(1024);
    let mut work: Vec<u32> = Vec::with_capacity(256);
    let index = &dis.index;
    let at = |pc: u32| -> Option<u32> {
        match index.get(pc as usize) {
            Some(&i) if i != NO_INSTR && i != BAD_INSTR => Some(i - 1),
            _ => None,
        }
    };
    let flows: Vec<Flow> = templates.iter().map(flow_of).collect();
    let flow = |h: u16| -> Flow { flows.get(h as usize).copied().unwrap_or_default() };
    for (fi, &entry) in dis.entries.iter().enumerate() {
        let stamp = fi as u32;
        let Some(root) = at(entry) else {
            continue;
        };
        members.clear();
        work.clear();
        work.push(root);
        owner[root as usize] = stamp;
        preds[root as usize] = 0;
        leader[root as usize] = true;
        while let Some(i) = work.pop() {
            members.push(i);
            let ins = dis.instrs[i as usize];
            let f = flow(ins.handler);
            if !f.ok {
                continue;
            }
            let mut visit = |pc: u32, forced: bool, work: &mut Vec<u32>| {
                if let Some(j) = at(pc) {
                    let j = j as usize;
                    if owner[j] != stamp {
                        owner[j] = stamp;
                        preds[j] = 0;
                        leader[j] = false;
                        work.push(j as u32);
                    }
                    preds[j] += 1;
                    if forced {
                        leader[j] = true;
                    }
                }
            };
            if f.falls {
                visit(ins.next, f.ender, &mut work);
            }
            if let Some(t) = ins.target {
                visit(t, true, &mut work);
            }
            if f.targets
                && let Some(Ok(tpl)) = templates.get(ins.handler as usize)
            {
                let ops = &dis.operands[ins.ops.range()];
                for &s in &tpl.targets {
                    if let Some(t) = ops.get(s as usize).and_then(int_target) {
                        visit(t, true, &mut work);
                    }
                }
            }
        }
        for &i in &members {
            if preds[i as usize] != 1 {
                leader[i as usize] = true;
            }
        }
        leader[root as usize] = true;
        let first_block = blocks.len() as u32;
        for &start in &members {
            if !leader[start as usize] {
                continue;
            }
            let instr_start = block_instrs.len() as u32;
            let mut cur = start;
            let term = loop {
                block_instrs.push(cur);
                let ins = dis.instrs[cur as usize];
                let f = flow(ins.handler);
                if !f.ok {
                    break Term::Invalid;
                }
                if let Some((cond, when)) = f.branch {
                    break match ins.target {
                        Some(target) => Term::Branch {
                            cond,
                            when,
                            target,
                            fall: ins.next,
                        },
                        None => Term::Invalid,
                    };
                }
                if !f.falls {
                    break if f.dynamic { Term::Dynamic { fall: None } } else { Term::Exit };
                }
                if f.dynamic {
                    break Term::Dynamic { fall: Some(ins.next) };
                }
                if f.jump {
                    break Term::Jump(ins.next);
                }
                match at(ins.next) {
                    Some(j) if owner[j as usize] == stamp && !leader[j as usize] => cur = j,
                    Some(_) => break Term::Next(ins.next),
                    None => break Term::Invalid,
                }
            };
            blocks.push(Block {
                pc: dis.instrs[start as usize].pc,
                instrs: Span32 {
                    start: instr_start,
                    len: block_instrs.len() as u32 - instr_start,
                },
                term,
            });
        }
        functions.push(Function {
            entry,
            blocks: Span32 {
                start: first_block,
                len: blocks.len() as u32 - first_block,
            },
        });
    }
    Program {
        strings,
        pool,
        templates,
        instrs: dis.instrs,
        operands: dis.operands,
        block_instrs,
        blocks,
        functions,
    }
}
