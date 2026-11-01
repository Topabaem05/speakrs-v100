//! PTX lint: reject generic shared addresses narrowed without conversion to shared space,
//! and instructions an `sm_70` module cannot contain

use std::collections::{BTreeMap, BTreeSet};

use eyre::{Result, bail};

/// Opcode prefixes that need sm_75 or newer. cuda-oxide rejects some of them when it
/// builds for sm_70, but inline PTX such as `mma.sync.aligned.m16n8k8` slips through,
/// and then only ptxas or the driver JIT would fail, on the GPU box or at load time
const SM75_PLUS_OPCODES: &[&str] = &[
    // sm_75
    "ldmatrix",
    "movmatrix",
    "mma.sync.aligned.m16n8",
    "tanh.approx",
    "ex2.approx.f16",
    // sm_80 and newer
    "cp.async",
    "cp.reduce.async",
    "cvt.rna.tf32",
    "cvt.rn.f16x2.f32",
    "cvt.rn.bf16",
    "redux.sync",
    "mbarrier",
    "elect.sync",
    "griddepcontrol",
    "setmaxnreg",
    "stmatrix",
    "wgmma",
    "tcgen05",
];

/// Fails when an `sm_70` module uses an instruction that needs sm_75 or newer, with
/// each offending statement
pub fn check_sm70_opcodes(ptx: &str) -> Result<()> {
    let statements = sm75_plus_statements(ptx);
    if !statements.is_empty() {
        bail!(
            "sm_70 PTX uses instructions that need sm_75 or newer:\n    {}",
            statements.join("\n    ")
        );
    }

    Ok(())
}

fn sm75_plus_statements(ptx: &str) -> Vec<String> {
    let code = without_comments(ptx);
    // directives end at the line, not at a semicolon, and inline asm braces put several
    // statements on one line
    code.split([';', '{', '}', '\n'])
        .filter_map(|statement| {
            // skip labels such as `$L__BB0_2:` and guards such as `@%p1` or `@!%p1`
            let opcode = statement
                .split_whitespace()
                .find(|word| !word.ends_with(':') && !word.starts_with('@'))?;
            // an operand line, such as a `call` target LLVM puts on its own line
            if opcode.ends_with(',') {
                return None;
            }
            let sm75_plus = SM75_PLUS_OPCODES
                .iter()
                .any(|banned| opcode.starts_with(banned))
                // the f16 m8n8k4 shape is Volta's own; only the f64 one needs sm_80
                || (opcode.starts_with("mma.sync.aligned.m8n8k4") && opcode.contains(".f64"));
            sm75_plus.then(|| statement.split_whitespace().collect::<Vec<_>>().join(" "))
        })
        .collect()
}

/// Fails when a generic shared address reaches a narrowing conversion without
/// `cvta.to.shared`, following values within each PTX function
///
/// PTX register reuse and branches require a conservative fixed point: an address
/// that reaches a narrowing conversion on any path is unsafe; never carry taint
/// between functions, whose register names are independent
pub fn check_shared_truncation(ptx: &str) -> Result<()> {
    let sites = shared_truncations(ptx);
    if !sites.is_empty() {
        bail!(
            "generic shared address reaches a narrowing conversion; use cvta.to.shared before narrowing:\n    {}",
            sites.join("\n    ")
        );
    }

    Ok(())
}

fn shared_truncations(ptx: &str) -> Vec<String> {
    let code = without_comments(ptx);
    let tokens: Vec<_> = code
        .split(|c: char| c.is_whitespace() || "(),;{}:".contains(c))
        .filter(|token| !token.is_empty())
        .collect();
    let mut sites = Vec::new();
    let mut rest = code.as_str();
    for token in &tokens {
        if !matches!(*token, ".entry" | ".func") {
            continue;
        }

        let Some(start) = rest.find(token) else {
            break;
        };
        rest = &rest[start + token.len()..];
        // return parameters precede the name of a .func
        let declaration = rest.trim_start();
        let declaration = if declaration.starts_with('(') {
            declaration
                .split_once(')')
                .map_or("", |(_, tail)| tail)
                .trim_start()
        } else {
            declaration
        };
        let name = declaration
            .split(|c: char| c.is_whitespace() || c == '(')
            .next()
            .unwrap_or("?");
        let Some(body_start) = rest.find('{') else {
            continue;
        };
        let before_body = &rest[..body_start];
        // an external declaration has no body; the brace belongs to another function
        if before_body.contains(';')
            || before_body.contains(".entry")
            || before_body.contains(".func")
        {
            continue;
        }

        let mut depth = 1;
        let tail = &rest[body_start + 1..];
        let Some(body_end) = tail.char_indices().find_map(|(offset, c)| {
            depth += i32::from(c == '{') - i32::from(c == '}');
            (depth == 0).then_some(offset)
        }) else {
            continue;
        };
        let instructions: Vec<_> = tail[..body_end]
            .split(';')
            .filter_map(Instruction::parse)
            .collect();
        // register reuse must not hide an earlier wide definition
        let mut widths = BTreeMap::new();
        for instruction in &instructions {
            if !instruction.has_destination() {
                continue;
            }
            for register in instruction.destinations() {
                let width = instruction.destination_width();
                widths
                    .entry(register)
                    .and_modify(|previous: &mut Width| {
                        *previous = previous.join(width);
                    })
                    .or_insert(width);
            }
        }
        let mut tainted = BTreeSet::new();
        for instruction in &instructions {
            if instruction.generic_shared() {
                tainted.extend(instruction.destinations());
            }
        }

        loop {
            let mut changed = false;
            for instruction in &instructions {
                if instruction.sanitizes() || !instruction.has_destination() {
                    continue;
                }
                if instruction.sources().any(|source| tainted.contains(source)) {
                    for destination in instruction.destinations() {
                        changed |= tainted.insert(destination);
                    }
                }
            }
            if !changed {
                break;
            }
        }

        for instruction in &instructions {
            if instruction.generic_shared() && instruction.opcode.ends_with(".u32") {
                sites.extend(
                    instruction
                        .destinations()
                        .map(|register| format!("{name}: {register}")),
                );
                continue;
            }
            if let Some(destination_width) = instruction.narrowing_width() {
                sites.extend(
                    instruction
                        .sources()
                        .filter(|source| {
                            tainted.contains(source)
                                && instruction
                                    .conversion_source_width()
                                    .or_else(|| widths.get(source).copied())
                                    .unwrap_or(Width::Unknown)
                                    .can_exceed(destination_width)
                        })
                        .map(|source| format!("{name}: {source}")),
                );
            }
        }
        rest = &tail[body_end + 1..];
    }

    sites
}

struct Instruction<'a> {
    opcode: &'a str,
    operands: Vec<Vec<&'a str>>,
}

impl<'a> Instruction<'a> {
    fn parse(statement: &'a str) -> Option<Self> {
        let statement = statement.trim();
        // declarations and block labels can precede the instruction in a statement
        let opcode = statement
            .split(|c: char| c.is_whitespace() || "{}:".contains(c))
            .find(|word| {
                !word.starts_with('.')
                    && (word.contains('.') || *word == "call")
                    && !word.starts_with('@')
            })?;
        let (_, operands) = statement.split_once(opcode)?;
        let is_call = opcode.split('.').next() == Some("call");
        let has_returns = !is_call || operands.trim_start().starts_with('(');
        let mut depth = 0;
        let mut operands: Vec<Vec<_>> = operands
            .split(|c| {
                if matches!(c, '{' | '(') {
                    depth += 1;
                }
                if matches!(c, '}' | ')') {
                    depth -= 1;
                }
                c == ',' && depth == 0
            })
            .map(|operand| {
                operand
                    .split(|c: char| c.is_whitespace() || "{}(),[]|+-".contains(c))
                    .filter(|word| {
                        word.starts_with('%')
                            || word.starts_with('$')
                            || word.starts_with('_')
                            || word.starts_with(|c: char| c.is_ascii_alphabetic())
                    })
                    .collect()
            })
            .collect();
        // a call without return operands must not treat its callee as a destination
        if !has_returns {
            operands.insert(0, Vec::new());
        }
        Some(Self { opcode, operands })
    }

    fn generic_shared(&self) -> bool {
        self.opcode.starts_with("cvta.shared.")
    }

    fn sanitizes(&self) -> bool {
        self.opcode.starts_with("cvta.to.shared.")
    }

    fn has_destination(&self) -> bool {
        // memory writes and control flow have no register result
        !matches!(
            self.opcode.split('.').next(),
            Some("st" | "red" | "bra" | "brx" | "ret" | "exit")
        )
    }

    fn destinations(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.operands.first().into_iter().flatten().copied()
    }

    fn sources(&self) -> impl Iterator<Item = &'a str> + '_ {
        self.operands.iter().skip(1).flatten().copied()
    }

    fn scalar_widths(&self) -> impl Iterator<Item = usize> + '_ {
        self.opcode.split('.').filter_map(|part| {
            let (kind, digits) = part.split_at_checked(1)?;
            matches!(kind, "u" | "s" | "b" | "f")
                .then(|| digits.parse::<usize>().ok())
                .flatten()
        })
    }

    fn destination_width(&self) -> Width {
        let Some(mut width) = self.scalar_widths().next() else {
            return Width::Unknown;
        };
        let operation = self.opcode.split('.').next();
        if matches!(operation, Some("mul" | "mad"))
            && self.opcode.split('.').any(|part| part == "wide")
        {
            width *= 2;
        }
        // packed moves split a scalar; vector loads give each register the full width
        if operation == Some("mov") {
            width /= self
                .operands
                .first()
                .map_or(1, |registers| registers.len().max(1));
        }

        Width::Bits(width)
    }

    fn conversion_source_width(&self) -> Option<Width> {
        (self.opcode.split('.').next() == Some("cvt"))
            .then(|| self.scalar_widths().nth(1).map(Width::Bits))
            .flatten()
    }

    fn narrowing_width(&self) -> Option<usize> {
        if !matches!(self.opcode.split('.').next(), Some("cvt" | "mov")) {
            return None;
        }
        match self.destination_width() {
            Width::Bits(width) => Some(width),
            Width::Unknown => None,
        }
    }
}

#[derive(Clone, Copy)]
enum Width {
    Bits(usize),
    Unknown,
}

impl Width {
    fn join(self, other: Self) -> Self {
        match (self, other) {
            (Self::Bits(left), Self::Bits(right)) => Self::Bits(left.max(right)),
            _ => Self::Unknown,
        }
    }

    fn can_exceed(self, destination: usize) -> bool {
        match self {
            Self::Bits(source) => source > destination,
            Self::Unknown => true,
        }
    }
}

fn without_comments(ptx: &str) -> String {
    let mut code = String::new();
    let mut chars = ptx.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '/' {
            code.push(c);
            continue;
        }
        match chars.peek() {
            Some('/') => {
                chars.next();
                for c in chars.by_ref() {
                    if c == '\n' {
                        code.push(c);
                        break;
                    }
                }
            }
            Some('*') => {
                chars.next();
                while let Some(c) = chars.next() {
                    if c == '*' && chars.peek() == Some(&'/') {
                        chars.next();
                        break;
                    }
                }
                code.push(' ');
            }
            _ => code.push(c),
        }
    }
    code
}

#[cfg(test)]
mod tests {
    use super::{check_shared_truncation, check_sm70_opcodes, shared_truncations};

    #[test]
    fn sm70_lint_rejects_turing_and_ampere_instructions_wherever_they_appear() {
        // the speakrs h16 shapes, a guarded and a labelled statement, and inline asm
        // braces that put several statements on one line
        let banned = [
            "\tmma.sync.aligned.m16n8k8.row.col.f32.f16.f16.f32 {%f1, %f2, %f3, %f4}, {%r1, %r2}, {%r3}, {%f1, %f2, %f3, %f4};",
            "\tldmatrix.sync.aligned.m8n8.x4.shared.b16 {%r1, %r2, %r3, %r4}, [%r5];",
            "\t@%p1 cp.async.ca.shared.global [%r1], [%rd2], 4;",
            "$L__BB0_3:\n\tredux.sync.add.u32 %r1, %r2, -1;",
            "\t{ .reg .u64 g; cvta.to.global.u64 g, %rd1; cp.async.ca.shared.global [%r1], [g], 4; }",
            "\tcvt.rna.tf32.f32 %r1, %f1;",
            "\tmma.sync.aligned.m8n8k4.row.col.f64.f64.f64.f64 {%fd1, %fd2}, {%fd3}, {%fd4}, {%fd1, %fd2};",
        ];
        for statement in banned {
            let ptx = format!(".version 6.5\n.target sm_70\n{statement}\n");
            let error = check_sm70_opcodes(&ptx).expect_err(statement).to_string();
            assert!(error.contains("need sm_75 or newer"), "{error}");
        }
    }

    #[test]
    fn sm70_lint_accepts_volta_instructions_and_comments() {
        let ptx = ".version 6.5\n.target sm_70\n\
            // ldmatrix in a comment is not an instruction\n\
            /* cp.async either */\n\
            \twmma.mma.sync.aligned.row.col.m16n16k16.f32.f32 {%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8}, {%r1, %r2, %r3, %r4, %r5, %r6, %r7, %r8}, {%r9, %r10, %r11, %r12, %r13, %r14, %r15, %r16}, {%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8};\n\
            \tmma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32 {%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8}, {%r1, %r2}, {%r3, %r4}, {%f1, %f2, %f3, %f4, %f5, %f6, %f7, %f8};\n\
            \tshfl.sync.bfly.b32 %r1, %r2, 16, 31, -1;\n\
            \tcvt.rn.f16.f32 %rs1, %f1;\n\
            \tred.global.or.b32 [%rd1], 1;\n\
            \tcall.uni (retval0), \n\
            \tldmatrix_helper, \n\
            \t(\n\
            \tparam0\n\
            \t);\n";
        check_sm70_opcodes(ptx).expect("Volta-legal PTX");
    }

    #[test]
    fn pre_fix_lstm_has_all_eight_truncations() {
        let fixture = include_str!("../tests/fixtures/ptx/shared-truncation-lstm.ptx");
        let sites = shared_truncations(fixture);
        assert_eq!(
            sites,
            [
                "%rd69", "%rd79", "%rd94", "%rd104", "%rd115", "%rd126", "%rd143", "%rd145"
            ]
            .map(|register| format!("spk_lstm_projection_tf32: {register}"))
        );
        assert!(check_shared_truncation(fixture).is_err());
    }

    #[test]
    fn all_shared_propagation_and_truncation_paths_are_checked() {
        let fixture = include_str!("../tests/fixtures/ptx/shared-truncation-paths.ptx");
        let mut expected: Vec<_> = (0..8).map(|index| format!("path{index}: %out")).collect();
        expected.extend(
            [
                "shared32: %base",
                "signed: %base",
                "bits: %base",
                "move32: %base",
                "vector: %base",
                "vector: %wide",
                "backwards: %copy",
                "shared32_widened: %base",
                "shared32_widened: %wide",
                "named: %copy",
                "signed_mad: %out",
                "signed_mul: %out",
                "rzi: %base",
                "call_return: %out",
                "call_returns: %first",
                "call_returns: %second",
            ]
            .map(str::to_owned),
        );
        assert_eq!(shared_truncations(fixture), expected);
        assert!(check_shared_truncation(fixture).is_err());
    }

    #[test]
    fn dollar_wide_vector_and_subword_paths_are_rejected() {
        let fixture = include_str!("../tests/fixtures/ptx/shared-truncation-review.ptx");
        assert_eq!(
            shared_truncations(fixture),
            [
                "dollar_origin: $base",
                "dollar_copy: $copy",
                "wide_mad: %wide",
                "wide_mul: %base",
                "wide_mul: %wide",
                "vector_load: %first",
                "vector_load: %second",
                "unsigned16: %base",
                "signed16: %base",
                "unsigned8: %base",
                "packed16: %base",
                "unknown_reuse: %value",
            ]
        );
        assert!(check_shared_truncation(fixture).is_err());
    }

    #[test]
    fn converted_shared_addresses_and_function_local_registers_pass() {
        let fixture = include_str!("../tests/fixtures/ptx/shared-truncation-safe.ptx");
        assert!(check_shared_truncation(fixture).is_ok());
    }

    #[test]
    fn both_arithmetic_operands_predicates_and_transitive_moves_are_checked() {
        let fixture = ".visible .entry unsafe() {\ncvta.shared.u64 %base, smem;\nmov.b64 %copy, %base;\nadd.u64 %offset, 16, %copy;\nsub.s64 %address, 32, %offset;\n@%p cvt.u32.u64 %r, %address;\n}";
        assert_eq!(shared_truncations(fixture), ["unsafe: %address"]);
    }

    #[test]
    fn device_functions_and_external_declarations_are_isolated() {
        let fixture = ".extern .func external();\n.func (.param .b32 result) unsafe_func() { cvta.shared.u64 %base, smem; cvt.u32.u64 %r, %base; }\n.visible .entry safe() { mov.u64 %base, 1; cvt.u32.u64 %r, %base; }";
        assert_eq!(shared_truncations(fixture), ["unsafe_func: %base"]);
    }
}
