from collections.abc import Callable
from enum import Enum
from typing import ClassVar

from isa import ArgType, BranchMark, Instruction, MarkedInstruction, Opcode, dec, inc, init
from tokenizer import Token, Tokenizer, TokenType
from translator import Translator

# ===-------------------------------------------
# Ast Errors
# ===-------------------------------------------


class AstError(Exception):
    def __init__(self, tokenizer, msg):
        self.tokenizer = tokenizer
        super().__init__(f"{tokenizer.line} | error: {msg}")


class AstTypeMismatchError(AstError):
    def __init__(self, tokenizer, actual, expected):
        self.expected = expected
        self.actual = actual
        super().__init__(
            tokenizer, f"Expected {expected} but got {actual} | No valid binary operator for {expected} & {actual}"
        )


class AstUnresolvedVarError(AstError):
    def __init__(self, tokenizer, var_name):
        self.var_name = var_name
        super().__init__(tokenizer, f"Unresolved variable {self.var_name}")


class AstUnexpectedTokenError(AstError):
    def __init__(self, tokenizer, actual):
        self.actual = actual
        super().__init__(tokenizer, f"No known parsers for {self.actual.token_type}")


class AstExpectedTokenError(AstError):
    def __init__(self, tokenizer, actual, expected):
        self.expected = expected
        self.actual = actual
        super().__init__(tokenizer, f"Expected {expected} token but got {actual}")


class AstExpectedConditionalExpressionError(AstError):
    def __init__(self, tokenizer, actual):
        self.actual = actual
        super().__init__(tokenizer, f"Expected conditional expression but got {actual}")


class AstExpectedArgumentError(AstError):
    def __init__(self, tokenizer, actual):
        self.actual = actual
        super().__init__(tokenizer, f"Expected argument but got {actual}")


class AstCreatingPrototypeError(AstError):
    def __init__(self, tokenizer):
        super().__init__(tokenizer, "Prototype creation failed")


# ===-------------------------------------------
# Ast Core
# ===-------------------------------------------


class AstExprType(Enum):
    NUM = 1
    STRING = 2
    CHAR = 3
    NONE = 4


class AstExpr:
    def get_expr_type(self) -> AstExprType:
        return AstExprType.NONE

    def __str__(self):
        return "Empty expression"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        raise NotImplementedError()


class AstEchoable(AstExpr):
    pass


class AstLit:
    def __init__(self, value) -> None:
        self.value = value


class AstLitNumber(AstEchoable, AstLit):
    def __init__(self, value: int) -> None:
        super().__init__(value)

    def get_expr_type(self) -> AstExprType:
        return AstExprType.NUM

    def __str__(self) -> str:
        return f"Number Literal {self.value}"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        return [Instruction(Opcode.LD, self.value, ArgType.IMMEDIATE)]


class AstLitString(AstEchoable, AstLit):
    def __init__(self, value: str) -> None:
        super().__init__(value)

    def get_expr_type(self) -> AstExprType:
        return AstExprType.STRING

    @staticmethod
    def serialize(string: str) -> list[int]:
        return [len(string), *list(map(ord, string))]

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions: list[Instruction | BranchMark | MarkedInstruction] = list()
        serialized_string = self.serialize(self.value)
        instructions.append(
            Instruction(
                Opcode.LD, translator.allocate_data(len(serialized_string), serialized_string), ArgType.IMMEDIATE
            )
        )
        return instructions

    def __str__(self) -> str:
        return f'String Literal "{self.value}"'


class AstVar(AstEchoable):
    def __init__(self, name: str, var_type: AstExprType, unique_id: str | None = None) -> None:
        self.name = name
        self.var_type = var_type
        # For future ( to make context variables )
        self.unique_id = unique_id

    def get_unique_id(self) -> str:
        if self.unique_id is None:
            return self.name
        return self.unique_id

    def get_expr_type(self) -> AstExprType:
        return self.var_type

    def __str__(self) -> str:
        return f"Variable {self.name}: {self.var_type.name}"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions: list[Instruction | BranchMark | MarkedInstruction] = list()
        instructions.append(Instruction(Opcode.LD, translator.get_var_addr(self.name), ArgType.DIRECT))
        return instructions

    def get_var_addr(self, translator: Translator) -> int:
        return translator.get_var_addr(self.name)


class AstBinary(AstExpr):
    opertator2opcode: ClassVar[dict] = {
        "+": Opcode.ADD,
        "-": Opcode.SUB,
        "*": Opcode.MUL,
        "/": Opcode.DIV,
        "%": Opcode.MOD,
        "&": Opcode.AND,
        "|": Opcode.OR,
    }

    def __init__(self, op: Token, left: AstExpr, right: AstExpr) -> None:
        self.left = left
        self.right = right
        self.op = op

    def get_expr_type(self):
        return self.left.get_expr_type()

    def __str__(self) -> str:
        return f"{self.left!s} {self.op.value} {self.right!s}"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions = []

        if isinstance(self.right, AstVar):
            right = translator.get_var_addr(self.right.name)
            right_is_direct = True
        elif isinstance(self.right, AstLit) and isinstance(self.left, AstLit):
            right = self.right.value
            right_is_direct = False
        else:
            instructions += self.right.translate(translator)
            right = translator.allocate_for_tmp_expr()
            instructions.append(Instruction(Opcode.ST, right, ArgType.IMMEDIATE))
            right_is_direct = True

        instructions += self.left.translate(translator)

        if self.op.value in self.opertator2opcode.keys():
            instructions.append(
                Instruction(
                    self.opertator2opcode[self.op.value],
                    right,
                    ArgType.DIRECT if right_is_direct else ArgType.IMMEDIATE,
                )
            )
        elif self.op.token_type == TokenType.GT:
            instructions.append(
                Instruction(Opcode.CMP, right, ArgType.DIRECT if right_is_direct else ArgType.IMMEDIATE)
            )
            instructions.append(Instruction(Opcode.SETG))
        elif self.op.token_type == TokenType.LT:
            instructions.append(
                Instruction(Opcode.CMP, right, ArgType.DIRECT if right_is_direct else ArgType.IMMEDIATE)
            )
            instructions.append(Instruction(Opcode.SETL))
        elif self.op.token_type == TokenType.EQ:
            instructions.append(
                Instruction(Opcode.CMP, right, ArgType.DIRECT if right_is_direct else ArgType.IMMEDIATE)
            )
            instructions.append(Instruction(Opcode.SETE))
        elif self.op.token_type == TokenType.NEQ:
            instructions.append(
                Instruction(Opcode.CMP, right, ArgType.DIRECT if right_is_direct else ArgType.IMMEDIATE)
            )
            instructions.append(Instruction(Opcode.SETE))
            instructions.append(Instruction(Opcode.NOT))
        else:
            raise AstUnexpectedTokenError(self.op, self.op.token_type)

        return instructions


class AstDecl(AstExpr):
    def __init__(self, name: str, expr: AstExpr) -> None:
        self.var = AstVar(name, expr.get_expr_type())
        self.expr = expr

    def get_expr_type(self) -> AstExprType:
        return self.var.get_expr_type()

    def __str__(self) -> str:
        return f"{self.var.name!s} = {self.expr!s}"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions: list[Instruction | BranchMark | MarkedInstruction] = list()
        instructions += self.expr.translate(translator)
        instructions.append(Instruction(Opcode.ST, translator.allocate_var(self.var.name), ArgType.IMMEDIATE))
        return instructions


class AstBlockBody(AstExpr):
    def __init__(self, ast_list: list[AstExpr]) -> None:
        self.ast_list = ast_list

    def __str__(self) -> str:
        result = ""
        for ast in self.ast_list:
            result += str(ast) + "\n"
        return result

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions = list()
        for ast in self.ast_list:
            instructions += ast.translate(translator)
        return instructions


class AstIf(AstExpr):
    def __init__(self, expr: AstExpr, conditional_body: AstBlockBody) -> None:
        self.expr = expr
        self.condition_body = conditional_body

    def __str__(self) -> str:
        return f"if ({self.expr!s})" + " {\n" + str(self.condition_body) + "}"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions: list[Instruction | BranchMark | MarkedInstruction] = list()
        instructions += self.expr.translate(translator)
        branch_mark = BranchMark()
        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE), branch_mark))
        instructions += self.condition_body.translate(translator)
        instructions.append(branch_mark)
        return instructions


class AstWhile(AstExpr):
    def __init__(self, expr: AstExpr, conditional_body: AstBlockBody) -> None:
        self.expr = expr
        self.condition_body = conditional_body

    def __str__(self) -> str:
        return f"while ({self.expr!s})" + " {\n" + str(self.condition_body) + "}"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions: list[Instruction | BranchMark | MarkedInstruction] = list()
        in_mark = BranchMark()
        out_mark = BranchMark()
        instructions.append(in_mark)
        instructions += self.expr.translate(translator)
        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE), out_mark))
        instructions += self.condition_body.translate(translator)
        instructions.append(MarkedInstruction(Instruction(Opcode.JMP, None, ArgType.IMMEDIATE), in_mark))
        instructions.append(out_mark)
        return instructions


# One argument proto
class AstFuncProto(AstExpr):
    def __init__(self, argument: AstVar, proto_body: AstBlockBody) -> None:
        self.argument = argument
        self.body = proto_body

    def __str__(self) -> str:
        return f"({self.argument.name}) " + "{\n" + str(self.body) + "}"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions: list[Instruction | BranchMark | MarkedInstruction] = list()
        ptr = translator.allocate_var(self.argument.name)
        instructions.append(Instruction(Opcode.IN, None, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.ST, ptr, ArgType.IMMEDIATE))
        instructions += self.body.translate(translator)
        return instructions


class AstInterrupt(AstExpr):
    def __init__(self, proto: AstFuncProto) -> None:
        self.proto = proto

    def __str__(self) -> str:
        return "interrupt" + str(self.proto)

    def translate(self, translator: Translator) -> list[Instruction | MarkedInstruction | BranchMark]:
        instructions: list[Instruction | BranchMark | MarkedInstruction] = list()
        skip_interrupt_mark = BranchMark()

        # to main program
        instructions.append(MarkedInstruction(Instruction(Opcode.JMP, None, ArgType.IMMEDIATE), skip_interrupt_mark))

        # interrupt itself - saving acc
        instructions.append(Instruction(Opcode.ILOCK, None, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.ST, 0x1FFFFFFF, ArgType.IMMEDIATE))
        instructions += self.proto.translate(translator)
        instructions.append(Instruction(Opcode.LD, 0x1FFFFFFF, ArgType.DIRECT))
        instructions.append(Instruction(Opcode.IRET, None, ArgType.IMMEDIATE))
        instructions.append(skip_interrupt_mark)

        return instructions


class AstEcho(AstExpr):
    def __init__(self, expr: AstEchoable) -> None:
        self.echo_expr = expr

    def __str__(self) -> str:
        return "echo(" + str(self.echo_expr) + ")"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions = list()
        if self.echo_expr.get_expr_type() is AstExprType.STRING:
            tmp_iterator_ptr = translator.allocate_for_tmp_expr()
            tmp_length = translator.allocate_for_tmp_expr()
            loop_mark = BranchMark()
            end_mark = BranchMark()

            instructions += self.echo_expr.translate(translator)
            instructions.append(Instruction(Opcode.ST, tmp_iterator_ptr, ArgType.IMMEDIATE))

            instructions.append(Instruction(Opcode.LD, tmp_iterator_ptr, ArgType.INDIRECT))
            instructions.append(Instruction(Opcode.ST, tmp_length, ArgType.IMMEDIATE))
            instructions.append(Instruction(Opcode.OUT, None, ArgType.IMMEDIATE))

            instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE), end_mark))

            instructions += inc(tmp_iterator_ptr, ArgType.DIRECT)

            instructions.append(loop_mark)
            instructions.append(Instruction(Opcode.LD, tmp_iterator_ptr, ArgType.INDIRECT))
            instructions.append(Instruction(Opcode.OUT, None, ArgType.IMMEDIATE))
            instructions += inc(tmp_iterator_ptr, ArgType.DIRECT)
            instructions += dec(tmp_length, ArgType.DIRECT)
            instructions.append(Instruction(Opcode.LD, tmp_length, ArgType.DIRECT))
            instructions.append(Instruction(Opcode.SETG, None, ArgType.IMMEDIATE))
            instructions.append(MarkedInstruction(Instruction(Opcode.JNZ, None, ArgType.IMMEDIATE), loop_mark))

            instructions.append(end_mark)
            translator.free_tmp_expr(tmp_iterator_ptr)
            translator.free_tmp_expr(tmp_length)
        else:
            instructions += self.echo_expr.translate(translator)
            instructions.append(Instruction(Opcode.OUT, None, ArgType.IMMEDIATE))
        return instructions


class AstStrCat(AstEchoable):
    # total_size is the result buffer size (with null-term included)
    def __init__(self, dest: AstExpr, src: AstExpr) -> None:
        if dest.get_expr_type() != AstExprType.STRING:
            # TODO Throw error here ( expected string, got smthng instead )
            return
        if isinstance(dest, AstLit):
            # TODO Throw error here ( dest must be a variable )
            return
        if src.get_expr_type() != AstExprType.STRING:
            return
        self.dest = dest
        self.src = src

    def __str__(self) -> str:
        return "strcat(" + str(self.dest) + ", " + str(self.src) + ")"

    def get_expr_type(self) -> AstExprType:
        return AstExprType.STRING

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions = list()

        instructions += self.dest.translate(translator)

        tmp_dest_ptr = translator.allocate_for_tmp_expr()
        tmp_append_ptr = translator.allocate_for_tmp_expr()
        tmp_size_left_ptr = translator.allocate_for_tmp_expr()

        instructions.append(Instruction(Opcode.ST, tmp_dest_ptr, ArgType.IMMEDIATE))

        instructions.append(Instruction(Opcode.ADD, 1, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.ST, tmp_append_ptr, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.LD, tmp_dest_ptr, ArgType.INDIRECT))
        instructions.append(Instruction(Opcode.ST, tmp_size_left_ptr, ArgType.IMMEDIATE))

        # seek for empty space in dest
        seek_mark = BranchMark()
        end_mark = BranchMark()
        no_empty_mark = BranchMark()
        instructions.append(seek_mark)

        instructions.append(Instruction(Opcode.SETG, None, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.SETG, None, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.SETG, None, ArgType.IMMEDIATE))

        instructions.append(Instruction(Opcode.LD, tmp_append_ptr, ArgType.INDIRECT))
        # empty spcae found
        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE), end_mark))
        instructions.append(Instruction(Opcode.LD, tmp_size_left_ptr, ArgType.DIRECT))
        # no empty space
        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE), no_empty_mark))
        instructions += inc(tmp_append_ptr, ArgType.DIRECT)
        instructions += dec(tmp_size_left_ptr, ArgType.DIRECT)
        instructions.append(MarkedInstruction(Instruction(Opcode.JMP, None, ArgType.IMMEDIATE), seek_mark))
        instructions.append(end_mark)

        # empty space found - copying till the end
        instructions += self.src.translate(translator)
        tmp_src = translator.allocate_for_tmp_expr()
        tmp_src_iterator = translator.allocate_for_tmp_expr()
        tmp_src_size_left_ptr = translator.allocate_for_tmp_expr()

        instructions.append(Instruction(Opcode.ST, tmp_src, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.ADD, 1, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.ST, tmp_src_iterator, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.LD, tmp_src, ArgType.INDIRECT))
        instructions.append(Instruction(Opcode.ST, tmp_src_size_left_ptr, ArgType.IMMEDIATE))

        copy_mark = BranchMark()
        end_copy_mark = BranchMark()

        instructions.append(copy_mark)

        instructions.append(Instruction(Opcode.SETE, None, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.SETE, None, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.SETE, None, ArgType.IMMEDIATE))

        instructions.append(Instruction(Opcode.LD, tmp_src_iterator, ArgType.INDIRECT))
        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE), end_copy_mark))
        # saved char to dest
        instructions.append(Instruction(Opcode.ST, tmp_append_ptr, ArgType.DIRECT))
        # no empty space left
        instructions += dec(tmp_size_left_ptr, ArgType.DIRECT)
        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE), no_empty_mark))
        # no symbols to copy
        instructions += dec(tmp_src_size_left_ptr, ArgType.DIRECT)
        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE), end_copy_mark))
        instructions += inc(tmp_append_ptr, ArgType.DIRECT)

        instructions.append(MarkedInstruction(Instruction(Opcode.JMP, None, ArgType.IMMEDIATE), copy_mark))

        instructions.append(end_copy_mark)
        instructions.append(no_empty_mark)
        instructions.append(Instruction(Opcode.LD, tmp_dest_ptr, ArgType.DIRECT))

        translator.free_tmp_expr(tmp_src)
        translator.free_tmp_expr(tmp_src_iterator)
        translator.free_tmp_expr(tmp_src_size_left_ptr)
        translator.free_tmp_expr(tmp_append_ptr)
        translator.free_tmp_expr(tmp_dest_ptr)
        translator.free_tmp_expr(tmp_size_left_ptr)

        return instructions

class AstToString(AstExpr):
    def __init__(self, expr: AstExpr) -> None:
        self.to_string = expr

    def __str__(self) -> str:
        return "to_string(" + str(self.to_string) + ")"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions = list()

        instructions += self.to_string.translate(translator)

        pascal_string_ptr = translator.allocate_memory(2)
        print(hex(pascal_string_ptr))
        instructions.append(Instruction(Opcode.ST, pascal_string_ptr + 1, ArgType.IMMEDIATE))
        instructions += init(pascal_string_ptr, ArgType.IMMEDIATE, 1)
        instructions.append(Instruction(Opcode.LD, pascal_string_ptr, ArgType.IMMEDIATE))

        return instructions

    def get_expr_type(self) -> AstExprType:
        return AstExprType.STRING


class AstCreateString(AstExpr):
    def __init__(self, expr: AstExpr, size: int) -> None:
        self.expr = expr
        self.size = size

    def __str__(self) -> str:
        return "create_string(" + str(self.expr) + ", " + str(self.size) + ")"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark | MarkedInstruction]:
        instructions = list()

        instructions += self.expr.translate(translator)
        pascal_str_ptr = translator.allocate_memory(self.size + 1)
        tmp_iterator = translator.allocate_for_tmp_expr()
        tmp_pascal_iterator = translator.allocate_for_tmp_expr()
        tmp_size = translator.allocate_for_tmp_expr()

        instructions.append(Instruction(Opcode.ST, tmp_iterator, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.LD, tmp_iterator, ArgType.INDIRECT))
        instructions.append(Instruction(Opcode.SUB, 1, ArgType.IMMEDIATE))
        instructions.append(Instruction(Opcode.ST, tmp_size, ArgType.IMMEDIATE))

        instructions += init(tmp_pascal_iterator, ArgType.IMMEDIATE, pascal_str_ptr)

        instructions += init(pascal_str_ptr, ArgType.IMMEDIATE, self.size)
        instructions += inc(tmp_iterator, ArgType.DIRECT)
        instructions += inc(tmp_pascal_iterator, ArgType.DIRECT)

        copying_mark = BranchMark()
        exit_mark = BranchMark()

        instructions.append(copying_mark)

        instructions.append(Instruction(Opcode.LD, tmp_iterator, ArgType.INDIRECT))
        instructions.append(Instruction(Opcode.ST, tmp_pascal_iterator, ArgType.DIRECT))

        instructions.append(Instruction(Opcode.LD, tmp_size, ArgType.DIRECT))
        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE), exit_mark))
        instructions.append(Instruction(Opcode.CMP, pascal_str_ptr, ArgType.DIRECT))
        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE),exit_mark))

        instructions += dec(tmp_size, ArgType.DIRECT)
        instructions += inc(tmp_pascal_iterator, ArgType.DIRECT)
        instructions += inc(tmp_iterator, ArgType.DIRECT)
        instructions.append(MarkedInstruction(Instruction(Opcode.JMP, None, ArgType.IMMEDIATE), copying_mark))

        instructions.append(exit_mark)

        instructions.append(Instruction(Opcode.LD, pascal_str_ptr, ArgType.IMMEDIATE))

        translator.free_tmp_expr(tmp_iterator)
        translator.free_tmp_expr(tmp_pascal_iterator)
        translator.free_tmp_expr(tmp_size)

        return instructions

    def get_expr_type(self) -> AstExprType:
        return AstExprType.STRING


# ===-------------------------------------------
# Branch Mark Resolver
# ===-------------------------------------------


def resolve_marks(instruction_list: list[Instruction | MarkedInstruction | BranchMark]):
    mark_cnt = 0
    for pos, i in enumerate(instruction_list):
        if isinstance(i, BranchMark):
            i.set_position(pos - mark_cnt)
            mark_cnt += 1

    res: list[Instruction] = []
    for r in filter(lambda x: not isinstance(x, BranchMark), instruction_list):
        if isinstance(r, MarkedInstruction):
            r.instruction.arg = r.mark.position
            res.append(r.instruction)
        if isinstance(r, Instruction):
            res.append(r)

    return res


# ===-------------------------------------------
# Ast Builder
# ===-------------------------------------------


class AstBuilder:
    def __init__(self, tokenizer: Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.current_token: Token = self.tokenizer.get_next_token()
        self.token2parser: dict[TokenType, Callable[[], AstExpr]] = {
            TokenType.L_PAR: lambda: self.parse_paren_lit()[0],
            TokenType.VAR_LIT: self.parse_identifier,
            TokenType.NUM_LIT: self.parse_number_lit,
            TokenType.STR_LIT: self.parse_string_lit,
            TokenType.STRCAT: self.parse_strcat,
            TokenType.TO_STRING: self.parse_tostring,
            TokenType.CREATE_STR: self.parse_createstring
        }
        self.handle2handler = {
            TokenType.VAR_LIT: self.handle_decl,
            TokenType.IF: self.handle_if,
            TokenType.WHILE: self.handle_while,
            TokenType.INTERRUPT: self.handle_interrupt,
            TokenType.ECHO: self.handle_echo,
            TokenType.STRCAT: self.handle_strcat,
            TokenType.TO_STRING: self.handle_tostring,
            TokenType.CREATE_STR: self.handle_createstring,
        }
        self.bin_op_rank = {
            TokenType.LT: 10,
            TokenType.GT: 10,
            TokenType.NOT: 10,
            TokenType.AND: 20,
            TokenType.OR: 10,
            TokenType.EQ: 10,
            TokenType.NEQ: 10,
            TokenType.PLUS: 20,
            TokenType.MINUS: 20,
            TokenType.MUL: 40,
            TokenType.DIV: 40,
            TokenType.MOD: 40,
        }

        self.symbol_table: dict[str, AstExprType] = {}

    # ===-------------------------------------------
    # Parsers
    # ===-------------------------------------------

    def parse_number_lit(self) -> AstLitNumber:
        expr = AstLitNumber(int(self.current_token.value))
        self.next_token()  # num -> ...
        return expr

    def parse_string_lit(self) -> AstLitString:
        expr = AstLitString(str(self.current_token.value))
        self.next_token()  # string -> ...
        return expr

    def parse_paren_lit(self) -> list[AstExpr]:
        self.next_token()  # L_PAR -> expr
        paren_expr = [self.parse_expr()]  # expr -> R_PAR | COMMA
        if paren_expr is None:
            raise AstError(self.tokenizer, "Unexpected empty paren expression")

        while self.current_token.token_type == TokenType.COMMA:
            self.next_token()
            paren_expr.append(self.parse_expr())

        if self.current_token.token_type != TokenType.R_PAR:
            raise AstExpectedTokenError(self.tokenizer, self.current_token.token_type, TokenType.R_PAR)
        self.next_token()  # R_PAR -> ...
        return paren_expr

    def parse_identifier(self, unique_id: str = "") -> AstVar:
        var_id = unique_id + self.current_token.value
        if var_id not in self.symbol_table:
            raise AstUnresolvedVarError(self.tokenizer, var_id)

        var = AstVar(self.current_token.value, self.symbol_table[var_id], unique_id)
        self.next_token()  # VAR_LIT -> ...
        return var

    def parse_primary(self) -> AstExpr:
        if self.current_token.token_type not in self.token2parser.keys():
            raise AstUnexpectedTokenError(self.tokenizer, self.current_token)
        token_parser = self.token2parser[self.current_token.token_type]
        return token_parser()

    def parse_binop_rhs(self, expr_rank: int, lhs: AstExpr) -> AstExpr:
        while True:
            current_token_rank = self.get_current_token_rank()
            if current_token_rank <= expr_rank:
                return lhs
            token = self.current_token
            self.next_token()

            rhs = self.parse_primary()
            if rhs is None:
                raise AstError(self.tokenizer, "Failed while parsing binary operator (rhs expected)")

            next_token_rank = self.get_current_token_rank()
            if current_token_rank < next_token_rank:
                rhs = self.parse_binop_rhs(current_token_rank + 1, lhs)
                if rhs is None:
                    raise AstError(self.tokenizer, "Failed while parsing binary operator (rhs expected)")

            if lhs.get_expr_type() != rhs.get_expr_type():
                raise AstTypeMismatchError(self.tokenizer, lhs.get_expr_type(), rhs.get_expr_type())

            lhs = AstBinary(token, lhs, rhs)

    def parse_expr(self) -> AstExpr:
        lhs = self.parse_primary()
        if lhs is None:
            raise AstExpectedTokenError(self.tokenizer, "None", "Expression")
        return self.parse_binop_rhs(0, lhs)

    def parse_decl(self) -> AstDecl:
        current_var_token = self.current_token
        self.next_token()  # VAR_LIT -> = (binop)

        if self.current_token.token_type != TokenType.ASSIGN:
            raise AstExpectedTokenError(self.tokenizer, self.current_token.token_type, TokenType.ASSIGN)

        self.next_token()  # ASSIGN -> expr
        expr = self.parse_expr()
        decl = AstDecl(current_var_token.value, expr)

        self.symbol_table[decl.var.name] = decl.get_expr_type()  # append the symbol table with this (var | type)

        return decl

    def parse_block(self) -> AstBlockBody:
        instruction_list = []
        self.next_token()  # L_BRACE -> ...
        while self.current_token.token_type != TokenType.R_BRACE:
            instruction = self.handle_block_body()
            instruction_list.append(instruction)
        self.next_token()  # } -> ...

        return AstBlockBody(instruction_list)

    def parse_if(self) -> AstIf:
        self.next_token()  # IF -> ( | expr)
        cond_expr = self.parse_paren_lit()[0]
        if cond_expr.get_expr_type() is not AstExprType.NUM:
            raise AstExpectedConditionalExpressionError(self.tokenizer, cond_expr.get_expr_type())

        block = self.parse_block()
        return AstIf(cond_expr, block)

    def parse_while(self) -> AstWhile:
        self.next_token()  # WHILE -> ( | expr)
        cond_expr = self.parse_paren_lit()[0]
        if cond_expr.get_expr_type() is not AstExprType.NUM:
            raise AstExpectedConditionalExpressionError(self.tokenizer, cond_expr.get_expr_type())

        block = self.parse_block()
        return AstWhile(cond_expr, block)

    def parse_proto(self, seed: str) -> AstFuncProto:
        if self.current_token.token_type != TokenType.L_PAR:
            raise AstExpectedTokenError(self.tokenizer, self.current_token.token_type, TokenType.L_PAR)

        self.next_token()
        self.symbol_table[self.current_token.value] = AstExprType.CHAR
        var = self.parse_identifier()
        if not isinstance(var, AstVar):
            raise AstExpectedArgumentError(self.tokenizer, var)

        if self.current_token.token_type != TokenType.R_PAR:
            raise AstExpectedTokenError(self.tokenizer, self.current_token.token_type, TokenType.R_PAR)
        self.next_token()

        proto_body = self.parse_block()
        return AstFuncProto(var, proto_body)

    def parse_interrupt(self) -> AstInterrupt:
        self.next_token()
        proto = self.parse_proto("in")
        if proto is None:
            raise AstCreatingPrototypeError(self.tokenizer)
        return AstInterrupt(proto)

    def parse_echo(self) -> AstEcho:
        self.next_token()  # ECHO -> ( | expr)
        var = self.parse_paren_lit()[0]
        if not isinstance(var, AstEchoable):
            raise AstExpectedTokenError(self.tokenizer, var.__class__.__name__, AstEchoable.__class__.__name__)

        return AstEcho(var)

    def parse_strcat(self) -> AstStrCat:
        self.next_token()
        dest, src = self.parse_paren_lit()
        if not isinstance(dest, AstVar) and dest.get_expr_type() is not AstExprType.STRING:
            raise AstExpectedTokenError(self.tokenizer, dest.get_expr_type(), src.get_expr_type())
        return AstStrCat(dest, src)

    def parse_tostring(self):
        self.next_token() # TO_STRING -> (
        expr = self.parse_paren_lit()[0]
        if expr.get_expr_type() is not AstExprType.CHAR:
            raise AstExpectedTokenError(self.tokenizer, expected=AstExprType.CHAR, actual=expr.get_expr_type())
        return AstToString(expr)

    def parse_createstring(self):
        self.next_token() # CREATE_STRING -> (
        expr, size = self.parse_paren_lit()
        if expr.get_expr_type() is not AstExprType.STRING:
            raise AstExpectedTokenError(self.tokenizer, expr.get_expr_type(), AstExprType.STRING)

        if size.get_expr_type() is not AstExprType.NUM:
            raise AstExpectedTokenError(self.tokenizer, size.get_expr_type(), AstExprType.NUM)

        if not isinstance(size, AstLitNumber):
            raise AstExpectedTokenError(self.tokenizer, size.get_expr_type(), AstLitNumber.__class__.__name__)

        return AstCreateString(expr, size.value)


    # ===-------------------------------------------
    # Handlers
    # ===-------------------------------------------

    def handle(self) -> AstExpr | None:
        if self.current_token.token_type == TokenType.EOF:
            return None
        if self.current_token.token_type in self.handle2handler.keys():
            return self.handle2handler[self.current_token.token_type]()
        raise AstUnexpectedTokenError(self.tokenizer, self.current_token.token_type)

    def handle_block_body(self) -> AstExpr:
        if self.current_token.token_type in self.handle2handler.keys():
            return self.handle2handler[self.current_token.token_type]()
        raise AstUnexpectedTokenError(self.tokenizer, self.current_token.token_type)

    def handle_decl(self) -> AstDecl:
        return self.parse_decl()

    def handle_if(self) -> AstIf:
        return self.parse_if()

    def handle_while(self) -> AstWhile:
        return self.parse_while()

    def handle_interrupt(self) -> AstInterrupt:
        # TODO only one interrupt handler can be in code
        return self.parse_interrupt()

    def handle_echo(self) -> AstEcho:
        return self.parse_echo()

    def handle_strcat(self) -> AstStrCat:
        return self.parse_strcat()

    def handle_tostring(self) -> AstToString:
        return self.parse_tostring()

    def handle_createstring(self) -> AstCreateString:
        return self.handle_createstring()

    # ===-------------------------------------------
    # Builder
    # ===-------------------------------------------

    def get_current_token_rank(self) -> int:
        if self.current_token.token_type in self.bin_op_rank.keys():
            return self.bin_op_rank[self.current_token.token_type]
        return 0

    def next_token(self) -> None:
        self.current_token = self.tokenizer.get_next_token()

    def build(self):
        instructions = list()

        instruction = self.handle()
        while instruction is not None:
            instructions.append(instruction)
            instruction = self.handle()

        return instructions
