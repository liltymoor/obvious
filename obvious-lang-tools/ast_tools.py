from enum import Enum
from io import TextIOBase

from tokenizer import Tokenizer, Token, TokenType
from translator import Translator, TranslateError
from isa import Opcode, Instruction, MarkedInstruction, ArgType, BranchMark

# ===-------------------------------------------
# Ast Errors
# ===-------------------------------------------

class AstError(Exception):
    def __init__(self, tokenizer, msg):
        self.tokenizer = tokenizer
        super().__init__(f"{tokenizer.line} | error: {msg}")

class AstErrorTypeMismatch(AstError):
    def __init__(self, tokenizer , actual, expected):
        self.expected = expected
        self.actual = actual
        super().__init__(tokenizer, f"Expected {expected} but got {actual} | No valid binary operator for {expected} & {actual}")

class AstErrorUnresolvedVar(AstError):
    def __init__(self, tokenizer, var_name):
        self.var_name = var_name
        super().__init__(tokenizer, f"Unresolved variable {self.var_name}")

class AstErrorUnexpectedToken(AstError):
    def __init__(self, tokenizer, actual):
        self.actual = actual
        super().__init__(tokenizer, f"No known parsers for {self.actual.token_type}")

class AstErrorExpectedToken(AstError):
    def __init__(self, tokenizer, actual, expected):
        self.expected = expected
        self.actual = actual
        super().__init__(tokenizer, f"Expected {expected} token but got {actual}")

class AstErrorExpectedConditionalExpression(AstError):
    def __init__(self, tokenizer, actual):
        self.actual = actual
        super().__init__(tokenizer, f"Expected conditional expression but got {actual}")

class AstErrorExpectedArgument(AstError):
    def __init__(self, tokenizer, actual):
        self.actual = actual
        super().__init__(tokenizer, f"Expected argument but got {actual}")

class AstErrorCreatingPrototype(AstError):
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
    @staticmethod
    def get_expr_type():
        return AstExprType.NONE

    def __str__(self):
        return "Empty expression"

    def translate(self, translator: Translator) -> list[Instruction]:
        raise NotImplementedError()

class AstEchoable(AstExpr):
    pass

class AstLit:
    pass

class AstLitNumber(AstEchoable, AstLit):
    def __init__(self, value: int) -> None:
        self.value = value

    @staticmethod
    def get_expr_type() -> AstExprType:
        return AstExprType.NUM

    def __str__(self) -> str:
        return f"Number Literal {self.value}"

    def translate(self, translator: Translator) -> list[Instruction]:
        return [Instruction(Opcode.LD, self.value, ArgType.IMMEDIATE)]

class AstLitString(AstEchoable, AstLit):
    def __init__(self, value: str) -> None:
        self.value = value

    @staticmethod
    def get_expr_type() -> AstExprType:
        return AstExprType.STRING

    @staticmethod
    def serialize(string: str) -> list[int]:
        return list(map(ord, string)) + [0]

    def translate(self, translator: Translator) -> list[Instruction]:
        instructions = list()
        serialized_string = self.serialize(self.value)
        instructions.append(Instruction(Opcode.LD, translator.allocate_data(len(serialized_string), serialized_string), ArgType.IMMEDIATE))
        return instructions

    def __str__(self) -> str:
        return f"String Literal \"{self.value}\""

class AstVar(AstEchoable):
    def __init__(self, name: str, var_type: AstExprType, unique_id: str = None) -> None:
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

    def translate(self, translator: Translator) -> list[Instruction]:
        instructions = list()
        instructions.append(Instruction(Opcode.LD, translator.get_var_addr(self.name), ArgType.DIRECT))
        return instructions

    def get_var_addr(self, translator: Translator) -> int:
        return translator.get_var_addr(self.name)

class AstBinary(AstExpr):
    opertator2opcode = {
        "+" : Opcode.ADD,
        "-" : Opcode.SUB,
        "*" : Opcode.MUL,
        "/" : Opcode.DIV,
        "%" : Opcode.MOD,
    }

    def __init__(self, op: Token, left: AstExpr, right: AstExpr) -> None:
        self.left = left
        self.right = right
        self.op = op

    def get_expr_type(self):
        return self.left.get_expr_type()

    def __str__(self) -> str:
        return f"{str(self.left)} {self.op.value} {str(self.right)}"


    def translate(self, translator: Translator) -> list[Instruction]:
        instructions = []

        if isinstance(self.left, AstVar):
            left = translator.get_var_addr(self.left.name)
            left_is_direct = True
        elif isinstance(self.left, AstLit) and isinstance(self.right, AstLit):
            left = self.left.value
            left_is_direct = False
        else:
            instructions += self.left.translate(translator)
            left = translator.allocate_for_tmp_expr()
            instructions.append(Instruction(Opcode.ST, left, ArgType.IMMEDIATE))
            left_is_direct = True


        instructions += self.right.translate(translator)

        if self.op.value in self.opertator2opcode.keys():
            instructions.append(
                Instruction(
                    self.opertator2opcode[self.op.value],
                    left,
                    ArgType.DIRECT if left_is_direct else ArgType.IMMEDIATE
                )
            )
        elif self.op.token_type == TokenType.GT:
            instructions.append(Instruction(Opcode.CMP, left, ArgType.DIRECT if left_is_direct else ArgType.IMMEDIATE))
            instructions.append(Instruction(Opcode.SETG))
        elif self.op.token_type == TokenType.LT:
            instructions.append(Instruction(Opcode.CMP, left, ArgType.DIRECT if left_is_direct else ArgType.IMMEDIATE))
            instructions.append(Instruction(Opcode.SETL))
        elif self.op.token_type == TokenType.EQ:
            instructions.append(Instruction(Opcode.CMP, left, ArgType.DIRECT if left_is_direct else ArgType.IMMEDIATE))
            instructions.append(Instruction(Opcode.SETE))
        elif self.op.token_type == TokenType.NEQ:
            instructions.append(Instruction(Opcode.CMP, left, ArgType.DIRECT if left_is_direct else ArgType.IMMEDIATE))
            instructions.append(Instruction(Opcode.SETE))
            instructions.append(Instruction(Opcode.NOT))
        else:
            raise AstErrorUnexpectedToken(self.op, self.op.token_type)

        return instructions


class AstDecl(AstExpr):
    def __init__(self, name: str, expr: AstExpr) -> None:
        self.var = AstVar(name, expr.get_expr_type())
        self.expr = expr

    def get_expr_type(self) -> AstExprType:
        return self.var.get_expr_type()

    def __str__(self) -> str:
        return f"{str(self.var.name)} = {str(self.expr)}"

    def translate(self, translator: Translator) -> list[Instruction]:
        instructions = list()
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

    def translate(self, translator: Translator) -> list[Instruction]:
        instructions = list()
        for ast in self.ast_list:
            instructions += ast.translate(translator)
        return instructions

class AstIf(AstExpr):
    def __init__(self, expr: AstExpr, conditional_body: AstBlockBody) -> None:
        self.expr = expr
        self.condition_body = conditional_body

    def __str__(self) -> str:
        return f"if ({str(self.expr)})" + " {\n" + str(self.condition_body) + "}"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark]:
        instructions = list()
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
        return f"while ({str(self.expr)})" + " {\n" + str(self.condition_body) + "}"

    def translate(self, translator: Translator) -> list[Instruction | BranchMark]:
        instructions = list()
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

    def translate(self, translator: Translator) -> list[Instruction]:
        instructions = list()
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

    def translate(self, translator: Translator) -> list[Instruction]:
        instructions = list()
        instructions += self.proto.translate(translator)
        instructions.append(Instruction(Opcode.IRET, None, ArgType.IMMEDIATE))
        return instructions

class AstEcho(AstExpr):
    def __init__(self, expr: AstEchoable) -> None:
        self.echo_expr = expr

    def __str__(self) -> str:
        return "echo(" + str(self.echo_expr) + ")"

    def translate(self, translator: Translator) -> list[Instruction]:
        instructions = list()
        #todo если Litstring реализовать другую логику
        instructions += self.echo_expr.translate(translator)
        instructions.append(Instruction(Opcode.OUT, None, ArgType.IMMEDIATE))
        return instructions

class AstStrCat(AstExpr):
    def __init__(self, dest: AstExpr, src: AstExpr) -> None:
        if dest.get_expr_type() != AstExprType.STRING:
            #TODO Throw error here ( expected string, got smthng instead )
            return None
        if isinstance(dest, AstLit):
            #TODO Throw error here ( dest must be a variable )
            return None
        if src.get_expr_type() != AstExprType.STRING:
            return None
        self.dest = dest
        self.src = src

    def __str__(self) -> str:
        return "strcat(" + str(self.dest) + ", " + str(self.src) + ")"

    def translate(self, translator: Translator) -> list[Instruction]:
        instructions = list()
        instructions += self.dest.translate(translator)

        tmp_iterator = translator.allocate_for_tmp_expr()
        instructions.append(Instruction(Opcode.ST, tmp_iterator, ArgType.IMMEDIATE))

        check_for_terminator_mark = BranchMark()
        go_to_copying_from_src_mark = BranchMark()

        instructions.append(check_for_terminator_mark)

        instructions.append(Instruction(Opcode.LD, tmp_iterator, ArgType.INDIRECT))
        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE), go_to_copying_from_src_mark))
        instructions.append(Instruction(Opcode.LD, tmp_iterator, ArgType.DIRECT))
        instructions.append(Instruction(Opcode.ADD, 1, ArgType.IMMEDIATE)) #TODO?? на единицу ли мы смещаемся или на 4 байта
        instructions.append(MarkedInstruction(Instruction(Opcode.JMP, None, ArgType.IMMEDIATE), check_for_terminator_mark))

        instructions.append(go_to_copying_from_src_mark)

        # получаем адрев дест идем до терминатора
        # как ток дошли начинаем записывать значения из src
        tmp_src_iterator = translator.allocate_for_tmp_expr()

        copying_from_src_mark = BranchMark()
        src_terminator_reached_mark = BranchMark()

        instructions += self.src.translate(translator)
        instructions.append(Instruction(Opcode.ST, tmp_src_iterator, ArgType.IMMEDIATE))

        instructions.append(copying_from_src_mark)

        instructions.append(Instruction(Opcode.LD, tmp_src_iterator, ArgType.INDIRECT))
        instructions.append(Instruction(Opcode.ST, tmp_iterator, ArgType.INDIRECT))

        instructions.append(Instruction(Opcode.LD, tmp_iterator, ArgType.DIRECT))
        instructions.append(Instruction(Opcode.ADD, 1, ArgType.IMMEDIATE))  # TODO?? на единицу ли мы смещаемся или на 4 байта

        instructions.append(MarkedInstruction(Instruction(Opcode.JZ, None, ArgType.IMMEDIATE ), src_terminator_reached_mark))
        instructions.append(MarkedInstruction(Instruction(Opcode.JMP, None, ArgType.IMMEDIATE), copying_from_src_mark))

        instructions.append(src_terminator_reached_mark)

        translator.free_tmp_expr(tmp_iterator)
        translator.free_tmp_expr(tmp_src_iterator)
        return instructions

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
    def __init__(self, tokenizer : Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.current_token: Token | None = None
        self.token2parser = {
            TokenType.L_PAR: lambda: self.parse_paren_lit()[0],
            TokenType.VAR_LIT: self.parse_identifier,
            TokenType.NUM_LIT: self.parse_number_lit,
            TokenType.STR_LIT: self.parse_string_lit,
            TokenType.STRCAT: self.parse_strcat
        }
        self.handle2handler = {
            TokenType.VAR_LIT: self.handle_decl,
            TokenType.IF: self.handle_if,
            TokenType.WHILE: self.handle_while,
            TokenType.INTERRUPT: self.handle_interrupt,
            TokenType.ECHO: self.handle_echo,
            TokenType.STRCAT: self.handle_strcat
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

        self.symbol_table = {}

    # ===-------------------------------------------
    # Parsers
    # ===-------------------------------------------

    def parse_number_lit(self) -> AstLitNumber:
        expr = AstLitNumber(int(self.current_token.value))
        self.next_token() # num -> ...
        print(f"parse_number_lit: {expr}")
        return expr

    def parse_string_lit(self) -> AstLitString:
        expr = AstLitString(str(self.current_token.value))
        self.next_token() # string -> ...
        print(f"parse_string_lit: {expr}")
        return expr

    def parse_paren_lit(self) -> list[AstExpr] | None:
        self.next_token() # L_PAR -> expr
        paren_expr = [self.parse_expr()] # expr -> R_PAR | COMMA
        if paren_expr is None:
            return None

        while self.current_token.token_type == TokenType.COMMA:
            self.next_token()
            paren_expr.append(self.parse_expr())

        if self.current_token.token_type != TokenType.R_PAR:
            raise AstErrorExpectedToken(self.tokenizer, self.current_token.token_type, TokenType.R_PAR)
        self.next_token() # R_PAR -> ...
        print(f"parse_paren_lit: {paren_expr}")
        return paren_expr

    def parse_identifier(self, unique_id: str = "") -> AstVar | None:
        var_id = unique_id + self.current_token.value
        if var_id not in self.symbol_table:
            raise AstErrorUnresolvedVar(self.tokenizer, var_id)

        var = AstVar(self.current_token.value, self.symbol_table[var_id], unique_id)
        self.next_token() # VAR_LIT -> ...
        print(f"parse_identifier: {var}: {var.get_expr_type()}")
        return var

    def parse_primary(self) -> AstExpr | None:
        if self.current_token.token_type not in self.token2parser.keys():
            raise AstErrorUnexpectedToken(self.tokenizer, self.current_token)
        token_parser = self.token2parser[self.current_token.token_type]
        expr = token_parser()
        print(f"parse_primary: {expr}")
        return expr

    def parse_binop_rhs(self, expr_rank : int, lhs : AstExpr) -> AstBinary | None:
        while True:
            current_token_rank = self.get_current_token_rank()
            if current_token_rank <= expr_rank:
                return lhs
            token = self.current_token
            self.next_token()

            rhs = self.parse_primary()
            if rhs is None:
                return None

            print(f"current_token: {self.current_token}")
            print(f"parse_binop_rhs: {lhs} {rhs}")

            next_token_rank = self.get_current_token_rank()
            if current_token_rank < next_token_rank:
                rhs = self.parse_binop_rhs(current_token_rank + 1, lhs)
                if rhs is None:
                    return None

            if lhs.get_expr_type() != rhs.get_expr_type():
                raise AstErrorTypeMismatch(self.tokenizer, lhs.get_expr_type(), rhs.get_expr_type())

            lhs = AstBinary(token, lhs, rhs)


    def parse_expr(self) -> AstExpr | None:
        lhs = self.parse_primary()
        print(f"parse_expr lhs: {lhs}")
        if lhs is None:
            raise AstErrorExpectedToken(self.tokenizer, "None", "Expression")
        return self.parse_binop_rhs(0, lhs)

    def parse_decl(self) -> AstDecl | None:
        current_var_token = self.current_token
        self.next_token() # VAR_LIT -> = (binop)

        if self.current_token.token_type != TokenType.ASSIGN:
            raise AstErrorExpectedToken(self.tokenizer, self.current_token.token_type, TokenType.ASSIGN)

        self.next_token() # ASSIGN -> expr
        expr = self.parse_expr()
        decl = AstDecl(current_var_token.value, expr)

        self.symbol_table[decl.var.name] = decl.get_expr_type() # append the symbol table with this (var | type)

        return decl

    def parse_block(self) -> AstBlockBody | None:
        instruction_list = []
        self.next_token() # L_BRACE -> ...
        while self.current_token.token_type != TokenType.R_BRACE:
            instruction = self.handle()
            instruction_list.append(instruction)
        self.next_token() # } -> ...

        return AstBlockBody(instruction_list)

    def parse_if(self) -> AstIf | None:
        self.next_token() # IF -> ( | expr)
        cond_expr = self.parse_paren_lit()[0]
        if cond_expr.get_expr_type() is not AstExprType.NUM:
            raise AstErrorExpectedConditionalExpression(self.tokenizer, cond_expr.get_expr_type())

        block = self.parse_block()
        return AstIf(cond_expr, block)

    def parse_while(self) -> AstExpr | None:
        self.next_token() # WHILE -> ( | expr)
        cond_expr = self.parse_paren_lit()[0]
        if cond_expr.get_expr_type() is not AstExprType.NUM:
            raise AstErrorExpectedConditionalExpression(self.tokenizer, cond_expr.get_expr_type())

        block = self.parse_block()
        return AstWhile(cond_expr, block)

    def parse_proto(self, seed: str) -> AstFuncProto | None:
        if self.current_token.token_type != TokenType.L_PAR:
            raise AstErrorExpectedToken(self.tokenizer, self.current_token.token_type, TokenType.L_PAR)

        self.next_token()
        self.symbol_table[self.current_token.value] = AstExprType.STRING #TODO String type ???
        var = self.parse_identifier()
        if not isinstance(var, AstVar):
            raise AstErrorExpectedArgument(self.tokenizer, var)

        if self.current_token.token_type != TokenType.R_PAR:
            raise AstErrorExpectedToken(self.tokenizer, self.current_token.token_type, TokenType.R_PAR)
        self.next_token()

        proto_body = self.parse_block()
        return AstFuncProto(var, proto_body)

    def parse_interrupt(self) -> AstInterrupt | None:
        self.next_token()
        proto = self.parse_proto("in")
        if proto is None:
            raise AstErrorCreatingPrototype(self.tokenizer)
        return AstInterrupt(proto)

    def parse_echo(self) -> AstEcho | None:
        self.next_token() # ECHO -> ( | expr)
        var = self.parse_paren_lit()[0]
        if not isinstance(var, AstEchoable):
            raise AstErrorExpectedToken(self.tokenizer, var.__class__.__name__, AstEchoable.__class__.__name__)

        return AstEcho(var)

    def parse_strcat(self) -> AstStrCat | None:
        self.next_token()
        dest, src = self.parse_paren_lit()
        if not isinstance(dest, AstVar) and dest.get_expr_type() is not AstExprType.STRING:
            #TODO Raise error here (destination must be a variable with type string)
            return None
        return AstStrCat(dest, src)

    #===-------------------------------------------
    # Handlers
    #===-------------------------------------------

    def handle(self) -> AstExpr | None:
        if self.current_token.token_type == TokenType.EOF:
            return None
        if self.current_token.token_type in self.handle2handler.keys():
            ast_node = self.handle2handler[self.current_token.token_type]()
            return ast_node
        else:
            raise AstErrorUnexpectedToken(self.tokenizer, self.current_token.token_type)

    def handle_decl(self) -> AstDecl | None:
        return self.parse_decl()

    def handle_if(self) -> AstIf | None:
        return self.parse_if()

    def handle_while(self) -> AstWhile | None:
        return self.parse_while()

    def handle_interrupt(self) -> AstInterrupt | None:
        #TODO only one interrupt handler can be in code
        return self.parse_interrupt()

    def handle_echo(self) -> AstEcho | None:
        return self.parse_echo()

    def handle_strcat(self) -> AstStrCat | None:
        return self.parse_strcat()

    #===-------------------------------------------
    # Builder
    #===-------------------------------------------

    def get_current_token_rank(self) -> int:
        if self.current_token.token_type in self.bin_op_rank.keys():
            return self.bin_op_rank[self.current_token.token_type]
        else:
            return 0

    def next_token(self) -> None:
        self.current_token = self.tokenizer.get_next_token()
        print(f"next token: {self.current_token}")

    def build(self):
        if self.current_token is None:
            self.next_token()

        instructions = list()

        instruction = self.handle()
        while instruction is not None:
            instructions.append(instruction)
            instruction = self.handle()

        return instructions

