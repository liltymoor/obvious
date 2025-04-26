from enum import Enum
from io import TextIOBase

from tokenizer import Tokenizer, Token, TokenType

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

class AstLitNumber(AstExpr):
    def __init__(self, value: int) -> None:
        self.value = value

    @staticmethod
    def get_expr_type() -> AstExprType:
        return AstExprType.NUM

    def __str__(self) -> str:
        return f"Number Literal {self.value}"

class AstVar(AstExpr):
    def __init__(self, name: str, var_type: AstExprType) -> None:
        self.name = name
        self.var_type = var_type

    def get_expr_type(self) -> AstExprType:
        return self.var_type

    def __str__(self) -> str:
        return f"Variable {self.name}: {self.var_type.name}"

class AstBinary(AstExpr):
    # TODO op ???
    def __init__(self, op: Token, left: AstExpr, right: AstExpr) -> None:
        if left.get_expr_type() != right.get_expr_type():
            #TODO Raise error here (opearnds type mismatch)
            print(f"Type mismatch {left.get_expr_type()} != {right.get_expr_type()}")
            return None

        self.left = left
        self.right = right
        self.op = op

    def get_expr_type(self):
        return self.left.get_expr_type()

    def __str__(self) -> str:
        return f"{str(self.left)} {self.op.value} {str(self.right)}"

class AstDecl(AstExpr):
    def __init__(self, name: str, expr: AstExpr) -> None:
        self.var = AstVar(name, expr.get_expr_type())
        self.expr = expr

    def get_expr_type(self) -> AstExprType:
        return self.var.get_expr_type()

    def __str__(self) -> str:
        return f"{str(self.var.name)} = {str(self.expr)}"

#TODO Insted of this make some hardocded functions like echo and so on

# class AstFuncProto(AstExpr):
#     def __init__(self, name: str, args: list[AstExpr]) -> None:
#         self.name = name
#         self.args = args
#
# class AstFunc(AstExpr):
#     def __init__(self, func_proto: AstFuncProto, func_body: AstExpr) -> None:
#         self.func_proto = func_proto
#         self.func_body = func_body

# ===-------------------------------------------
# Ast Builder
# ===-------------------------------------------

class AstBuilder:
    def __init__(self, tokenizer : Tokenizer) -> None:
        self.tokenizer = tokenizer
        self.current_token: Token | None = None
        self.token2parser = {
            TokenType.L_PAR: self.parse_paren_lit,
            TokenType.VAR_LIT: self.parse_identifier,
            TokenType.NUM_LIT: self.parse_number_lit,
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

    def parse_paren_lit(self) -> AstExpr | None:
        self.next_token() # L_PAR -> expr
        paren_expr = self.parse_expr() # expr -> R_PAR
        if paren_expr is None:
            return None
        if self.current_token.token_type != TokenType.R_PAR:
            #TODO return error here
            return None
        self.next_token() # R_PAR -> ...
        print(f"parse_paren_lit: {paren_expr}")
        return paren_expr

    def parse_identifier(self) -> AstVar | None:
        if self.current_token.value not in self.symbol_table:
            #TODO Raise error here (unresolved var)
            return None
        var = AstVar(self.current_token.value, self.symbol_table[self.current_token.value])
        self.next_token() # VAR_LIT -> ...
        print(f"parse_identifier: {var}: {var.get_expr_type()}")
        return var

    def parse_primary(self) -> AstExpr | None:
        if self.current_token.token_type not in self.token2parser.keys():
            #TODO Raise error here
            return None
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

            lhs = AstBinary(token, lhs, rhs)


    def parse_expr(self) -> AstExpr | None:
        lhs = self.parse_primary()
        print(f"parse_expr lhs: {lhs}")
        if lhs is None:
            #TODO Raise error here
            return None
        return self.parse_binop_rhs(0, lhs)

    def parse_decl(self) -> AstDecl | None:
        current_var_token = self.current_token
        self.next_token() # VAR_LIT -> = (binop)

        if self.current_token.token_type != TokenType.ASSIGN:
            #TODO Raise error here (assign expected)
            return None

        self.next_token() # ASSIGN -> expr
        expr = self.parse_expr()
        decl = AstDecl(current_var_token.value, expr)

        self.symbol_table[decl.var.name] = decl.get_expr_type() # append the symbol table with this (var | type)

        return decl

    #===-------------------------------------------
    # Handlers
    #===-------------------------------------------

    def handle_decl(self):
        self.parse_decl()

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
        self.handle_decl()

