"""
compiler.py: Compilador para nuestro programa simple de juguete.

No estoy muy orgulloso de este programa porque pudo ser mejor y mas leible,
pero no tuve tiempo por el trabajo y las otras materias, y se me hizo poquito
tiempo una semana :(.
"""
__author__ = "Jorge Vinicio Quintero Santos. A01634502"

import graphviz
import argparse
import os.path
from enum import Enum

RESERVED_SYMBOLS = ['p', 'i', 'f', '=', '+', '-', '.']


class CompilerException(Exception):
    """Declares that the compiler has encountered an error."""
    def __init__(self, message, lineStr, lineCounter) -> None:
        self.message = message
        self.lineStr = lineStr
        self.lineCounter = lineCounter


class TokenType(Enum):
    """Types of valid tokens in the program."""
    FLOAT = 0
    INTEGER = 1
    PRINT = 2
    ASSIGN = 3
    PLUS = 4
    MINUS = 5
    ID = 6
    FNUMBER = 7
    INUMBER = 8
    INT2FLOAT = 9


class Token:
    """Defines a single token in the program."""
    def __init__(self, value: str, type: TokenType, lineNumber: str):
        self.value = value
        self.type = type
        self.lineNumber = lineNumber

    def __str__(self) -> str:
        return f"{self.value} {self.type}"


class TreeNode:
    """Represents a node in the AST tree."""
    # Conversion is not a great practice but it works for now for the nodes that need it.
    def __init__(self, token: Token, childs: list = [], conversion: TokenType = None):
        self.token = token
        self.childs = childs
        self.conversion = conversion

    def __str__(self) -> str:
        return f"{self.token.value} {self.token.type} {self.conversion}"


class Compiler:
    """The simple program compiler for our toy language."""
    _availableSymbols = [chr(97 + i) for i in range(26) if chr(97 + i)
                         not in RESERVED_SYMBOLS]  # [a-z] - RESERVED_SYMBOLS
    _numbers = [str(i) for i in range(10)] # [0-9]
    _arithmetic = [TokenType.MINUS, TokenType.PLUS] # Valid arithmetic operations.


    def __init__(self) -> None:
        self.declarationTokens: list[TokenType] = []
        self.tokens: list[list[TokenType]] = []
        self.symbolTable = {}
        self.currentLineIndex = 1
        self.currentLineStr = ''
        self.astroot: TreeNode = None
        self.gen = self._nodeGenerator()
        self.tmpGen = self._tempGenerator()
        self.assigned: list[str] = []

    def processFile(self, file_path: str, verbose=False):
        """Process a file and outputs the three address code.
        
        The output of the code is stored in a file called 'output.ez' where the program was runned.
        The AST created during parsing and modified during type checking will be exported in the
        folder /graph where the program was runned.

        If verbose is True, it will shows some important parts of the process on the stdout.
        """
        self._lexicalAnalysis(file_path)
        self._parseTokens()
        self._typeChecking()
        lines = self._threeAddressCode()
        self._saveTAC(lines)

        self.dot = graphviz.Digraph(comment='Final AST')
        self._buildAstGraph(self.astroot, next(self.gen))
        self.dot.format = 'png'
        self.dot.render('graph/output_graph')
        if verbose:
            print("============== LEXICAL ANALYSIS ==============")
            print("==> Declarations")
            for v in self.declarationTokens:
                print(str(v))
            print("==> Assignments")
            for v in self.tokens:
                for l in v:
                    print(str(l))
            print("==> Symbol Table")
            print(self.symbolTable)
            print("============== PARSER AND TYPE CHECKING ==============")
            print("========== RESULT AST")
            self._printastConsole(self.astroot, 1)

    def _validateNumber(self, num: str):
        """Validates if a string is a number.
        
        Returns a position of the error (>= 0) if it's not a number, else -1.
        """
        # We can do some other simpler stuff but just to simulate we are
        # in a low level language with no libraries we will do something rustic.
        dotfound = False
        for i, v in enumerate(num):
            if v in self._numbers:
                continue
            if v == '.':
                if not dotfound:
                    dotfound = True
                    continue
                else:
                    return i
            return i
        return -1

    def _generateDeclarationTokens(self, line: str):
        """ Process the first lines of the program where all declarations are.
        
        It creates the appropiate tokens and assign them the TokenType FLOAT or INTEGER.
        """
        # We assume that every token is separated by a white space and
        # declarations are first.
        lineTokens = line.split()
        t = lineTokens[0]
        tType = TokenType.FLOAT if t == 'f' else TokenType.INTEGER
        for c in range(1, len(lineTokens)):
            symbol = lineTokens[c]
            if symbol not in self._availableSymbols:
                raise CompilerException(
                    f'Symbol "{symbol}" is not a valid id.', self.currentLineStr, self.currentLineIndex)
            self.declarationTokens.append(
                Token(symbol, tType, self.currentLineIndex))
            if symbol not in self.symbolTable:
                self.symbolTable[symbol] = tType
            else:
                raise CompilerException(f"Variable \"{symbol}\"was already declared", self.currentLineStr, self.currentLineIndex)

    def _generateTokens(self, line: str):
        """Process all the tokens that are part of assignment and print below the declarations."""
        lineTokens = line.split()
        processedTokens = []
        for c in lineTokens:
            if c == " ":
                continue
            elif c[0] in self._numbers:
                invalid = self._validateNumber(c)
                if invalid != -1:
                    raise CompilerException(
                        f"Lexical error. '{c}' is not a valid number, problem found at char '{c[invalid]}' at position {invalid}",
                        self.currentLineStr,
                        self.currentLineIndex)
                if "." in c:
                    processedTokens.append(
                        Token(c, TokenType.FNUMBER, self.currentLineIndex))
                else:
                    processedTokens.append(
                        Token(c, TokenType.INUMBER, self.currentLineIndex))
            elif c == "p":
                processedTokens.append(
                    Token(c, TokenType.PRINT, self.currentLineIndex))
            elif c == "=":
                processedTokens.append(
                    Token(c, TokenType.ASSIGN, self.currentLineIndex))
            elif c == "+":
                processedTokens.append(
                    Token(c, TokenType.PLUS, self.currentLineIndex))
            elif c == "-":
                processedTokens.append(
                    Token(c, TokenType.MINUS, self.currentLineIndex))
            elif c in self._availableSymbols:
                processedTokens.append(
                    Token(c, TokenType.ID, self.currentLineIndex))
            elif c in ["f", "i"]:
                raise CompilerException(
                    "Lexical Error. Declarations must be at the start of the file.", self.currentLineStr, self.currentLineIndex)
            else:
                raise CompilerException(
                    f"Lexical Error. '{c}' is not a valid symbol.", self.currentLineStr, self.currentLineIndex)
        if (len(processedTokens) != 0):
            self.tokens.append(processedTokens)

    def _lexicalAnalysis(self, file_path: str):
        """ Function in charge of lexical analysis of the language.

        It receives the file path and process it. It first process declarations and then the rest.
        If a declaration is encountered after assignments and prints start appearing, it will return
        an error indicating that declarations must be first in the file.
        """
        self.programLines: list(str) = []
        with open(file_path, "r") as f:
            declarations = True
            lineCounter = 1
            while line := f.readline():  # EOF
                line = line.strip("\n")
                if line == "":
                    continue
                self.programLines.append(line)
                self.currentLineStr = line
                self.currentLineIndex = lineCounter
                if not line.split()[0] in ['f', 'i']:
                    declarations = False
                if declarations:
                    self._generateDeclarationTokens(line)
                else:
                    self._generateTokens(line)
                lineCounter += 1

    def _createCompilerError(self, message, token):
        """Creates a new exception based on a token."""
        return CompilerException(message, self.programLines[token.lineNumber-1], token.lineNumber)

    def _checkTokenExists(self, t: Token):
        """Verifies if a token exists on the symbol table."""
        if t.type == TokenType.ID:
            if not (t.value in self.symbolTable):
                raise self._createCompilerError(
                    f"Parse Error. '{t.value}' is not defined.", t)
    
    def _checkTokenHasValue(self, t: Token):
        if t.value not in self.assigned:
            raise self._createCompilerError(f"Parse Error. '{t.value}' has not been assigned a value.", t)


    def _parseStmts(self):
        """Parses the statements of the program."""
        validNums = [TokenType.INUMBER, TokenType.FNUMBER, TokenType.ID]
        operations = [TokenType.MINUS, TokenType.PLUS]
        for lineTokens in self.tokens:
            currentT = lineTokens[0]
            if (currentT.type == TokenType.ID):
                # Check if it exists on the symbol table
                if currentT.value not in self.symbolTable:
                    raise self._createCompilerError(f"Parse Error. ID '{currentT.value}' is not declared.", currentT)
                # An assignment must be at least 3 tokens in size.
                if not (len(lineTokens) >= 3):
                    raise self._createCompilerError(
                        "Parse Error. Not a valid expression.", currentT)
                if lineTokens[1].type != TokenType.ASSIGN:
                    raise self._createCompilerError(
                        "Parse Error. Equal must be present in assignment must be present.", lineTokens[1])
                # Add new base assign node.
                node = TreeNode(lineTokens[1], [TreeNode(currentT, [])])
                currentNode = node

                previousT = lineTokens[2]
                if not (previousT.type in validNums):
                    raise self._createCompilerError(
                        f"Parse Error. Invalid token '{previousT.value}'.", previousT)
                for index in range(3, len(lineTokens)):
                    if previousT.type == TokenType.ID:
                        self._checkTokenHasValue(previousT)
                    t = lineTokens[index]
                    if t.type in operations and previousT.type in validNums:
                        newChild = TreeNode(t, [TreeNode(previousT, [])])
                        currentNode.childs.append(newChild)
                        # Update the current node.
                        currentNode = newChild
                        previousT = t
                    elif t.type in validNums and previousT.type in operations:
                        if t.type == TokenType.ID:
                            self._checkTokenExists(t)
                        previousT = t
                    else:
                        raise self._createCompilerError(
                            f"Parse Error. Invalid token '{t.value}'.", t)
                # Process the last token
                if (previousT.type in validNums):
                    if previousT.type == TokenType.ID:
                        self._checkTokenExists(previousT)
                        self._checkTokenHasValue(previousT)
                    currentNode.childs.append(TreeNode(previousT, []))
                else:
                    raise self._createCompilerError(
                        f"Parse Error. Invalid token '{t.value}'.", t)
                self.assigned.append(currentT.value)
                self.astroot.childs.append(node)
            elif (currentT.type == TokenType.PRINT):
                if (len(lineTokens) < 2):
                    raise self._createCompilerError(
                        "Parse Error. Invalid print statement. Print must be followed by one id or number.", currentT)
                token = lineTokens[1]
                if (len(lineTokens) > 2 or token.type not in [TokenType.ID, TokenType.INUMBER, TokenType.FNUMBER]):
                    raise self._createCompilerError(
                        "Parse Error. Print statement can only be followed by one id or number.", lineTokens[0])
                if token.type == TokenType.ID:
                    if token.value not in self.symbolTable:
                        raise self._createCompilerError(f"Parse Error. ID '{token.value}' is not declared.", currentT)
                    self._checkTokenHasValue(token)
                self.astroot.childs.append(
                    TreeNode(Token(token.value, TokenType.PRINT, token.lineNumber), []))
            else:
                raise self._createCompilerError(
                    "Parse Error. Invalid statement encountered.", currentT)

    def _parseTokens(self):
        """Parser of the compiler.
        
        It creates an AST of the program.
        """
        self.astroot = TreeNode(Token('Program', None, -1), [])
        # Process declarations first.
        for tdcl in self.declarationTokens:
            self.astroot.childs.append(TreeNode(tdcl, []))
        self._parseStmts()

    def _getConversionToken(self, child):
        """ Creates a new TreeNode that represents the conversion of an integer to float."""
        return TreeNode(Token("int2float", TokenType.INT2FLOAT, child.token.lineNumber), [child])

    def _typeNodeHelper(self, current: TreeNode) -> TreeNode:
        """Recursive helper function for the _typeNode"""
        validFloats = [TokenType.FLOAT, TokenType.FNUMBER]
        rightNode = current.childs[1]
        if rightNode.token.type not in self._arithmetic:
            leftToken = current.childs[0].token
            leftType = self.symbolTable[leftToken.value] if leftToken.type == TokenType.ID else leftToken.type
            rightToken = rightNode.token
            rightType = self.symbolTable[rightToken.value] if rightToken.type == TokenType.ID else rightToken.type
            if leftType in validFloats or rightType in validFloats:
                current.conversion = TokenType.FLOAT
                if leftType not in validFloats:
                    current.childs[0] = self._getConversionToken(
                        current.childs[0])
                if rightType not in validFloats:
                    current.childs[1] = self._getConversionToken(
                        current.childs[1])
            return current
        processedRightNode = self._typeNodeHelper(current.childs[1])
        leftToken = current.childs[0].token
        leftType = self.symbolTable[leftToken.value] if leftToken.type == TokenType.ID else leftToken.type
        rightType = processedRightNode.conversion
        if leftType in validFloats or rightType in validFloats:
            current.conversion = TokenType.FLOAT
            if leftType not in validFloats:
                current.childs[0] = self._getConversionToken(current.childs[0])
            if rightType not in validFloats:
                current.childs[1] = self._getConversionToken(
                    processedRightNode)
        return current

    def _typeNode(self, node: TreeNode) -> None:
        """Modifies the AST and adds nodes where conversion is needed, also assigns types."""
        # Check if the right side is an operation.
        idType = self.symbolTable[node.childs[0].token.value]
        rightNodeToken = node.childs[1].token
        if rightNodeToken.type in [TokenType.MINUS, TokenType.PLUS]:
            newNode = self._typeNodeHelper(node.childs[1])
            if idType == TokenType.INTEGER and newNode.conversion == TokenType.FLOAT:
                raise self._createCompilerError(
                    f"Type Checking Error. Invalid explicit conversion from float to int on assignment.", rightNodeToken)
            node.childs[1] = newNode
            node.conversion = newNode.conversion
        else:
            # Is a one assignment operation.
            if idType == TokenType.INTEGER and rightNodeToken.type == TokenType.FNUMBER:
                raise self._createCompilerError(
                    f"Type Checking Error. Invalid explicit conversion of '{rightNodeToken.value}' from float to int.", rightNodeToken)
            if idType == TokenType.FLOAT and rightNodeToken.type in [TokenType.INTEGER, TokenType.INUMBER]:
                node.childs[1] = self._getConversionToken(node.childs[1])
            node.conversion = idType

    def _typeChecking(self):
        """Process all the tokens for type checking."""
        for child in self.astroot.childs:
            if child.token.type == TokenType.ASSIGN:
                self._typeNode(child)

    def _nodeToCode(self, token: Token) -> str:
        """Converts an prints, integer and float declaration to a code representation."""
        if token.type == TokenType.INTEGER:
            return f"DeclareInt {token.value}"
        elif token.type == TokenType.FLOAT:
            return f"DeclareFloat {token.value}"
        elif token.type == TokenType.PRINT:
            return f"Print {token.value}"
        return

    def _tempGenerator(self) -> str:
        """Generates temporal variable names."""
        counter = 0
        while True:
            yield f"t{counter}"
            counter += 1

    def _threeAddressConversion(self, node, lines):
        """Converts a int2float TreeNode to code representation."""
        if node.token.type == TokenType.INT2FLOAT:
            tmpVar = next(self.tmpGen)
            lines.append(f"{tmpVar} = int2float({node.childs[0].token.value})")
            return tmpVar
        return node.token.value

    def _threeAddressCodeHelper(self, node: TreeNode, lines: list) -> str:
        """Helper recursive function for _threeAddressCode.
        
        Return can be ignored, it is some weird recursion that uses it on the
        same function, this is not great for readability, but I didn't realize
        that in python you can create inner functions, so... rip, I am not
        redoing it, busy with work zzz.
        """
        if len(node.childs) == 0:
            if (node.token.type in [TokenType.INTEGER, TokenType.FLOAT, TokenType.PRINT]):
                lines.append(self._nodeToCode(node.token))
            return
        if node.token.type == TokenType.ASSIGN:
            right = node.childs[1]
            if right.token.type in self._arithmetic:
                newVar = self._threeAddressCodeHelper(right, lines)
                lines.append(f"{node.childs[0].token.value} = {newVar}")
                return
            lines.append(
                f"{node.childs[0].token.value} = {node.childs[1].token.value}")
            return
        # It's an operator
        leftNode = node.childs[0]
        rightNode = node.childs[1]
        if rightNode.token.type in self._arithmetic:
            rightVar = self._threeAddressCodeHelper(rightNode, lines)
        else:
            if (rightNode.token.type == TokenType.INT2FLOAT
                    and rightNode.childs[0].token.type in self._arithmetic):
                returnedVar = self._threeAddressCodeHelper(
                    rightNode.childs[0], lines)
                rightVar = next(self.tmpGen)
                lines.append(f"{rightVar} = int2float({returnedVar})")
            else:
                rightVar = self._threeAddressConversion(rightNode, lines)
        leftVar = self._threeAddressConversion(leftNode, lines)
        opVar = next(self.tmpGen)
        lines.append(f"{opVar} = {leftVar} {node.token.value} {rightVar}")
        return opVar

    def _threeAddressCode(self):
        """Generates the lines for the three address code output."""
        lines = []
        for child in self.astroot.childs:
            self._threeAddressCodeHelper(child, lines)
        return lines

    def _printastConsole(self, root: TreeNode, depth: int):
        """Prints the generated AST into the stdout."""
        if len(root.childs) == 0:
            print('--' * depth + "> " +
                  (root.token.value if root.token.type != TokenType.PRINT else "print " + root.token.value))
            return
        else:
            print('--' * depth + "> " + root.token.value +
                  (str(root.conversion) if root.conversion else ""))
            for c in root.childs:
                self._printastConsole(c, depth+1)

    def _nodeGenerator(self):
        """ Generator that follows the following cycle in alphabetical order:
        - [A-Z]
        - [A-Z][A-Z]
        - [A-Z][A-Z][A-Z]
        - ....
        """
        stored = ''
        counter = 0
        cycle = 0
        while True:
            current = stored + chr(65 + counter)
            counter += 1
            yield current
            if (counter == 26):
                if cycle == 26:
                    cycle = 0
                counter = 0
                stored += chr(65 + cycle)
                cycle += 1

    def _treeNodeToString(self, node: TreeNode) -> str:
        """Assigns a string value to every node in the AST for display purposes.
        
        Note: I think I should have mapped this at the start... but this made it look cool.
        """
        token = node.token
        if token.type == None:
            return "Program Start"
        elif token.type == TokenType.FLOAT:
            return f"floatdcl\n{token.value}"
        elif token.type == TokenType.INTEGER:
            return f"intdcl\n{token.value}"
        elif token.type == TokenType.ASSIGN:
            if node.conversion == TokenType.FLOAT:
                return f"ASSIGN\nfloat"
            return f"assign\ninteger"
        elif token.type == TokenType.ID:
            return f"id\n{token.value}"
        elif token.type == TokenType.INUMBER:
            return f"inum\n{token.value}"
        elif token.type == TokenType.FNUMBER:
            return f"fnum\n{token.value}"
        elif token.type == TokenType.PLUS:
            if node.conversion == TokenType.FLOAT:
                return f"plus\nfloat"
            return f"plus\ninteger"
        elif token.type == TokenType.MINUS:
            if node.conversion == TokenType.FLOAT:
                return f"minus\nfloat"
            return f"minus\ninteger"
        elif token.type == TokenType.PRINT:
            return f"print\n{node.token.value}"
        elif token.type == TokenType.INT2FLOAT:
            return f"int2float\nfloat"
        else:
            return "unknown :("

    def _buildAstGraph(self, root: TreeNode, rootname: str) -> TreeNode:
        """Builds the AST graph for graphviz."""
        if len(root.childs) == 0:
            return root
        else:
            self.dot.node(rootname, self._treeNodeToString(root))
            for c in root.childs:
                childname = next(self.gen)
                node = self._buildAstGraph(c, childname)
                self.dot.node(childname, self._treeNodeToString(node))
                self.dot.edge(rootname, childname)
            return root

    def _saveTAC(self, lines):
        """Saves the Three Address Code in the 'output.ez' file."""
        with open("output.ez", "w") as f:
            for line in lines:
                f.write(f"{line}\n")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "file_path", help="Location of the file to compile, relative to current location.")
    parser.add_argument(
        "-v", "--verbose", help="Add output prints to show results.", action="store_true")
    args = parser.parse_args()

    if not os.path.isfile(args.file_path):
        raise Exception(f"File provided: \"{args.file_path}\" does not exist.")

    try:
        c = Compiler()
        if args.verbose:
            c.processFile(args.file_path, verbose=True)
        else:
            c.processFile(args.file_path)
    except CompilerException as ce:
        print(
            f"Error found at line {ce.lineCounter}:\n\t{ce.lineStr}\n{ce.message}")
