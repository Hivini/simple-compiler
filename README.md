# Simple Compiler

## Requirements
```
sudo apt-get install graphviz
pip install graphviz
```

## Usage

```
python3 compiler.py input_file [-v]

Example:
python3 compiler.py test.simple -v
```
### Optional Flags
- **-v | --verbose** : Displays on stdout the tokens, symbol table and AST in text format.

## Language Capabilities
- Simple error detection.
- Declarations are at the start and can initialize multiple at a time. Example: **'i a b c'** will make three integer variables.
- Plus and minus algebraic operations.
- Three address code output.
- Output AST after semantic analysis with type checking at *graph/*, .png and digraph files.
