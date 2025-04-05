import ast
from typing import Dict

class PythonToJavaTranslator:
    """
    A Python to Java code translator that handles basic syntax conversion.
    This translator uses Python's AST to parse Python code and convert it to equivalent Java code.
    """

    def __init__(self):
        self.indent_level = 0
        self.indent_str = "    "  # 4 spaces for indentation
        self.current_class = None
        self.imports = set()
        self.variable_types: Dict[str, str] = {}

    def translate(self, python_code: str) -> str:
        """Main method to translate Python code to Java."""
        try:
            # Parse the Python code into an AST
            tree = ast.parse(python_code)

            # Perform initial analysis to gather type information
            self._analyze_types(tree)

            # Translate the AST to Java
            java_code = self._translate_module(tree)

            return java_code
        except SyntaxError as e:
            return f"// Error parsing Python code: {str(e)}"
        except Exception as e:
            return f"// Error during translation: {str(e)}"

    def _analyze_types(self, node: ast.AST) -> None:
        """Pre-process AST to gather type information."""
        # Visit all nodes to collect variable type annotations
        for child_node in ast.walk(node):
            # Handle type annotations in assignments
            if isinstance(child_node, ast.AnnAssign) and isinstance(child_node.target, ast.Name):
                var_name = child_node.target.id
                type_node = child_node.annotation

                if isinstance(type_node, ast.Name):
                    self.variable_types[var_name] = type_node.id
                elif isinstance(type_node, ast.Subscript) and isinstance(type_node.value, ast.Name):
                    base_type = type_node.value.id
                    if base_type == "List":
                        self.variable_types[var_name] = "ArrayList<Object>"
                    elif base_type == "Dict":
                        self.variable_types[var_name] = "HashMap<Object, Object>"

    def _translate_module(self, node: ast.Module) -> str:
        """Translate a Python module to Java."""
        result = []

        # Collect all class names first
        class_names = []
        for child in node.body:
            if isinstance(child, ast.ClassDef):
                class_names.append(child.name)

        # Use first class name as file name or "Main" if no classes
        main_class = class_names[0] if class_names else "Main"

        # Translate each statement
        for stmt in node.body:
            if isinstance(stmt, ast.Import) or isinstance(stmt, ast.ImportFrom):
                # Handle imports later
                self._process_import(stmt)
            elif isinstance(stmt, ast.ClassDef):
                result.append(self._translate_class(stmt))
            elif isinstance(stmt, ast.FunctionDef):
                # Top-level functions go into the Main class
                if not self.current_class:
                    if main_class not in class_names:
                        result.append(f"public class {main_class} {{")
                        self.indent_level += 1

                result.append(self._translate_function(stmt, is_method=False))

                if not self.current_class and main_class not in class_names:
                    self.indent_level -= 1
                    result.append("}")
            else:
                # Other top-level statements go into main method
                if not any(isinstance(s, ast.FunctionDef) and s.name == "main" for s in node.body):
                    if main_class not in class_names:
                        if not any(r.startswith("public class " + main_class) for r in result):
                            result.append(f"public class {main_class} {{")
                            self.indent_level += 1

                    result.append(self._indent() + "public static void main(String[] args) {")
                    self.indent_level += 1
                    result.append(self._translate_statement(stmt))
                    self.indent_level -= 1
                    result.append(self._indent() + "}")

                    if main_class not in class_names and not any(r.endswith("}") and not r.endswith("});") for r in result):
                        self.indent_level -= 1
                        result.append("}")
                else:
                    # Skip top-level statements if there's already a main method
                    pass

        # Add imports at the beginning
        imports = []
        for imp in sorted(self.imports):
            imports.append(f"import {imp};")

        if imports:
            imports.append("")  # Add a blank line

        return "\n".join(imports + result)

    def _translate_class(self, node: ast.ClassDef) -> str:
        """Translate a Python class to Java."""
        old_class = self.current_class
        self.current_class = node.name

        result = []
        class_line = f"public class {node.name}"

        # Handle inheritance
        if node.bases:
            base_names = []
            for base in node.bases:
                if isinstance(base, ast.Name):
                    base_names.append(base.id)

            if base_names:
                class_line += f" extends {base_names[0]}"

        result.append(class_line + " {")
        self.indent_level += 1

        # Add class fields and methods
        for item in node.body:
            if isinstance(item, ast.Assign):
                result.append(self._translate_class_field(item))
            elif isinstance(item, ast.AnnAssign):
                result.append(self._translate_annotated_class_field(item))
            elif isinstance(item, ast.FunctionDef):
                result.append(self._translate_function(item, is_method=True))

        self.indent_level -= 1
        result.append("}")

        self.current_class = old_class
        return "\n".join(result)

    def _translate_class_field(self, node: ast.Assign) -> str:
        """Translate a class field assignment."""
        if len(node.targets) != 1 or not isinstance(node.targets[0], ast.Name):
            return self._translate_statement(node)

        var_name = node.targets[0].id

        # Try to infer the type from the assigned value
        java_type = self._infer_type(node.value)

        # If type was explicitly annotated, use that instead
        if var_name in self.variable_types:
            java_type = self.variable_types[var_name]

        value = self._translate_expression(node.value)
        return f"{self._indent()}private {java_type} {var_name} = {value};"

    def _translate_annotated_class_field(self, node: ast.AnnAssign) -> str:
        """Translate a class field with type annotation."""
        if not isinstance(node.target, ast.Name):
            return self._translate_statement(node)

        var_name = node.target.id

        # Get type from annotation
        if isinstance(node.annotation, ast.Name):
            java_type = self._python_to_java_type(node.annotation.id)
        elif isinstance(node.annotation, ast.Subscript) and isinstance(node.annotation.value, ast.Name):
            base_type = node.annotation.value.id
            if base_type == "List":
                java_type = "ArrayList<Object>"
                self.imports.add("java.util.ArrayList")
            elif base_type == "Dict":
                java_type = "HashMap<Object, Object>"
                self.imports.add("java.util.HashMap")
            else:
                java_type = "Object"
        else:
            java_type = "Object"

        if node.value:
            value = self._translate_expression(node.value)
            return f"{self._indent()}private {java_type} {var_name} = {value};"
        else:
            return f"{self._indent()}private {java_type} {var_name};"

    def _translate_function(self, node: ast.FunctionDef, is_method: bool) -> str:
        """Translate a Python function/method to Java."""
        result = []

        # Handle method vs static method vs main method
        is_static = False
        is_main = False

        for decorator in node.decorator_list:
            if isinstance(decorator, ast.Name) and decorator.id == "staticmethod":
                is_static = True

        # Special case for main method
        if node.name == "main" and not is_method:
            is_static = True
            is_main = True

        # Function signature
        return_type = "void"  # Default return type

        # Try to get return type from return statements
        for item in ast.walk(node):
            if isinstance(item, ast.Return) and item.value:
                inferred_type = self._infer_type(item.value)
                if inferred_type != "Object":  # Only use if it's more specific
                    return_type = inferred_type
                    break

        # Try to get return type from type annotations
        if node.returns:
            if isinstance(node.returns, ast.Name):
                return_type = self._python_to_java_type(node.returns.id)
            elif isinstance(node.returns, ast.Subscript) and isinstance(node.returns.value, ast.Name):
                base_type = node.returns.value.id
                if base_type == "List":
                    return_type = "ArrayList<Object>"
                    self.imports.add("java.util.ArrayList")
                elif base_type == "Dict":
                    return_type = "HashMap<Object, Object>"
                    self.imports.add("java.util.HashMap")

        # Method modifiers
        modifiers = []
        if is_method:
            if is_static:
                modifiers.append("public static")
            else:
                modifiers.append("public")
        else:
            if is_main:
                modifiers.append("public static")
            else:
                modifiers.append("private static")

        # Special case for main method
        if is_main:
            param_str = "String[] args"
            return_type = "void"
        else:
            # Process parameters
            params = []
            for arg in node.args.args:
                arg_name = arg.arg

                # Try to get type from type annotation
                if arg.annotation:
                    if isinstance(arg.annotation, ast.Name):
                        arg_type = self._python_to_java_type(arg.annotation.id)
                    elif isinstance(arg.annotation, ast.Subscript) and isinstance(arg.annotation.value, ast.Name):
                        base_type = arg.annotation.value.id
                        if base_type == "List":
                            arg_type = "ArrayList<Object>"
                            self.imports.add("java.util.ArrayList")
                        elif base_type == "Dict":
                            arg_type = "HashMap<Object, Object>"
                            self.imports.add("java.util.HashMap")
                        else:
                            arg_type = "Object"
                    else:
                        arg_type = "Object"
                else:
                    arg_type = "Object"

                params.append(f"{arg_type} {arg_name}")

            param_str = ", ".join(params)

        # Build function signature
        signature = f"{' '.join(modifiers)} {return_type} {node.name}({param_str})"

        result.append(f"{self._indent()}{signature} {{")
        self.indent_level += 1

        # Process function body
        for item in node.body:
            result.append(self._translate_statement(item))

        # Add default return statement for non-void methods if not present
        if return_type != "void" and not any(isinstance(item, ast.Return) for item in node.body):
            if return_type == "int":
                result.append(f"{self._indent()}return 0;")
            elif return_type == "boolean":
                result.append(f"{self._indent()}return false;")
            elif return_type == "double" or return_type == "float":
                result.append(f"{self._indent()}return 0.0;")
            elif return_type.startswith("ArrayList"):
                result.append(f"{self._indent()}return new {return_type}();")
                self.imports.add("java.util.ArrayList")
            elif return_type.startswith("HashMap"):
                result.append(f"{self._indent()}return new {return_type}();")
                self.imports.add("java.util.HashMap")
            elif return_type == "String":
                result.append(f"{self._indent()}return \"\";")
            else:
                result.append(f"{self._indent()}return null;")

        self.indent_level -= 1
        result.append(f"{self._indent()}}}")

        return "\n".join(result)

    def _translate_statement(self, node: ast.AST) -> str:
        """Translate a Python statement to Java."""
        if isinstance(node, ast.Assign):
            if len(node.targets) != 1:
                targets = []
                for target in node.targets:
                    targets.append(self._translate_expression(target))
                value = self._translate_expression(node.value)
                result = []
                for target in targets:
                    result.append(f"{self._indent()}{target} = {value};")
                return "\n".join(result)
            else:
                target = self._translate_expression(node.targets[0])
                value = self._translate_expression(node.value)

                # Try to infer the type for variable declarations
                if isinstance(node.targets[0], ast.Name):
                    var_name = node.targets[0].id
                    if var_name not in self.variable_types:
                        java_type = self._infer_type(node.value)
                        if "." not in target:  # Don't include type for attribute access
                            return f"{self._indent()}{java_type} {target} = {value};"

                return f"{self._indent()}{target} = {value};"

        elif isinstance(node, ast.AnnAssign):
            target = self._translate_expression(node.target)

            # Get type from annotation
            if isinstance(node.annotation, ast.Name):
                java_type = self._python_to_java_type(node.annotation.id)
            elif isinstance(node.annotation, ast.Subscript) and isinstance(node.annotation.value, ast.Name):
                base_type = node.annotation.value.id
                if base_type == "List":
                    java_type = "ArrayList<Object>"
                    self.imports.add("java.util.ArrayList")
                elif base_type == "Dict":
                    java_type = "HashMap<Object, Object>"
                    self.imports.add("java.util.HashMap")
                else:
                    java_type = "Object"
            else:
                java_type = "Object"

            if node.value:
                value = self._translate_expression(node.value)
                return f"{self._indent()}{java_type} {target} = {value};"
            else:
                return f"{self._indent()}{java_type} {target};"

        elif isinstance(node, ast.Expr):
            # Expression statement
            expr = self._translate_expression(node.value)
            if expr.strip():  # Check if the expression is not empty
                return f"{self._indent()}{expr};"
            return ""

        elif isinstance(node, ast.If):
            # If statement
            condition = self._translate_expression(node.test)
            result = [f"{self._indent()}if ({condition}) {{"]

            self.indent_level += 1
            for item in node.body:
                result.append(self._translate_statement(item))
            self.indent_level -= 1

            if node.orelse:
                result.append(f"{self._indent()}}} else {{")
                self.indent_level += 1
                for item in node.orelse:
                    result.append(self._translate_statement(item))
                self.indent_level -= 1

            result.append(f"{self._indent()}}}")
            return "\n".join(result)

        elif isinstance(node, ast.For):
            # For loop - try to convert Python for to Java for/foreach
            if isinstance(node.target, ast.Name) and isinstance(node.iter, ast.Call) and isinstance(node.iter.func, ast.Name):
                # Handle range-based for loops
                if node.iter.func.id == "range":
                    var_name = node.target.id

                    # Determine range parameters
                    start, stop, step = "0", "", "1"

                    if len(node.iter.args) == 1:
                        stop = self._translate_expression(node.iter.args[0])
                    elif len(node.iter.args) == 2:
                        start = self._translate_expression(node.iter.args[0])
                        stop = self._translate_expression(node.iter.args[1])
                    elif len(node.iter.args) == 3:
                        start = self._translate_expression(node.iter.args[0])
                        stop = self._translate_expression(node.iter.args[1])
                        step = self._translate_expression(node.iter.args[2])

                    # Build for loop
                    if step == "1":
                        loop_header = f"for (int {var_name} = {start}; {var_name} < {stop}; {var_name}++)"
                    else:
                        loop_header = f"for (int {var_name} = {start}; {var_name} < {stop}; {var_name} += {step})"

                    result = [f"{self._indent()}{loop_header} {{"]

                    self.indent_level += 1
                    for item in node.body:
                        result.append(self._translate_statement(item))
                    self.indent_level -= 1

                    result.append(f"{self._indent()}}}")
                    return "\n".join(result)

            # Generic for-each loop
            target = self._translate_expression(node.target)
            iterable = self._translate_expression(node.iter)

            # Try to infer element type
            element_type = "Object"

            result = [f"{self._indent()}for ({element_type} {target} : {iterable}) {{"]

            self.indent_level += 1
            for item in node.body:
                result.append(self._translate_statement(item))
            self.indent_level -= 1

            result.append(f"{self._indent()}}}")
            return "\n".join(result)

        elif isinstance(node, ast.While):
            # While loop
            condition = self._translate_expression(node.test)
            result = [f"{self._indent()}while ({condition}) {{"]

            self.indent_level += 1
            for item in node.body:
                result.append(self._translate_statement(item))
            self.indent_level -= 1

            result.append(f"{self._indent()}}}")
            return "\n".join(result)

        elif isinstance(node, ast.Break):
            return f"{self._indent()}break;"

        elif isinstance(node, ast.Continue):
            return f"{self._indent()}continue;"

        elif isinstance(node, ast.Return):
            if node.value:
                value = self._translate_expression(node.value)
                return f"{self._indent()}return {value};"
            else:
                return f"{self._indent()}return;"

        elif isinstance(node, ast.Pass):
            return f"{self._indent()}// pass"

        elif isinstance(node, ast.Assert):
            test = self._translate_expression(node.test)
            if node.msg:
                msg = self._translate_expression(node.msg)
                return f"{self._indent()}assert {test} : {msg};"
            else:
                return f"{self._indent()}assert {test};"

        elif isinstance(node, ast.Raise):
            if node.exc:
                exc = self._translate_expression(node.exc)
                return f"{self._indent()}throw {exc};"
            else:
                return f"{self._indent()}throw new Exception();"

        elif isinstance(node, ast.Try):
            result = [f"{self._indent()}try {{"]

            self.indent_level += 1
            for item in node.body:
                result.append(self._translate_statement(item))
            self.indent_level -= 1

            for handler in node.handlers:
                if handler.type:
                    exc_type = self._translate_expression(handler.type)
                    exc_name = handler.name if handler.name else "e"
                    result.append(f"{self._indent()}}} catch ({exc_type} {exc_name}) {{")
                else:
                    result.append(f"{self._indent()}}} catch (Exception e) {{")

                self.indent_level += 1
                for item in handler.body:
                    result.append(self._translate_statement(item))
                self.indent_level -= 1

            if node.finalbody:
                result.append(f"{self._indent()}}} finally {{")

                self.indent_level += 1
                for item in node.finalbody:
                    result.append(self._translate_statement(item))
                self.indent_level -= 1

            result.append(f"{self._indent()}}}")
            return "\n".join(result)

        elif isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
            # Imports are handled separately
            return ""

        elif isinstance(node, ast.ClassDef) or isinstance(node, ast.FunctionDef):
            # These are handled separately
            return ""

        else:
            return f"{self._indent()}// Unsupported statement: {ast.dump(node)}"

    def _translate_expression(self, node: ast.AST) -> str:
        """Translate a Python expression to Java."""
        if node is None:
            return ""

        elif isinstance(node, ast.BinOp):
            # Binary operation
            left = self._translate_expression(node.left)
            right = self._translate_expression(node.right)

            op_map = {
                ast.Add: "+",
                ast.Sub: "-",
                ast.Mult: "*",
                ast.Div: "/",
                ast.Mod: "%",
                ast.Pow: "Math.pow",
                ast.FloorDiv: "/",  # Java doesn't have floor division operator
                ast.BitOr: "|",
                ast.BitAnd: "&",
                ast.BitXor: "^",
                ast.LShift: "<<",
                ast.RShift: ">>"
            }

            op_cls = node.op.__class__

            if op_cls == ast.Pow:
                self.imports.add("java.lang.Math")
                return f"{op_map[op_cls]}({left}, {right})"
            elif op_cls == ast.FloorDiv:
                return f"(int)Math.floor({left} {op_map[op_cls]} {right})"
            else:
                return f"({left} {op_map[op_cls]} {right})"

        elif isinstance(node, ast.UnaryOp):
            # Unary operation
            operand = self._translate_expression(node.operand)

            op_map = {
                ast.UAdd: "+",
                ast.USub: "-",
                ast.Not: "!",
                ast.Invert: "~"
            }

            return f"{op_map[node.op.__class__]}{operand}"

        elif isinstance(node, ast.Compare):
            # Comparison operation
            left = self._translate_expression(node.left)

            ops = []
            for i, (op, comparator) in enumerate(zip(node.ops, node.comparators)):
                right = self._translate_expression(comparator)

                op_map = {
                    ast.Eq: "==",
                    ast.NotEq: "!=",
                    ast.Lt: "<",
                    ast.LtE: "<=",
                    ast.Gt: ">",
                    ast.GtE: ">=",
                    ast.Is: "==",  # Java doesn't have 'is', use '=='
                    ast.IsNot: "!=",  # Java doesn't have 'is not', use '!='
                    ast.In: ".contains",  # Special handling for 'in'
                    ast.NotIn: "!"  # Special handling for 'not in'
                }

                op_cls = op.__class__

                if op_cls == ast.In:
                    if i == 0:
                        ops.append(f"{right}.contains({left})")
                    else:
                        # Chained 'in' operators are complex, simplify to just the first one
                        ops.append(f"{right}.contains({left})")
                elif op_cls == ast.NotIn:
                    if i == 0:
                        ops.append(f"!{right}.contains({left})")
                    else:
                        ops.append(f"!{right}.contains({left})")
                else:
                    if i == 0:
                        ops.append(f"{left} {op_map[op_cls]} {right}")
                    else:
                        # Java doesn't support chained comparisons natively
                        ops.append(f"{left} {op_map[op_cls]} {right}")

            if len(ops) == 1:
                return ops[0]
            else:
                # Join with && for chained comparisons
                return " && ".join(f"({op})" for op in ops)

        elif isinstance(node, ast.BoolOp):
            # Boolean operation
            op_map = {
                ast.And: "&&",
                ast.Or: "||"
            }

            values = []
            for value in node.values:
                values.append(self._translate_expression(value))

            return f" {op_map[node.op.__class__]} ".join(f"({val})" for val in values)

        elif isinstance(node, ast.Name):
            # Variable name
            name_map = {
                "True": "true",
                "False": "false",
                "None": "null",
                "self": "this"
            }

            return name_map.get(node.id, node.id)

        elif isinstance(node, ast.NameConstant):
            # Named constant
            value_map = {
                True: "true",
                False: "false",
                None: "null"
            }

            return value_map.get(node.value, str(node.value))

        elif isinstance(node, ast.Num):
            # Numeric literal
            return str(node.n)

        elif isinstance(node, ast.Constant):
            # Constant value (Python 3.8+)
            if node.value is None:
                return "null"
            elif isinstance(node.value, bool):
                return "true" if node.value else "false"
            elif isinstance(node.value, str):
                # Escape special characters
                escaped = node.value.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\t", "\\t")
                return f"\"{escaped}\""
            else:
                return str(node.value)

        elif isinstance(node, ast.Str):
            # String literal
            # Escape special characters
            escaped = node.s.replace("\\", "\\\\").replace("\"", "\\\"").replace("\n", "\\n").replace("\t", "\\t")
            return f"\"{escaped}\""

        elif isinstance(node, ast.List):
            # List literal
            self.imports.add("java.util.ArrayList")
            self.imports.add("java.util.Arrays")

            elements = []
            for elt in node.elts:
                elements.append(self._translate_expression(elt))

            # Use Arrays.asList for initialization
            if elements:
                return f"new ArrayList<>(Arrays.asList({', '.join(elements)}))"
            else:
                return "new ArrayList<>()"

        elif isinstance(node, ast.Dict):
            # Dictionary literal
            self.imports.add("java.util.HashMap")

            if not node.keys:
                return "new HashMap<>()"

            result = ["new HashMap<>() {{"]

            for key, value in zip(node.keys, node.values):
                key_str = self._translate_expression(key)
                value_str = self._translate_expression(value)
                result.append(f"    put({key_str}, {value_str});")

            result.append("}}")

            return "\n".join(result)

        elif isinstance(node, ast.Set):
            # Set literal
            self.imports.add("java.util.HashSet")
            self.imports.add("java.util.Arrays")

            elements = []
            for elt in node.elts:
                elements.append(self._translate_expression(elt))

            # Use Arrays.asList for initialization
            if elements:
                return f"new HashSet<>(Arrays.asList({', '.join(elements)}))"
            else:
                return "new HashSet<>()"

        elif isinstance(node, ast.Tuple):
            # Java doesn't have tuples, use List instead
            self.imports.add("java.util.ArrayList")
            self.imports.add("java.util.Arrays")

            elements = []
            for elt in node.elts:
                elements.append(self._translate_expression(elt))

            # Use Arrays.asList for initialization
            if elements:
                return f"Arrays.asList({', '.join(elements)})"
            else:
                return "new ArrayList<>()"

        elif isinstance(node, ast.Call):
            # Function call
            func = self._translate_expression(node.func)

            args = []
            for arg in node.args:
                args.append(self._translate_expression(arg))

            # Handle common Python built-ins
            if isinstance(node.func, ast.Name):
                if node.func.id == "print":
                    self.imports.add("java.lang.System")
                    return f"System.out.println({', '.join(args) if args else None})"

                elif node.func.id == "str":
                    return f"String.valueOf({args[0] if args else None})"

                elif node.func.id == "int":
                    if args:
                        return f"Integer.parseInt({args[0]})"
                    else:
                        return "0"

                elif node.func.id == "float" or node.func.id == "double":
                    if args:
                        return f"Double.parseDouble({args[0]})"
                    else:
                        return "0.0"

                elif node.func.id == "list":
                    self.imports.add("java.util.ArrayList")
                    if args:
                        return f"new ArrayList<>({args[0]})"
                    else:
                        return "new ArrayList<>()"

                elif node.func.id == "dict":
                    self.imports.add("java.util.HashMap")
                    return "new HashMap<>()"

                elif node.func.id == "set":
                    self.imports.add("java.util.HashSet")
                    if args:
                        return f"new HashSet<>({args[0]})"
                    else:
                        return "new HashSet<>()"

                elif node.func.id == "sum":
                    if args:
                        # For simple cases, we can use a stream
                        self.imports.add("java.util.stream.Collectors")
                        return f"{args[0]}.stream().collect(Collectors.summingDouble(val -> ((Number)val).doubleValue()))"
                    else:
                        return "0"

                elif node.func.id == "sorted":
                    self.imports.add("java.util.Collections")
                    if args:
                        # Create a new ArrayList to avoid modifying the original
                        self.imports.add("java.util.ArrayList")
                        return f"new ArrayList<>({args[0]}).stream().sorted().collect(Collectors.toList())"
                    else:
                        return "new ArrayList<>()"

                elif node.func.id == "any" or node.func.id == "all":
                    # These are harder to translate directly, use a simple implementation
                    func_name = "any" if node.func.id == "any" else "all"
                    return f"/* TODO: Implement equivalent of Python's {func_name}() */{args[0] if args else 'false'}"

            # Check for method calls on objects
            elif isinstance(node.func, ast.Attribute):
                obj = self._translate_expression(node.func.value)
                method = node.func.attr

                # Map common Python methods to Java equivalents
                method_map = {
                    "append": "add",
                    "extend": "addAll",
                    "remove": "remove",
                    "pop": "remove",
                    "clear": "clear",
                    "keys": "keySet",
                    "values": "values",
                    "items": "entrySet",
                    "get": "get",
                    "upper": "toUpperCase",
                    "lower": "toLowerCase",
                    "strip": "trim",
                    "replace": "replace",
                    "split": "split",
                    "join": "join",  # Handled specially below
                    "format": "format",  # Handled specially below
                    "startswith": "startsWith",
                    "endswith": "endsWith",
                    "find": "indexOf",
                    "rfind": "lastIndexOf"
                }

                if method in method_map:
                    method = method_map[method]

                # Special handling for some methods
                if method == "join" and isinstance(node.func.value, ast.Str):
                    # String.join() in Python is reversed in Java
                    self.imports.add("java.lang.String")
                    return f"String.join({obj}, {args[0]})"
                elif method == "format":
                    # String.format() in Java uses % placeholders
                    self.imports.add("java.lang.String")
                    return f"String.format({obj}, {', '.join(args)})"

                return f"{obj}.{method}({', '.join(args)})"

            # Regular function call
            return f"{func}({', '.join(args)})"

        elif isinstance(node, ast.Attribute):
            # Attribute access (obj.attr)
            obj = self._translate_expression(node.value)
            attr = node.attr

            # Map some common Python attributes to Java
            attr_map = {
                "__len__": "size",
                "__str__": "toString"
            }

            if attr in attr_map:
                attr = attr_map[attr]

            return f"{obj}.{attr}"

        elif isinstance(node, ast.Subscript):
            # Subscript access (obj[key])
            obj = self._translate_expression(node.value)

            if isinstance(node.slice, ast.Index):
                # Simple indexing (Python < 3.9)
                key = self._translate_expression(node.slice.value)
                return f"{obj}.get({key})"
            elif isinstance(node.slice, ast.Slice):
                # Slice (obj[start:stop:step])
                start = self._translate_expression(node.slice.lower) if node.slice.lower else "0"
                stop = self._translate_expression(node.slice.upper) if node.slice.upper else f"{obj}.size()"
                step = self._translate_expression(node.slice.step) if node.slice.step else "1"

                # Java doesn't have built-in slicing, need to use subList for Lists
                if step == "1":
                    return f"{obj}.subList({start}, {stop})"
                else:
                    # More complex slicing with step requires additional code
                    return f"/* TODO: Implement Python-like slicing with step */{obj}.subList({start}, {stop})"
            else:
                # Direct slice access (Python >= 3.9)
                key = self._translate_expression(node.slice)
                return f"{obj}.get({key})"

        elif isinstance(node, ast.ListComp):
            # List comprehension
            self.imports.add("java.util.ArrayList")
            self.imports.add("java.util.stream.Collectors")

            # Java doesn't have a direct equivalent, we'd need to use streams
            # This is a simplified version for basic comprehensions
            return f"/* TODO: Convert this comprehension to Java streams */new ArrayList<>()"

        elif isinstance(node, ast.Lambda):
            # Lambda expression
            args = []
            for arg in node.args.args:
                args.append(arg.arg)

            body = self._translate_expression(node.body)

            return f"({', '.join(args)}) -> {body}"

        else:
            return f"/* Unsupported expression: {type(node).__name__} */"

    def _process_import(self, node: ast.AST) -> None:
        """Process Python import statements and add equivalent Java imports."""
        if isinstance(node, ast.Import):
            for name in node.names:
                module = name.name

                # Map common Python modules to Java packages
                if module == "math":
                    self.imports.add("java.lang.Math")
                elif module == "random":
                    self.imports.add("java.util.Random")
                elif module == "datetime":
                    self.imports.add("java.time.LocalDateTime")
                    self.imports.add("java.time.format.DateTimeFormatter")
                elif module == "re":
                    self.imports.add("java.util.regex.Pattern")
                    self.imports.add("java.util.regex.Matcher")
                elif module == "json":
                    self.imports.add("java.util.Map")
                    self.imports.add("java.util.HashMap")
                    self.imports.add("com.fasterxml.jackson.databind.ObjectMapper")
                elif module == "os" or module == "os.path":
                    self.imports.add("java.io.File")
                    self.imports.add("java.nio.file.Path")
                    self.imports.add("java.nio.file.Paths")
                elif module == "sys":
                    self.imports.add("java.lang.System")
                elif module == "collections":
                    self.imports.add("java.util.Collections")
                    self.imports.add("java.util.Map")
                    self.imports.add("java.util.HashMap")
                    self.imports.add("java.util.List")
                    self.imports.add("java.util.ArrayList")

        elif isinstance(node, ast.ImportFrom):
            module = node.module

            # Map common Python modules to Java packages
            if module == "datetime" and any(name.name == "datetime" for name in node.names):
                self.imports.add("java.time.LocalDateTime")
            elif module == "collections":
                for name in node.names:
                    if name.name == "defaultdict":
                        self.imports.add("java.util.HashMap")
                    elif name.name == "Counter":
                        self.imports.add("java.util.HashMap")
                    elif name.name == "deque":
                        self.imports.add("java.util.ArrayDeque")
                    elif name.name == "namedtuple":
                        # Java doesn't have a direct equivalent, use a class instead
                        pass

    def _infer_type(self, node: ast.AST) -> str:
        """Try to infer Java type from Python expression."""
        if isinstance(node, ast.Constant):
            # Python 3.8+
            if node.value is None:
                return "Object"
            elif isinstance(node.value, bool):
                return "boolean"
            elif isinstance(node.value, int):
                return "int"
            elif isinstance(node.value, float):
                return "double"
            elif isinstance(node.value, str):
                return "String"
            else:
                return "Object"
        elif isinstance(node, ast.Num):
            # Python < 3.8
            if isinstance(node.n, int):
                return "int"
            elif isinstance(node.n, float):
                return "double"
            else:
                return "Number"
        elif isinstance(node, ast.Str):
            # Python < 3.8
            return "String"
        elif isinstance(node, ast.List) or isinstance(node, ast.ListComp):
            self.imports.add("java.util.ArrayList")
            return "ArrayList<Object>"
        elif isinstance(node, ast.Dict):
            self.imports.add("java.util.HashMap")
            return "HashMap<Object, Object>"
        elif isinstance(node, ast.Set):
            self.imports.add("java.util.HashSet")
            return "HashSet<Object>"
        elif isinstance(node, ast.Tuple):
            self.imports.add("java.util.List")
            return "List<Object>"
        elif isinstance(node, ast.NameConstant):
            # Python < 3.8
            if node.value is None:
                return "Object"
            elif isinstance(node.value, bool):
                return "boolean"
            else:
                return "Object"
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                # Try to infer type from function name
                func_name = node.func.id
                if func_name == "int":
                    return "int"
                elif func_name == "float":
                    return "double"
                elif func_name == "str":
                    return "String"
                elif func_name == "bool":
                    return "boolean"
                elif func_name == "list":
                    self.imports.add("java.util.ArrayList")
                    return "ArrayList<Object>"
                elif func_name == "dict":
                    self.imports.add("java.util.HashMap")
                    return "HashMap<Object, Object>"
                elif func_name == "set":
                    self.imports.add("java.util.HashSet")
                    return "HashSet<Object>"

            return "Object"
        else:
            return "Object"

    def _python_to_java_type(self, type_name: str) -> str:
        """Convert Python type name to Java type name."""
        type_map = {
            "int": "int",
            "float": "double",
            "str": "String",
            "bool": "boolean",
            "list": "ArrayList<Object>",
            "dict": "HashMap<Object, Object>",
            "set": "HashSet<Object>",
            "tuple": "List<Object>",
            "None": "void",
            "any": "Object",
            "object": "Object"
        }

        # Add imports for collection types
        if type_name == "list":
            self.imports.add("java.util.ArrayList")
        elif type_name == "dict":
            self.imports.add("java.util.HashMap")
        elif type_name == "set":
            self.imports.add("java.util.HashSet")
        elif type_name == "tuple":
            self.imports.add("java.util.List")

        return type_map.get(type_name, type_name)  # Default to the same name if not in map

    def _indent(self) -> str:
        """Return the current indentation string."""
        return self.indent_str * self.indent_level


# Command-line interface
def main():
    import argparse

    parser = argparse.ArgumentParser(description="Translate Python code to Java")
    parser.add_argument("input_file", help="Python file to translate")
    parser.add_argument("-o", "--output", help="Output Java file (default: stdout)")

    args = parser.parse_args()

    try:
        with open(args.input_file, "r") as f:
            python_code = f.read()

        translator = PythonToJavaTranslator()
        java_code = translator.translate(python_code)

        if args.output:
            with open(args.output, "w") as f:
                f.write(java_code)
        else:
            print(java_code)

        print(f"Translation completed successfully.")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
