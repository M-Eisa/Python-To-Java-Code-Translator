# Python to Java Translator

A source-to-source compiler that translates Python code to equivalent Java code using Abstract Syntax Tree (AST) parsing.

## Overview

This project provides a tool to automatically convert Python source code to semantically equivalent Java code. It parses Python code into an AST and then traverses the tree to generate corresponding Java syntax. The translator handles many common Python language features including:

- Classes and object-oriented programming constructs
- Functions and methods with parameter and return type inference
- Control flow structures (if/else, loops, try/except)
- Variable declarations with type inference
- Common Python data structures (lists, dictionaries, sets)
- Basic error handling and exception translation

## Features

- **AST-based Parsing**: Uses Python's built-in `ast` module for reliable source code analysis
- **Type Inference**: Automatically determines appropriate Java types for Python variables
- **Import Management**: Adds necessary Java imports for equivalent functionality
- **Object-Oriented Support**: Translates Python classes, inheritance, and methods
- **Command-Line Interface**: Simple CLI for file processing

## Installation

Clone this repository:

```bash
git clone https://github.com/M-Eisa/Python-To-Java-Code-Translator.git
```

No additional dependencies are required beyond the Python standard library.

## Usage

### Command Line

Translate a Python file:

```bash
python python_to_java_translator.py input.py -o Output.java
```

If no output file is specified, the Java code will be printed to the console:

```bash
python python_to_java_translator.py input.py
```

Running without arguments will demonstrate a simple example:

```bash
python python_to_java_translator.py
```

### As a Module

You can also use the translator in your own Python code:

```python
from python_to_java_translator import PythonToJavaTranslator

# Create a translator instance
translator = PythonToJavaTranslator()

# Translate Python code
python_code = """
def hello():
    print("Hello, world!")
hello()
"""

java_code = translator.translate(python_code)
print(java_code)
```

This example is included in `run_translator.py`.

## Examples

### Input (Python)

```python
class Calculator:
    def __init__(self, initial_value=0):
        self.value = initial_value
        self.history = []
    
    def add(self, number):
        self.value = self.value + number
        self.history.append("Added " + str(number))
        return self.value
    
    def subtract(self, number):
        self.value = self.value - number
        self.history.append("Subtracted " + str(number))
        return self.value
    
    def multiply(self, number):
        self.value = self.value * number
        self.history.append("Multiplied by " + str(number))
        return self.value
    
    def divide(self, number):
        if number == 0:
            print("Error: Cannot divide by zero")
            return self.value
        
        self.value = self.value / number
        self.history.append("Divided by " + str(number))
        return self.value
    
    def get_value(self):
        return self.value
    
    def print_history(self):
        print("Calculation history:")
        for operation in self.history:
            print("- " + operation)


# Test the calculator
def main():
    calc = Calculator(10)
    
    calc.add(5)
    calc.multiply(2)
    calc.subtract(7)
    calc.divide(2)
    
    print("Final value: " + str(calc.get_value()))
    calc.print_history()

if __name__ == "__main__":
    main()
```

### Output (Java)

```java
import java.lang.System;
import java.util.ArrayList;
import java.util.Arrays;

public class Calculator {
    public void __init__(Object self, Object initial_value) {
        this.value = initial_value;
        this.history = new ArrayList<>();
    }
    public void add(Object self, Object number) {
        this.value = (this.value + number);
        this.history.add(("Added " + String.valueOf(number)));
        return this.value;
    }
    public void subtract(Object self, Object number) {
        this.value = (this.value - number);
        this.history.add(("Subtracted " + String.valueOf(number)));
        return this.value;
    }
    public void multiply(Object self, Object number) {
        this.value = (this.value * number);
        this.history.add(("Multiplied by " + String.valueOf(number)));
        return this.value;
    }
    public void divide(Object self, Object number) {
        if (number == 0) {
            System.out.println("Error: Cannot divide by zero");
            return this.value;
        }
        this.value = (this.value / number);
        this.history.add(("Divided by " + String.valueOf(number)));
        return this.value;
    }
    public void get_value(Object self) {
        return this.value;
    }
    public void print_history(Object self) {
        System.out.println("Calculation history:");
        for (Object operation : this.history) {
            System.out.println(("- " + operation));
        }
    }
}
public static void main(String[] args) {
    Object calc = Calculator(10);
    calc.add(5);
    calc.multiply(2);
    calc.subtract(7);
    calc.divide(2);
    System.out.println(("Final value: " + String.valueOf(calc.get_value())));
    calc.print_history();
}
```
