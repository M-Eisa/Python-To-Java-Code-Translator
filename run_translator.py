from python_to_java_translator import PythonToJavaTranslator
translator = PythonToJavaTranslator()
python_code = """
def hello():
    print("Hello, world!")
hello()
"""
java_code = translator.translate(python_code)
print(java_code)
