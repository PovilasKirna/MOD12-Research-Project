import io
import sys

from research_project.helloworld import hello_world


def test_hello_world():
    assert "Hello, World!" == "Hello, World!"


def test_addition():
    assert 1 + 1 == 2


def test_hello_world_function():

    # Redirect stdout to capture print statements
    captured_output = io.StringIO()
    sys.stdout = captured_output

    # Call the function
    hello_world()

    # Reset redirect.
    sys.stdout = sys.__stdout__

    # Check if the output is as expected
    assert captured_output.getvalue().strip() == "Hello, World!"
