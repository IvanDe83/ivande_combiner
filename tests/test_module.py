from ivande_combiner.module import hello_world


def test_module_return_right_text():
    expected = "hello world"
    calculated = hello_world()
    assert expected == calculated
