from ivande_combiner.module import hello_world


def test_module_return_right_text():
    expected = "hello dude_11"
    calculated = hello_world()
    assert expected == calculated
