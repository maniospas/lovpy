from tests.lovpy.importer.sample_properties import get_threading_sample_properties


def test_visualize_thread_properties():
    properties = get_threading_sample_properties()
    properties[0].visualize("Property 1")
    properties[1].visualize("Property 2")
