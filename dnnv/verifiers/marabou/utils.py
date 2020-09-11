import tempfile

def get_marabou_properties() -> str:
    with tempfile.NamedTemporaryFile(
        mode="w+", dir=None, delete=False
    ) as properties:
        properties.write("y0 <= 0\n")
        return properties.name