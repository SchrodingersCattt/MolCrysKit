"""Resource bounds and numeric validation tolerances for Mercury XML input."""

# Both compressed and expanded documents are bounded before XML parsing.
MRYX_MAX_COMPRESSED_BYTES = 16 * 1024 * 1024
MRYX_MAX_XML_BYTES = 64 * 1024 * 1024
MRYX_MAX_XML_NODES = 500_000
MRYX_MAX_XML_DEPTH = 128
MRYX_CELL_TOLERANCE = 1e-6
MRYX_COORDINATE_TOLERANCE_A = 1e-4
