"""Emit a function adding two non-splat constants of shape ROWSxCOLS."""
import sys

rows, cols = int(sys.argv[1]), int(sys.argv[2])
t = f"tensor<{rows}x{cols}xf32>"
# dense<"0x...">: raw little-endian bytes, the compact form MLIR prints for
# large constants. Element i holds float(i % 7) so the tensor is not a splat.
import struct
payload = b"".join(struct.pack("<f", float(i % 7)) for i in range(rows * cols))
hexdata = "0x" + payload.hex().upper()
print(f"""func.func @f() -> {t} {{
  %a = topt.constant dense<"{hexdata}"> : {t}
  %s = topt.add %a, %a : {t}
  return %s : {t}
}}""")
