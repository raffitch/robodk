"""Valve digital outputs: names for humans, indices for the controller.

The KUKA driver (RoboDK's ``robodksync`` KRL module) sets a digital output by
writing ``$OUT[<index>]``. It needs a NUMBER. Handing RoboDK's ``setDO`` the
string ``"IO_508"`` produced ``$OUT[0]`` on the cell — and KUKA's ``$OUT[]`` is
1-based, so index 0 is always invalid:

    KSS014444  Array index inadmissible  $OUT[0]  module robodksync570

That fault aborts the program on the controller before any motion, which is why
a dispatched layer "finished" in 0.5 s having never moved.

So the configured names stay readable (``IO_508``), and the index is derived
from them here — explicitly, and loudly when it cannot be, because silently
sending 0 is precisely the bug being fixed.
"""
from __future__ import annotations

import re

_TRAILING_DIGITS = re.compile(r"(\d+)\D*$")


def valve_output_index(output: "str | int") -> int:
    """The numeric ``$OUT`` index for a configured valve output.

    Accepts an int, a bare number, or a name whose digits are the index
    (``IO_508``, ``DO_12``, ``$OUT[7]``). Raises ``ValueError`` rather than
    guessing: an unresolvable name must fail here, not become ``$OUT[0]`` on the
    controller.
    """
    if isinstance(output, bool):        # bool is an int subclass; never an index
        raise ValueError(f"invalid valve output: {output!r}")
    if isinstance(output, int):
        index = output
    else:
        match = _TRAILING_DIGITS.search(str(output))
        if not match:
            raise ValueError(
                f"valve output {output!r} has no numeric index — the KUKA driver "
                "writes $OUT[<index>] and cannot resolve a name")
        index = int(match.group(1))
    if index < 1:
        raise ValueError(
            f"valve output {output!r} resolves to $OUT[{index}]; KUKA outputs are "
            "1-based, so index 0 is never valid")
    return index


def instructions_match(actual: "list[str]", outputs: "list[str]", value: int) -> bool:
    """Do ``actual`` program instructions set exactly these outputs to ``value``?

    Compared on the NUMBERS each instruction contains rather than an exact string.
    RoboDK's rendering of a digital-output instruction is not a stable contract
    (it differs between a named and a numeric output, and across versions), so
    pinning it would make a correct station fail verification. The pairing of
    index and value is what actually matters, and one instruction per output with
    nothing extra is what makes AirOff fail-safe.
    """
    indices = [valve_output_index(output) for output in outputs]
    if len(actual) != len(indices):
        return False
    for instruction, index in zip(actual, indices):
        numbers = [int(n) for n in re.findall(r"-?\d+", str(instruction))]
        if not numbers or index not in numbers or numbers[-1] != int(value):
            return False
    return True
