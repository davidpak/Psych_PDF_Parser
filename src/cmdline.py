from typing import Union, Callable
import argparse

SwitchValue = Union[None, str, int, float, bool, Callable[[], Union[int, None]]]
SwitchType = Union[None, type[str], type[int], type[float], type[bool]]


class Switch:
    def __init__(self, name: str, short: Union[str, None] = None, value: SwitchValue = None, type: SwitchType = None):
        self.name = name
        self.short = name[0] if short is None else short
        self.value = value
        self.type = type

        if self.type is None and self.value is None:
            raise Exception('Switch must have a type or value')

        if callable(self.value) and self.type is not None:
            raise Exception('Switch cannot have a type and a callable value')

    def handle(self, args) -> None:
        value = getattr(args, self.name, self.value)

        if callable(value):
            value = value()

        setattr(args, self.name, value)


def parse(argv: list[str], switches: Union[list[Switch], None] = None) -> Union[object, int]:
    result_switches = {}

    parser = argparse.ArgumentParser(description='Summarize')

    if switches is not None:
        for switch in switches:
            parser.add_argument(f'-{switch.short}', f'--{switch.name}', type=switch.type, default=switch.value,
                                help=f'Description for {switch.name} (default: {switch.value})')

    args, remaining_argv = parser.parse_known_args(argv)

    if switches is not None:
        for switch in switches:
            switch.handle(args)
            result_switches[switch.name] = getattr(args, switch.name)

    return args, result_switches
