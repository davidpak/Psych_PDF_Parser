import argparse

class Switch:
    def __init__(self, name, short, type, value):
        self.name = name
        self.short = short
        self.type = type
        self.value = value

def parse_args():
    parser = argparse.ArgumentParser(description='Script summarizer')

    switches = [
        Switch("verbose", short="v", type=float, value=3.0),
        Switch("no_general_overview", short="g", type=bool, value=False),
        Switch("no_key_concepts", short="k", type=bool, value=False),
        Switch("no_section_by_section", short="s", type=bool, value=False),
        Switch("no_additional_information", short="a", type=bool, value=False),
        Switch("no_helpful_vocabulary", short="he", type=bool, value=False),
        Switch("no_explain_to_5th_grader", short="e", type=bool, value=False),
        Switch("no_conclusion", short="c", type=bool, value=False)
    ]

    for switch in switches:
        parser.add_argument(f'-{switch.short}', f'--{switch.name}', type=switch.type, default=switch.value,
                            help=f'Description for {switch.name} (default: {switch.value})')

    args = parser.parse_args()

    for switch in switches:
        setattr(args, switch.name, getattr(args, switch.name, switch.value))

    return args

def main():
    args = parse_args()

    # Now you can access the values of the switches like this:
    print(f"Verbose: {args.verbose}")
    print(f"No General Overview: {args.no_general_overview}")
    print(f"No Key Concepts: {args.no_key_concepts}")
    # ... and so on

    # Your script logic goes here using the values of the switches

if __name__ == '__main__':
    main()
