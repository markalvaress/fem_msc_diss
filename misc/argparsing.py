from argparse import ArgumentParser, BooleanOptionalAction

parser = ArgumentParser()
parser.add_argument ("--regmesh", help = "Whether to use a regular mesh (identical cells equally spaced) or an irregular mesh (spline).", action = BooleanOptionalAction, default = True)
args = parser.parse_args()
print(args)