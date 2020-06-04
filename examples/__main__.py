
from glob import glob
from importlib import import_module
from os.path import basename, isfile, join
import sys

from . import cartpole
from . import mountaincar


def main(example_name):
    """
    Main function.

    Args:
        example_name (str): The name of an example (e.g. 'cartpole').
    """
    files = glob(join('examples', '*.py'))
    examples = [basename(f)[:-3] for f in files if isfile(f) and not f.endswith('__init__.py') and not f.endswith('__main__.py')]
    example_name = example_name.lower()
    if example_name in examples:
        module = import_module(f'.{example_name}', package='examples')
        module.main()
    else:
        examples_str = '\n\t'.join(sorted(examples))
        print(f'No example with that name was found!\n\nOptions:\n\t{examples_str}')


if __name__ == '__main__':
    if len(sys.argv) == 1 or len(sys.argv) > 2:
        print('Usage: python -m examples EXAMPLE_NAME')
        sys.exit(1)
    main(sys.argv[1])
