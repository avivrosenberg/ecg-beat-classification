import argparse
import os
import pathlib
import pprint
import re
import sys

import tqdm

MIN_AGE = 0
MAX_AGE = 120


def is_dir(dirname):
    if not os.path.isdir(dirname):
        raise argparse.ArgumentTypeError(f'{dirname} is not a directory')
    else:
        return dirname


def is_age(age_str):
    try:
        age = int(age_str)
        if age < MIN_AGE or age > MAX_AGE:
            raise ValueError()
        return age
    except ValueError:
        raise argparse.ArgumentTypeError(f'{age_str} is not a valid age')


def parse_cli():
    p = argparse.ArgumentParser(description='Split WFDB databases')
    p.set_defaults(handler_class=BaseSplitter)
    p.add_argument('--in', '-i', type=is_dir, dest='in_dir',
                   help='Input dir', required=True)
    p.add_argument('--out', '-o', type=str, dest='out_dir',
                   help='Output dir', required=False)
    p.add_argument('--symlink', '-s', action='store_true',
                   help='Create symlinks into output dir', required=False)

    # Subcommands
    sp = p.add_subparsers(help='Available splitters', dest='splitter_name')

    # Split by age
    sp_by_age = sp.add_parser(
        'by-age', help='Split dataset by age. Creates a groups of records, '
                       'each with records within a given age range.')
    sp_by_age.set_defaults(handler_class=SplitByAge)
    sp_by_age.add_argument('--age-group', '-g', type=is_age, nargs=2,
                           action='append', help='Age group',
                           dest='age_groups', metavar=('from_age', 'to_age'),
                           required=True)

    parsed = p.parse_args()
    if not parsed.splitter_name:
        p.error("Please specify a splitter to use")

    return parsed


class BaseSplitter(object):
    REC_EXTENSIONS = ('hea', 'dat')
    ANN_EXTENSIONS = ('atr', 'qrs', 'ecg', 'ecgatr')

    def __init__(self, in_dir, out_dir=None, **kwargs):
        if not out_dir:
            out_dir = in_dir

        self.in_dir = pathlib.Path(in_dir)
        self.out_dir = pathlib.Path(out_dir)

        header_files = self.in_dir.glob("**/*.hea")
        self.records = [h.parent.joinpath(h.stem)
                        for h in header_files if not h.is_symlink()]

        self._group_affiliation = {None: set(self.records)}

        if len(self.records) == 0:
            raise ValueError(f"Can't find any records in {in_dir}")

        self.stdout = sys.stdout if 'stdout' not in kwargs else kwargs[
            'stdout']
        self.stderr = sys.stderr if 'stderr' not in kwargs else kwargs[
            'stderr']

    def split(self):
        raise NotImplementedError('Must be implemented in deriving classes')

    def __iter__(self):
        return self.records.__iter__()

    @property
    def group_affiliation(self):
        return self._group_affiliation.copy()

    def add_to_group(self, rec_name, group):
        self._group_affiliation[None].remove(rec_name)

        members = self._group_affiliation.get(group, set())
        members.add(rec_name)
        self._group_affiliation[group] = members

    def get_group(self, rec_name):
        for group, members in self._group_affiliation.items():
            if rec_name in members:
                return group

    def get_affiliated_records(self):
        affiliated_records = set()
        for group, members in self._group_affiliation.items():
            if group:
                affiliated_records = affiliated_records.union(members)

        return affiliated_records

    def print_affiliation(self):
        for age_group, group_members in self._group_affiliation.items():
            members_stem = sorted(m.stem for m in group_members)
            print(f'* Group {age_group}, n={len(group_members):02d}:',
                  file=self.stdout)
            pprint.pprint(members_stem, width=120, compact=True,
                          stream=self.stdout)

    def symlink_record(self, rec_name, group_dir=None):
        if not group_dir:
            dest_dir = self.out_dir
        else:
            dest_dir = self.out_dir.joinpath(group_dir)

        rec_files = []
        for ext in (self.REC_EXTENSIONS + self.ANN_EXTENSIONS):
            rec_file = pathlib.Path(str.format("{}.{}", rec_name, ext))
            if rec_file.exists():
                rec_files.append(rec_file)

        for rec_file in rec_files:
            dest_file = dest_dir.joinpath(rec_file)
            if dest_file.exists():
                os.remove(str(dest_file))

            os.makedirs(str(dest_file.parent), exist_ok=True)

            dest_file.symlink_to(rec_file.absolute())

    def symlink_by_group(self, group_dirs=None):
        if not group_dirs:
            group_dirs = {g: str(g) for g in self._group_affiliation.keys()}

        print(f'Creating symlinks in {str(self.out_dir)}', file=self.stdout)

        progress_iter = tqdm.tqdm(self.get_affiliated_records())
        for record in progress_iter:
            progress_iter.set_description(record.stem)
            group = self.get_group(record)
            group_dir = group_dirs[group]
            self.symlink_record(record, group_dir)


class SplitByAge(BaseSplitter):
    AGE_REGEX = re.compile(
        r'#\s*(?:age:\s*)?(?P<age>[-.\d]+)\s+(?:sex:\s*)?[FM?]',
        flags=re.IGNORECASE)

    def __init__(self, age_groups, **kwargs):
        super().__init__(**kwargs)

        # Make sure all ages are valid
        for group in age_groups:
            assert len(group) == 2
            for age in group:
                assert age in range(MIN_AGE, MAX_AGE)

        self.groups = [tuple(sorted(g)) for g in age_groups]
        self.group_dirs = {g: f'age_{g[0]}_{g[1]}' for g in self.groups}

    def split(self):
        print(f'Splitting n={len(self.records)} records '
              f'by age groups={self.groups}', file=self.stdout)

        for rec_name in self:
            age = self.get_age(rec_name)
            if not age:
                print(f"Failed to determine age for {rec_name}, skipping...",
                      file=self.stderr)
                continue

            for age_group in self.groups:
                if age in range(*age_group):
                    self.add_to_group(rec_name, age_group)

    def get_age(self, rec_name):
        age = None
        header_path = f'{rec_name}.hea'
        with open(header_path, 'r') as header_file:
            header_contents = header_file.read()
            match = self.AGE_REGEX.search(header_contents)
            if match:
                age = int(float(match.group('age')))
        return age

    def symlink_by_group(self, **kwargs):
        super(SplitByAge, self).symlink_by_group(group_dirs=self.group_dirs)


if __name__ == '__main__':
    parsed_args = parse_cli()

    splitter = parsed_args.handler_class(**vars(parsed_args))
    splitter.split()

    print("")
    splitter.print_affiliation()

    if parsed_args.symlink:
        print("")
        splitter.symlink_by_group()

__all__ = [BaseSplitter, SplitByAge]

