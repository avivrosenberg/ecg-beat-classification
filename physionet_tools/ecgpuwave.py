import os
import re
import subprocess
import warnings
import glob

from .consts import ECGPUWAVE_BIN


class ECGPuWave(object):
    """
    A wrapper for PhysioNet's ecgpuwave tool, which segments WCG beats.
    See: https://www.physionet.org/physiotools/ecgpuwave/
    """

    def __init__(self, ecgpuwave_bin=ECGPUWAVE_BIN):
        self.ecgpuwave_bin = ecgpuwave_bin

    def __call__(self, record: str, out_ann_ext: str,
                 in_ann_ext: str=None, signal: int=None,
                 from_time: str=None, to_time: str=None):
        """
        Runs the ecgpuwave tool on a given record, producing an annotation
        file with a specified extension.

        :param record: Path to PhysioNet record, e.g. foo/bar/123 (no file
            extension allowed).
        :param out_ann_ext: The extension of the annotation file to create.
        :param in_ann_ext: Read an annotation file with the given extension
            as input to specify beat types in the record.
        :param signal: The index of the signal (channel) in the record to
            analyze.
        :param from_time: Start at the given time. Should be a string in one of
            the PhysioNet time formats (see link below).
        :param to_time: Stop at the given time. Should be a string in one of
            the PhysioNet time formats (see link below).
        :return: True if ran without error.

        PhysioNet time formats:
        https://www.physionet.org/physiotools/wag/intro.htm#time
        """

        rec_dir = os.path.dirname(record)
        rec_name = os.path.basename(record)
        ecgpuwave_rel_path = os.path.relpath(self.ecgpuwave_bin, rec_dir)

        ecgpuwave_command = [
            ecgpuwave_rel_path,
            '-r', rec_name,
            '-a', out_ann_ext,
        ]

        if in_ann_ext:
            ecgpuwave_command += ['-i', in_ann_ext]

        if signal:
            ecgpuwave_command += ['-s', str(signal)]

        if from_time:
            ecgpuwave_command += ['-f', from_time]

        if to_time:
            ecgpuwave_command += ['-t', to_time]

        try:
            ecgpuwave_result = subprocess.run(
                ecgpuwave_command,
                check=True, shell=False, universal_newlines=True, timeout=10,
                cwd=rec_dir, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )

            # ecgpuwave can sometimes fail but still return 0, so need to
            # also check the stderr output.
            if ecgpuwave_result.stderr:
                # Annoying case: sometimes ecgpuwave writes to stderr but it's
                # not an error...
                if not re.match(r'Rearranging annotations[\w\s.]+done!',
                                ecgpuwave_result.stderr):
                    raise subprocess.CalledProcessError(0, ecgpuwave_command)

        except subprocess.CalledProcessError as process_err:
            warnings.warn(f'Failed to run ecgpuwave on record '
                          f'{record}:\n'
                          f'stderr: {ecgpuwave_result.stderr}\n'
                          f'stdout: {ecgpuwave_result.stdout}\n')
            return False

        except subprocess.TimeoutExpired as timeout_err:
            warnings.warn(f'Timed-out runnning ecgpuwave on record '
                          f'{record}: '
                          f'{ecgpuwave_result.stdout}')
            return False
        finally:
            # Remove tmp files created by ecgpuwave
            for tmpfile in glob.glob(f'{rec_dir}/fort.*'):
                os.remove(tmpfile)

        return True
