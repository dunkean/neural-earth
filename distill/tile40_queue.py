"""Wait for verified successful parent jobs BEFORE taking the shared GPU lock."""
import argparse
import os
import re
import time

from distill.common import DATA
from distill.jobs import state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--after', action='append', required=True)
    parser.add_argument('--timeout', type=float, default=7200.)
    parser.add_argument('command', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ['--'] else args.command
    if not command or args.timeout <= 0 or any(not re.fullmatch('[a-z0-9_-]+', name) for name in args.after):
        raise ValueError('Safe parent job names, a positive timeout and a command are required.')
    started = time.monotonic()
    while True:
        pending = []
        for name in args.after:
            status = state(DATA/'jobs'/name)
            if status['status'] in ('failed', 'missing'):
                raise RuntimeError(f'Parent {name} is {status["status"]};no automatic retry.')
            if not status['live'] and status['status'] != 'complete':
                raise RuntimeError(f'Parent {name} has no verified live process and did not complete.')
            if status['live'] or status['status'] != 'complete':
                pending.append(name)
        if not pending:
            break
        if time.monotonic()-started > args.timeout:
            raise TimeoutError(f'Parents still pending:{pending}')
        print(f'Waiting before GPU lock:{pending}', flush=True)
        time.sleep(10.)
    print(f'Parents complete;executing:{command}', flush=True)
    os.execvp(command[0], command)


if __name__ == '__main__':
    main()
