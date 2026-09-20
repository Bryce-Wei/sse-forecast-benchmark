"""Run the original ARIMA, validation comparison, and final plots in sequence."""
import argparse
from pathlib import Path
import subprocess
import sys


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--smoke',action='store_true',help='Check two original fits and one validation fit per configuration only.')
    parser.add_argument('--workers',type=int,default=3,help='Parallel processes for the full comparison.')
    args=parser.parse_args()
    if args.workers<1:
        parser.error('--workers must be at least 1')
    folder=Path(__file__).resolve().parent
    commands=[['run_forecast.py'],['compare_models.py','--workers',str(args.workers)]]
    if args.smoke:
        for command in commands:
            command.append('--smoke')
    else:
        commands.append(['render_comparison.py'])
    for filename,*arguments in commands:
        subprocess.run([sys.executable,str(folder/filename),*arguments],check=True)
    print('ARIMA smoke checks passed; this does not reproduce full performance metrics.' if args.smoke else 'ARIMA full comparison completed.')


if __name__=='__main__':
    main()
