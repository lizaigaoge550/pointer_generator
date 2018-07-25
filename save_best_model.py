import argparse
import os
import glob

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='training of the abstractor (ML)'
    )
    parser.add_argument('--checkpoint', required=True, help='root of the model')
    args = parser.parse_args()
    best = set()
    for file in glob.glob(os.path.join(args.checkpoint,'*')):
        if 'model' in file:
            number = int(file.split('/')[-1].split('_')[1])
            best.add(number)
    if os.path.exists('model_tmp'):
        os.rmdir('model_tmp')
        os.mkdir('model_tmp')
    else:
        os.mkdir('model_tmp')
    best = list(best)
    best = sorted(best, reverse=True)
    os.system("mv " + args.checkpoint+'/'+'model_'+str(best[0])+'*' + " model_tmp")
    if len(best) > 0:
        os.system("mv " + args.checkpoint+'/'+'model_'+str(best[1])+'*' + " model_tmp")
    checkpoint = args.checkpoint
    os.system("rm -rf "+args.checkpoint)
    os.system("mv model_tmp " + checkpoint)
