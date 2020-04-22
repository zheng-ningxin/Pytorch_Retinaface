#!/bin/env python 
import os
import re
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help='Path to the result file.')
    parser.add_argument('-o', '--outdir', default='./pics', help='Dir that save the drawed pics.')
    return parser.parse_args()

def draw(name, data, outdir):
    easy, mid, hard = [], [], []
    ax = []
    for x in data:
        ratio = float(x[0])
        ratio = 1 - ratio
        ax.append(ratio)
        easy.append(float(x[1][0]))
        mid.append(float(x[1][1]))
        hard.append(float(x[1][2]))
    plt.figure(figsize=(8, 4))
    plt.plot(ax, easy, label='easy', linestyle='--',color='green', marker='o')
    plt.plot(ax, mid, label='middle', linestyle='-', color='yellow',marker='*')
    plt.plot(ax, hard, label='hard', linestyle='-.', color='red',marker='v')
    plt.xlabel('Prune Ratio')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title(name)
    filename = os.path.join(outdir, '%s.jpg' % name)
    plt.savefig(filename, dpi=1000)


if __name__ == '__main__':
    args = parse_args()
    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)
    if not os.path.exists(args.file):
        raise('File %s not found' % args.file)
    with open(args.file, 'r') as fd:
        dic = json.load(fd)
        for layer_name in dic.keys():
            draw(layer_name, dic[layer_name], args.outdir)