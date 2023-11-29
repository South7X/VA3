import argparse
from PIL import Image
from math import sqrt
import numpy as np
import os
import random
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from scipy.stats import norm
from utils import *

def parse_args():
    parser = argparse.ArgumentParser(description="evaluation script")
    parser.add_argument("--sample_path", type=str, default='./sample_results')
    parser.add_argument("--sample_name", type=str, default='anti-naf')
    parser.add_argument("--ar", type=float, default=0.1, 
                        help='acceptance rate for cp-k')
    parser.add_argument("--sim_thres", type=float, default=0.4241, 
                        help='sscd similarity threshold for copyright infringement judgment')
    parser.add_argument("--num_samples", type=int, default=100, 
                        help='number of samples for evaluation')
    parser.add_argument("--amp_steps", type=int, default=100, 
                        help='amplification steps')
    
    # For bandit prompt selection
    parser.add_argument("--bandit_select", default=False, action='store_true',
                        help='prompt selection using bandit methods')
    parser.add_argument("--bandit_mode", type=str, default='max', 
                        help='choose from max or cdf for epsilon-greedy-max/-cdf')
    parser.add_argument("--candidates_name", type=str, default='pez,anti-naf',
                        help='prompt candidates names, seperated by comma')
    parser.add_argument("--eps_mode", type=str, default='linear',
                        help='epsilon mode: linear, fix, exp')
    parser.add_argument("--m", type=int, default=10, help='trial number for each arm')
    parser.add_argument("--eps", type=float, default=0.5, help='initial value of epsilon')

    parser.add_argument("--save_path", type=str, default='./eval_results')
    parser.add_argument("--example_num", type=int, default=36)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return args


def add_border(image, color):
    width, height = image.size
    if color == 'gray':
        border_color = (128, 128, 128)  # Gray color
    else:
        border_color = (255, 0 , 0)  # Red color
    bordered_image = Image.new('RGB', (width + 2 * 20, height + 2 * 20), border_color)
    bordered_image.paste(image, (20, 20))
    return bordered_image


def combine_images(images, save_name):
    grid_width = grid_height = int(sqrt(args.example_num))
    image_width, image_height = images[0].size
    canvas_width = grid_width * image_width
    canvas_height = grid_height * image_height
    canvas = Image.new('RGB', (canvas_width, canvas_height))

    for i in range(grid_height):
        for j in range(grid_width):
            index = i * grid_width + j 
            canvas.paste(images[index], (j * image_width, i * image_height))

    canvas.save(save_name,'PDF',resolution=200)  
   

def get_images(img_ids, infringe_ids, save_name):
    images = []
    for i,id in enumerate(img_ids):
        filename = os.path.join(args.sample_path, args.sample_name,'cpfree','images',f'{args.sample_name}_sample_{id}.png')
        img = Image.open(filename)
        if id in infringe_ids:
            img = add_border(img, 'red')
        else:
            img = add_border(img, 'gray')
        images.append(img)
    combine_images(images, save_name)


def save_examples(length, amp_samples, pass_thres_ids, infringe_ids):
    save_name_0 = os.path.join(args.save_path, f'{args.sample_name}_cpfree_ac{args.ar}_amp{args.amp_steps}.pdf')
    save_name_1 = os.path.join(args.save_path, f'{args.sample_name}_no_protection.pdf')
    save_name_2 = os.path.join(args.save_path, f'{args.sample_name}_cpfree_ac{args.ar}.pdf')
    
    # -----no protection-----
    print(f'Save examples w/o copyright protection to {save_name_1}')
    random_sample_ids = random.sample(list(range(0,length)),k=args.example_num)
    get_images(random_sample_ids, infringe_ids, save_name_1)

    # -----cpfree w/o amp-------
    print(f'Save examples w/ copyright protection to {save_name_2}')
    remain_pass_ids = list(set(pass_thres_ids)-set(random_sample_ids))
    r = 0
    pass_thres_sample_ids = []
    for i in random_sample_ids:
        if i in pass_thres_ids:
            pass_thres_sample_ids.append(i)
        else:
            pass_thres_sample_ids.append(remain_pass_ids[r])
            r += 1
    get_images(pass_thres_sample_ids, infringe_ids, save_name_2)

    # -----cpfree amp------
    print(f'Save amplification examples w/ copyright protection to {save_name_0}')
    example_boost_sample_ids = random.sample(amp_samples, k=args.example_num)
    get_images(example_boost_sample_ids, infringe_ids, save_name_0)


def save_bandit_examples(candidate_names, boost_sample_id, actions, infringe_ids):
    images = []
    save_name = os.path.join(args.save_path, f'bandit_{args.bandit_mode}_cpfree_ac{args.ar}.pdf')
    for i,id in enumerate(boost_sample_id):
        image_name = candidate_names[actions[i]]
        filename = os.path.join(args.sample_path,image_name,'cpfree','images',f'{image_name}_sample_{id}.png')
        img = Image.open(filename)

        if id in infringe_ids[actions[i]]:
            img = add_border(img, 'red')
        else:
            img = add_border(img, 'gray')
        images.append(img)
    print(f'Save bandit examples to {save_name}')
    combine_images(images, save_name)


def amplification(accept_rate, sorted_rho_sim_matrix, length, sim, infringe_ids):
    candidates_num = int(length*accept_rate)

    pass_rho_sim_matrix = sorted_rho_sim_matrix[:,:candidates_num]
    pass_thres_ids = [int(pos) for pos in pass_rho_sim_matrix[2]]
    non_amp_samples = []
    non_amp_infringe_samples = []

    for _ in range(args.num_samples):
        pick_one = pass_thres_ids[np.random.randint(0, candidates_num)]
        non_amp_samples.append(pick_one)
        if pick_one in infringe_ids:
            non_amp_infringe_samples.append(pick_one)

    sim_pos_pairs = [(sim[int(pos)], int(pos)) for pos in pass_rho_sim_matrix[2]]

    amp_samples = []
    amp_infringe_samples = []
    for i in range(args.num_samples):
        max_sim  = 0.
        for _ in range(args.amp_steps):
            pos = np.random.randint(0, candidates_num)
            if sim_pos_pairs[pos][0] > max_sim:
                max_sim = sim_pos_pairs[pos][0]
                sample_id = sim_pos_pairs[pos][1]
        amp_samples.append(sample_id)
        if sample_id in infringe_ids:
            amp_infringe_samples.append(sample_id)
        
    return pass_thres_ids, amp_samples, amp_infringe_samples, non_amp_samples, non_amp_infringe_samples


def get_single_far():
    rho = np.load(os.path.join(args.sample_path,args.sample_name,'cpfree',f'{args.sample_name}_rho.npy'))
    sim = np.load(os.path.join(args.sample_path,args.sample_name,'cpfree',f'{args.sample_name}_sscd_score.npy'))
    length = len(rho)
    rho_sim_matrix = np.array([rho, sim, list(range(length))])
    sorted_indices = np.argsort(rho_sim_matrix[0])
    sorted_rho_sim_matrix = rho_sim_matrix[:, sorted_indices]
    infringe_ids = rho_sim_matrix[2][np.where(sim > args.sim_thres)[0]].astype(int)

    pass_thres_ids, amp_samples, amp_infringe_samples, non_amp_samples, non_amp_infringe_samples = amplification(
        args.ar, sorted_rho_sim_matrix, length, sim, infringe_ids)
    far_non_amp = len(non_amp_infringe_samples) / len(non_amp_samples)
    far_amp = len(amp_infringe_samples) / len(amp_samples)
    cir = len(infringe_ids) / length
    save_examples(length, amp_samples, pass_thres_ids, infringe_ids)

    return cir, far_non_amp, far_amp


def get_cdf_estimate(history):
    history = np.array(history)
    mean_i = np.mean(history)
    var_i = np.var(history)

    std_dev = var_i**0.5
    cdf_value = norm.cdf(args.sim_thres, loc=mean_i, scale=std_dev)
    return cdf_value


def eps_greedy_pick(sim_pos_pairs):

    total_samples = len(sim_pos_pairs[0])
    bandit_n = len(sim_pos_pairs)
    max_sim = 0.
    actions = []
    estimates = [0.0] * bandit_n
    estimate_history = {key:[] for key in range(bandit_n)}
    
    for t in range(args.amp_steps):
        if t < args.m * bandit_n:
            i = t % bandit_n
        else:
            tt = t - args.m * bandit_n
            total = args.amp_steps - args.m * bandit_n
            if 'linear' in args.eps_mode:
                cur_eps = args.eps * (1 - tt / total)
            elif 'exp' in args.eps_mode:
                cur_eps =  args.eps / (2 ** tt)
            else:   # fix eps
                cur_eps = args.eps

            if np.random.random() < cur_eps:
                i = np.random.randint(0, bandit_n)
            else: 
                i = max(range(bandit_n), key=lambda x: estimates[x])
        
        pos = np.random.randint(0, total_samples)

        # get reward
        pick_one = sim_pos_pairs[i][pos][0]
        
        # update amplification sample with highest sim score
        if pick_one > max_sim:  
            max_sim = pick_one
            amp_sample_id = sim_pos_pairs[i][pos][1]
            best_action = i
        
        # update estimate
        if 'cdf' in args.bandit_mode:  # eps-greedy-cdf
            estimate_history[i].append(pick_one)
            estimates[i] = 1 - get_cdf_estimate(estimate_history[i])
        elif 'max' in args.bandit_mode:  # eps-greedy-max  
            estimates[i] = max(estimates[i], pick_one)

        actions.append(i)

    return amp_sample_id, best_action, actions


def get_bandit_amp_far():
    far_amp_bandit = []
    matrix_all = []
    infringe_ids = []
    sims = []
    candidate_names = args.candidates_name.split(',')
    for image_name in candidate_names:  
        rho = np.load(os.path.join(args.sample_path,image_name,'cpfree',f'{image_name}_rho.npy'))
        sim = np.load(os.path.join(args.sample_path,image_name,'cpfree',f'{image_name}_sscd_score.npy'))
        length = len(rho)
        rho_sim_matrix = np.array([rho, sim, list(range(length))])
        sorted_indices = np.argsort(rho_sim_matrix[0])
        sorted_rho_sim_matrix = rho_sim_matrix[:, sorted_indices]
        sorted_rhonum_sim_matrix = np.array([list(range(length)),       # use sorted rho no. replace actual rho value
                                             sorted_rho_sim_matrix[1],  # sim
                                             sorted_rho_sim_matrix[2]]) # id

        infringe = rho_sim_matrix[2][np.where(sim > args.sim_thres)[0]].astype(int)
        infringe_ids.append(infringe)
        matrix_all.append(sorted_rhonum_sim_matrix)
        sims.append(sim)
    matrix_all = np.array(matrix_all)
    
    infringe_samples_id = []
    infringe_samples_from = []

    candidates_num = int(length*args.ar)
    pass_matrix_all = matrix_all[:,:,:candidates_num]
    sim_pos_pairs = [[(sims[i][int(pos)], int(pos)) for pos in pass_matrix_all[i][2]] for i in range(len(sims))]

    best_actions = []
    boost_sample_ids = []

    for id in range(args.num_samples):
        boost_sample_id, action, actions = eps_greedy_pick(sim_pos_pairs)
        best_actions.append(action)
        boost_sample_ids.append(boost_sample_id)
        if boost_sample_id in infringe_ids[action]:
            infringe_samples_id.append(boost_sample_id)
            infringe_samples_from.append(action)
    far_amp_bandit = len(infringe_samples_id) / args.num_samples
    save_bandit_examples(candidate_names, boost_sample_ids, best_actions, infringe_ids)
    return far_amp_bandit


if __name__ == '__main__':
    args = parse_args()
    set_random_seed(args.seed)
    if not os.path.exists(args.save_path):
        os.makedirs(args.save_path)
    
    if args.bandit_select:
        far = get_bandit_amp_far()
        print('Bandit mode: {}, FAR at AR {} is: {}'.format(args.bandit_mode, args.ar, far))
    else:
        cir, far_non_amp, far_amp = get_single_far()
        print('CIR: {}\nw/o Amp. FAR at AR {} is: {}\nw/ Amp. FAR at AR {} is: {}'.format(
            cir, args.ar, far_non_amp, args.ar, far_amp))
    