import matplotlib.pyplot as plt
from mol_metrics import *
from rdkit import Chem
import seaborn as sns
import matplotlib
import numpy as np
matplotlib.use('Agg')


def generate_parity_plot(ground_truth, predictions):
    # Convert lists to NumPy arrays
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)

    plt.scatter(ground_truth, predictions)
    # draw line of best fit
    m, b = np.polyfit(ground_truth, predictions, 1)
    plt.plot(ground_truth, m * ground_truth + b)
    # add labels of correlation coefficient
    # correlation coefficient
    r = np.corrcoef(ground_truth, predictions)[0, 1]
    # pearson's r squared
    r2 = sklearn.metrics.r2_score(ground_truth, predictions)
    plt.legend(
        ["Data", "y = {:.2f}x + {:.2f}; r={:.2f}; r2={:.2f}".format(m, b, r, r2)], loc="upper left")
    plt.xlabel("Ground Truth")
    plt.ylabel("Predictions")
    plt.title("Ground Truth vs Predictions")
    plt.savefig("zpve_parity_plot.png")

# ============================================================================
# Show Top-12 Molecules


def top_mols_show(filename, properties):
    """
                filename: NEGATIVE FILES (generated dataset of SMILES)
                properties: 'druglikeness' or 'solubility' or 'synthesizability'
    """
    mols, scores = [], []
    # Read the generated SMILES data
    smiles = open(filename, 'r').read()
    smiles = list(smiles.split('\n'))

    if properties == 'druglikeness':
        scores = batch_druglikeness(smiles)
    elif properties == 'synthesizability':
        scores = batch_SA(smiles)
    elif properties == 'solubility':
        scores = batch_solubility(smiles)

        # Sort the scores
    dic = dict(zip(smiles, scores))
    dic = sorted(dic.items(), key=lambda x: x[1], reverse=True)

    flag = 0
    top_mols = []
    for i in range(len(dic)):
        if flag < 12:
            if properties == 'synthesizability':
                if dic[i][0] not in top_mols and dic[i][1] > 0.95 and QED.default(Chem.MolFromSmiles(dic[i][0])) >= 0.5:
                    flag += 1
                    top_mols.append(Chem.MolFromSmiles(dic[i][0]))
                    print(dic[i][0], '\t %.3f' % dic[i][1])
            else:
                if dic[i][0] not in top_mols:
                    flag += 1
                    top_mols.append(Chem.MolFromSmiles(dic[i][0]))
                    print(dic[i][0], '\t %.3f' % dic[i][1])
    return top_mols


# Figure out the distributions
def distribution(real_file, gan_file, wgan_file):
    """
            real_file: original training dataset
            gan_file: the file of generated STGAN data
            wgan_file: the file of generated ST(W)GAN data
    """

    # Read trian Dataset
    real_lines = open(real_file, 'r').read()
    real_lines = list(real_lines.split('\n'))
    # Read STGAN results
    gan_lines = open(gan_file, 'r').read()
    gan_lines = list(gan_lines.split('\n'))
    # Read ST(W)GAN results
    wgan_lines = open(wgan_file, 'r').read()
    wgan_lines = list(wgan_lines.split('\n'))

    # Read the novel SMILES of STGAN
    gan_valid, gan_novelty = [], []
    for s in gan_lines:
        mol = Chem.MolFromSmiles(s)
        if mol and s != '':
            gan_valid.append(s)
    for s in list(set(gan_valid)):
        if s not in real_lines:
            gan_novelty.append(s)
    gan_lines = gan_novelty

    # Read the novel SMILES of ST(W)GAN
    wgan_valid, wgan_novelty = [], []
    for s in wgan_lines:
        mol = Chem.MolFromSmiles(s)
        if mol and s != '':
            wgan_valid.append(s)
    for s in list(set(wgan_valid)):
        if s not in real_lines:
            wgan_novelty.append(s)
    wgan_lines = wgan_novelty

    # Compute property scores for real dataset, STGAN and ST(W)GAN
    for name in ['QED Score', 'SA Score', 'logP Score']:
        real_scores, gan_scores, wgan_scores = [], [], []
        if name == 'QED Score':
            real_scores = batch_druglikeness(real_lines)
            gan_scores = batch_druglikeness(gan_lines)
            wgan_scores = batch_druglikeness(wgan_lines)
        elif name == 'SA Score':
            real_scores = batch_SA(real_lines)
            gan_scores = batch_SA(gan_lines)
            wgan_scores = batch_SA(wgan_lines)
        elif name == 'logP Score':
            real_scores = batch_solubility(real_lines)
            gan_scores = batch_solubility(gan_lines)
            wgan_scores = batch_solubility(wgan_lines)
        # Compute the mean socres
        avg = [np.mean(real_scores), np.mean(gan_scores), np.mean(wgan_scores)]
        # Print Mean socres
        print('Mean Real {}: {:.3f}'.format(name, avg[0]))
        print('Mean GAN {}: {:.3f}'.format(name, avg[1]))
        print('Mean WGAN {}: {:.3f}\n'.format(name, avg[2]))

        # Plot distribution figures for real dataset, STGAN and ST(W)GAN
        plt.subplots(figsize=(12, 7))
        # Font size
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.xlabel(name, size=15)
        plt.ylabel('Density', size=15)
        # Set the min and max value of X axis
        plt.xlim(-0.1, 1.1)

        sns.distplot(real_scores, hist=False, kde_kws={
                     'shade': True, 'linewidth': 1}, label='ORIGINAL')
        sns.distplot(gan_scores, hist=False, kde_kws={
                     'shade': True, 'linewidth': 1}, label='STGAN')
        sns.distplot(wgan_scores, hist=False, kde_kws={
                     'shade': True, 'linewidth': 1}, label='ST(W)GAN')
        plt.legend(loc='upper right', prop={'size': 15})
        plt.savefig('res/' + name + '.pdf')
