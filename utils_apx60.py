import matplotlib.pyplot as plt
import os
import glob


def plot_images():
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['xtick.top'] = True
    plt.rcParams['xtick.labeltop'] = False
    plt.rcParams['xtick.labelbottom'] = True
    plt.rcParams['xtick.bottom'] = True
    plt.rcParams['xtick.direction'] = 'in'
    plt.rcParams['xtick.minor.visible'] = True
    plt.rcParams['xtick.major.top'] = True
    plt.rcParams['xtick.minor.top'] = True
    plt.rcParams['xtick.minor.bottom'] = True
    plt.rcParams['xtick.alignment'] = 'center'

    plt.rcParams['ytick.left'] = True
    plt.rcParams['ytick.labelleft'] = True
    plt.rcParams['ytick.right'] = True
    plt.rcParams['ytick.minor.visible'] = True
    plt.rcParams['ytick.major.right'] = True
    plt.rcParams['ytick.major.left'] = True
    plt.rcParams['ytick.minor.right'] = True
    plt.rcParams['ytick.minor.left'] = True

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12

    plt.rcParams['legend.frameon'] = True
    plt.rcParams['legend.framealpha'] = 0.8
    plt.rcParams['legend.loc'] = 'best'
    plt.rcParams['legend.fancybox'] = True
    plt.rcParams['legend.fontsize'] = 12


class CreateResiduals:
    def __init__(self, X_values, Y_values, Offset, Gradient, max_val, range_factor):
        self.x = X_values
        self.y = Y_values
        self.offset = Offset
        self.gradient = Gradient
        self.max = max_val
        self.range = range_factor
        self.residuals = []
        self.residual_counts = []

        self.create()

    def create(self):
        print('creating residuals')
        for i in range(0, len(self.y)):
            calculated_level = self.offset + (self.gradient * self.x[i])

            Residuals = (self.y[i] - calculated_level) / (
                    self.range * self.max) * 100  # Equation 35 from EMVA1288-V3.1

            residualscts = (self.y[i] - calculated_level)

            self.residuals.append(Residuals)
            self.residual_counts.append(residualscts)

        # print('residuals [{}]'.format(self.residuals))
        # print('residual counts [{}]'.format(self.residual_counts))