import matplotlib.pyplot as plt

# Constructing the Plots for Any Given Problem Using Matplotlib
def construct_plot(epochs, errStore, learning_rate, plt_details):
    '''
    X-Axis: Epochs
    Y-Axis: Error
    plt_details[0]: Color
    plt_details[1]: Problem (XOR, OR, AND) for Saving
    plt_details[2]: Plot Number for Saving
    '''
    plt.figure(plt_details[2])
    plt.plot(range(epochs), errStore, plt_details[0], label=learning_rate)
    plt.legend(loc='upper right')
    plt.savefig(f'./output/{plt_details[1]}.png')

