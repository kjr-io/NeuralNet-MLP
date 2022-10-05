import matplotlib.pyplot as plt

def construct_plot(epochs, errStore, learning_rate, plt_details):
    
    plt.figure(plt_details[2])
    plt.plot(range(epochs), errStore, plt_details[0], label=learning_rate)
    plt.legend(loc='upper right')
    plt.savefig(f'./output/{plt_details[1]}.png')

