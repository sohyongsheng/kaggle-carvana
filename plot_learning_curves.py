import numpy
import matplotlib.pyplot
import pylab
import sys
    
def plot_learning_curves(experiment, epochs, train_losses, cross_validation_losses, dice_scores, x_limits = None, y_limits = None):
    axes = matplotlib.pyplot.figure().gca()
    x_axis = axes.get_xaxis()
    x_axis.set_major_locator(pylab.MaxNLocator(integer = True))
    
    matplotlib.pyplot.plot(epochs, train_losses)
    matplotlib.pyplot.plot(epochs, cross_validation_losses)
    matplotlib.pyplot.plot(epochs, dice_scores)
    matplotlib.pyplot.legend(['Training loss', 'Cross validation loss', 'Dice scores'])
    matplotlib.pyplot.xlabel('Epochs')
    matplotlib.pyplot.ylabel('Loss or Dice score')
    matplotlib.pyplot.title(experiment)
    if x_limits is not None: matplotlib.pyplot.xlim(x_limits)
    if y_limits is not None: matplotlib.pyplot.ylim(y_limits)
    
    output_directory = './results/' + experiment + '/learningCurves/'
    image_file = output_directory + 'learning_curves.png'
    matplotlib.pyplot.tight_layout()
    matplotlib.pyplot.savefig(image_file)

def process_results(experiment, x_limits, y_limits):
    output_directory = './results/' + experiment + '/learningCurves/'
    train_losses = numpy.load(output_directory + 'train_losses.npy')
    cross_validation_losses = numpy.load(output_directory + 'cross_validation_losses.npy')
    dice_scores = numpy.load(output_directory + 'dice_scores.npy')
    epochs = numpy.arange(1, len(train_losses) + 1)

    plot_learning_curves(experiment, epochs, train_losses, cross_validation_losses, dice_scores, x_limits, y_limits)
    training_curves = numpy.column_stack((epochs, train_losses, cross_validation_losses, dice_scores))
    numpy.savetxt(
        output_directory + 'training_curves.csv', 
        training_curves,
        fmt = '%d, %.5f, %.5f, %.5f', 
        header = 'Epochs, Train loss, Cross validation loss, Dice scores'
    )
    
if __name__ == '__main__':
    dice_score_limits = [0.995, 0.997]
    loss_limits = [0.02, 0.08]
    x_limits = [1, 150]
    # Assign either dice_score_limits or loss_limits depending on what you want to focus on.
    y_limits = loss_limits
    # experiments = ['experiment' + str(i) for i in [53, 60, 61]]
    experiments = ['my_solution']
    for experiment in experiments:
        process_results(experiment, x_limits, y_limits)
        
