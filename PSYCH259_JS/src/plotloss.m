training_loss = textread('../results/training_loss.txt');
training_error = textread('../results/training_error.txt');
testing_error = textread('../results/testing_error.txt');
x1 = [1:28]*200;
x2 = [1:5440];

figure % new figure
[hAx,hLine1,hLine2] = plotyy(x2, training_loss, x1, 1-testing_error);
title('RNN Training Over 2 Epochs')
xlabel('Number of Iterations')
hLine2.LineStyle = '--';
ylabel(hAx(1),'Loss') % left y-axis 
ylabel(hAx(2),'Error') % right y-axis
hold on
plot(x1, 1- training_error, 'g--');
grid;
legend('training loss','testing error','training error');