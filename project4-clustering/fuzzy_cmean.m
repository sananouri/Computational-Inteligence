clc
filename = 'data.txt';
[data,delimiterOut] = importdata(filename);
M = 2; C = 3; time = 0.5; f_old = 10000000;
colors = ['g';'r';'b'];
U = rands(C,length(data));
for i = 1:100
    [U, center, f] = stepfcm(data, U, C, M);
    fprintf('Iteration %d:',i);
    fprintf(' f = %f\n', f);
    maxU = max(U);
    hold on;
    for j = 1:C
        index = find(U(j,:) == max(U));
        plot(data(index,1),data(index,2),'o','color',colors(j,:));
        plot(center(j,1),center(j,2),'^','color',colors(j,:));
    end
    if f_old - f < 1e-4
        break
    end
    f_old = f;
    pause(time);
    clf    
end