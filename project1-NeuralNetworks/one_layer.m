
f = @(x,y) (x*x + y*y) * humps(x);

data_x = abs(rands(1,100));
data_y = abs(rands(1,100));

eta = 0.1; %learning rate
epsilon = 0.1; %error limit

W1 = abs(rands(10,2)); %first layer 2 inputs 10 neurons
Wb1 = abs(rands(10,1)); %first layer bios

W2 = abs(rands(1,10)); %second layer 10 inputs 1 neuson
Wb2 = abs(rands(1,1)); %second layer bios

%70 percent learn data
learn_x(1:70) = data_x(1:70);
learn_y(1:70) = data_y(1:70);
learn_data = [learn_x;learn_y];
learn_target = 1:70;
for i = 1:70
    learn_target(i) = f(learn_x(i), learn_y(i));
end 
learn_target = learn_target / max(learn_target(:)); %normalization

%10 percent validation data
valid_x(1:10) = data_x(71:80);
valid_y(1:10) = data_y(71:80);
valid_data = [valid_x;valid_y];
valid_target = 1:10;
for i = 1:10
    valid_target(i) = f(valid_x(i), valid_y(i));
end 
valid_target = valid_target / max(valid_target(:)); %normalization

%20 percent test data
test_x(1:20) = data_x(81:100);
test_y(1:20) = data_y(81:100);
test_data = [test_x;test_y];
test_target = 1:20;
for i = 1:20
    test_target(i) = f(test_x(i), test_y(i));
end 
test_target = test_target / max(test_target(:)); %normalization

k = 0;
E_valid = 10;
error_valid = 1:10;
error_learn = 1:10;
while (k < 10 && E_valid > epsilon)
    k = k + 1;
    for i = 1:70
        %feedforward
        net1 = W1 * learn_data(:,i) + Wb1;
        O1 = tansig(net1);
        diff_O1 = 4 * exp(-2 * net1) ./ (1 + exp(-2 * net1)).^2;
        net2 = W2 * O1 + Wb2;
        O2 = net2;
        error = learn_target(i) - O2;
        
        %backpropagation
        W2 = W2 + eta * error * O1';
        Wb2 = Wb2 + eta * error;
        delta = (W2' * error).*diff_O1;
        W1 = W1 + eta * delta * learn_data(:,i)';
        Wb1 = Wb1 + eta * delta;
    end
    
    %validation error
    WB1 = ones(10,100);
    for i = 1:10
        WB1(i,:) = Wb1(i,1)*ones(1,100);
    end
    net1_valid = W1 * valid_data + WB1(:,1:10);
    O1_valid = tansig(net1_valid);
    WB2 = Wb2 * ones(1,10);
    net2_valid = W2 * O1_valid + WB2;
    O2_valid = net2_valid;
    e_valid = valid_target - O2_valid;
    E_valid = 0.5 * trace(e_valid * e_valid');
    error_valid(k) = E_valid;
    
    %learn error
    net1_learn = W1 * learn_data + WB1(:,1:70);
    O1_learn = tansig(net1_learn);
    WB2 = Wb2 * ones(1,70);
    net2_learn = W2 * O1_learn + WB2;
    O2_learn = net2_learn;
    e_learn = learn_target - O2_learn;
    E_learn = 0.5 * trace(e_learn * e_learn');
    error_learn(k) = E_learn;
end

i = 1:length(error_learn);
figure; plot(i,error_learn,'g');
hold on;
plot(i,error_valid,'b');
xlabel('epoch');
ylabel('learning error(green) and validation error(blue)');

%test error
net1_test = W1 * test_data + WB1(:,1:20);
O1_test = tansig(net1_test);
WB2 = Wb2 * ones(1,20);
net2_test = W2 * O1_test + WB2;
O2_test = net2_test;
e_test = test_target - O2_test;
E_test = 0.5 * trace(e_test * e_test');
E_test
