
f = @(x,y) (x*x + y*y) * humps(x);

data_x = abs(rands(1,100));
data_y = abs(rands(1,100));

eta = 0.0001; %learning rate

W1 = abs(rands(10,2)); %first layer 2 inputs 10 neurons
Wb1 = abs(rands(10,1)); %first layer bios
WB1 = ones(10,100);
for i = 1:10
    WB1(i,:) = Wb1(i,1)*ones(1,100);
end

W2 = abs(rands(5,10)); %second layer 10 inputs 5 neurons
Wb2 = abs(rands(5,1)); %second layer bios
WB2 = ones(5,100);
for i = 1:5
    WB2(i,:) = Wb2(i,1)*ones(1,100);
end

W3 = abs(rands(1,5)); %third layer 5 inputs 1 neuson
Wb3 = abs(rands(1,1)); %third layer bios
WB3 = ones(1,100);
WB3(1,:) = Wb2(1,1)*ones(1,100);

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

error_valid = 1:50;
error_learn = 1:50;
for k = 1:50
    %feedforward
    net1 = W1 * learn_data + WB1(:,1:70);
    O1 = tansig(net1);
    net2 = W2 * O1 + WB2(:,1:70);
    O2 = logsig(net2);
    net3 = W3 * O2 + WB3(:,1:70);
    O3 = net3;
    
    diff_O1 = 4 * exp(-2 * net1) ./ (1 + exp(-2 * net1)).^2;
    diff_O2 = exp(-net2) ./ (1 + exp(-net2)).^2;   

    e_learn = learn_target - O3;
    E_Learn = 0.5 * trace(e_learn * e_learn');
    error_learn(k) = E_Learn / 70;
 
    %validation
    net1_valid = W1 * valid_data + WB1(:,1:10);
    O1_valid = tansig(net1_valid);
    net2_valid = W2 * O1_valid + WB2(:,1:10);
    O2_valid = logsig(net2_valid);
    net3_valid = W3 * O2_valid + WB3(:,1:10);
    O3_valid = net3_valid;
    e_valid = valid_target - O3_valid;
    E_valid = 0.5 * trace(e_valid * e_valid');
    error_valid(k) = E_valid / 10;
    
    %backpropagation
    W3 = W3 + eta * e_learn * O2';
    Wb3 = Wb3 + eta * e_learn * ones(70,1);
    WB3(1,:)=Wb3(1,1);
    delta2 = (W3' * e_learn).*diff_O2;
    W2 = W2 + eta * delta2 * O1';
    Wb2 = Wb2 + eta * (W3' * e_learn) * ones(70,1);
    for i = 1:5
        WB2(i,:) = Wb2(i,1);
    end
    delta1 = (W2' * delta2).*diff_O1;
    W1 = W1 + eta * delta1 * learn_data';
    Wb1 = Wb1 + eta * (W2' * delta2) * ones(70,1);
    for i = 1:10
        WB1(i,:) = Wb1(i,1);
    end
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
net2_test = W2 * O1_test + WB2(:,1:20);
O2_test = logsig(net2_test);
net3_test = W3 * O2_test + WB3(:,1:20);
O3_test = net3_test;
e_test = test_target - O3_test;
E_test = 0.5 * trace(e_test * e_test') / 20;
E_test
