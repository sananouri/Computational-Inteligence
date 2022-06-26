clc
filename = 'data.txt';
[data,delimiterOut] = importdata(filename);
num_data = length(data); C = 3; time = 1;
num = randi([1 fix(num_data/3)],1,1);
U = [ones(1,num) zeros(1,num_data-num);
     zeros(1,num) ones(1,num) zeros(1,num_data-2*num);
     zeros(1,2*num) ones(1,num_data-2*num)];
center = zeros(C,2);
distance = zeros(C,num_data);
colors = ['g';'r';'b'];
for i = 1:100
  fprintf('Iteration %d\n',i);
  % calculate centers
  for j = 1:C
      center(j,:) = (data' * U(j,:)') ./ sum(U(j,:));
  end
  % plot
  maxU = max(U);
  hold on;
  for j = 1:C
      index = find(U(j,:) == max(U));
      plot(data(index,1),data(index,2),'o','color',colors(j,:));
      plot(center(j,1),center(j,2),'^','color',colors(j,:));
  end
  % calculate distances
  for j = 1:num_data
      for k = 1:C
          distance(k,j) = pdist([data(j,:) ; center(k,:)],'euclidean');
      end
  end
  % update U
  minD = min(distance);
  new_U = U;
  for j = 1:num_data
      for k = 1:C
          new_U(k,j) = (distance(k,j) == minD(1,j));
      end
  end
  if isequal(U,new_U) 
      break
  end
  U = new_U;
  pause(time);
  clf
end