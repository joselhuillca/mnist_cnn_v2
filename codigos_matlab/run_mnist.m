

% -- settings start here ---
% set 1 to use gpu, and 0 to use cpu
use_gpu =  1;

% top K returned images
top_k = 5000;
feat_len = 48;

% set result folder
result_folder = './analysis';

map_file = sprintf('%s/map.txt', result_folder);
precision_file = sprintf('%s/precision-at-k.txt', result_folder);
M = csvread('mnist-data/output_Train_AE1_fc1_mnist64.csv');
N = csvread('mnist-data/output_Test_AE1_fc1_mnist64.csv');


dim = size(N,2);

mnist_train_data = N(:, [1:dim-1])';
mnist_train_label = N(:, [dim]);

mnist_test_data = M(:, [1:dim-1])';
mnist_test_label = M(:, [dim]);

binary_train = ( ( (mnist_train_data))) > 0.5;
binary_test = ( ( (mnist_test_data))) > 0.5;
%binary_train = mnist_train_data;
%binary_test = mnist_test_data;

[map, precision_at_k] = precision(mnist_train_label, binary_train, mnist_test_label, binary_test, top_k, 1);

fprintf('MAP = %f\n',map);
save(map_file, 'map', '-ascii');
P = [[1:1:top_k]' precision_at_k'];
save(precision_file, 'P', '-ascii');


