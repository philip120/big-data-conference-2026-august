data = [3 7 2 9 4 6 1 8 5 10];
total = 0;
for i = 1:length(data)
    total = total + data(i);
    fprintf('After %d values, mean = %.2f\n', i, total/i)
end
