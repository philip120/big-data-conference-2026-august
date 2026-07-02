arr = [64 34 25 12 22 11 90];
n = length(arr);
for i = 1:n-1
    for j = 1:n-i
        if arr(j) > arr(j+1)
            tmp = arr(j);
            arr(j) = arr(j+1);
            arr(j+1) = tmp;
        end
    end
end
disp('Sorted array:')
disp(arr)
